from zoo.models.image.objectdetection.object_detector import *


rpn_pre_nms_top_n_test = 6000
rpn_post_nms_top_n_test = 300
debug = False
ratios = (0.5, 1.0, 2.0)
scales = (8.0, 16.0, 32.0)
anchor_num = len(ratios) * len(scales)
skip_layers = ("rpn-data", "roi-data", "proposal", "roi", "conv3_1")

def max_pooling_ceil(pool):
    callJavaFunc(pool.value.ceil)
    return pool


def vgg16(input):
    conv1_1 = SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1, propagate_back = False)\
        .set_init_method(weight_init_method = Xavier(), bias_init_method = Zeros())\
        .set_name("conv1_1")(input)
    relu1_1 = ReLU(True).set_name("relu1_1")(conv1_1)
    relu1_2 = add_conv_relu(relu1_1, 64, 64, 3, 1, 1, "1_2")
    pool1 = max_pooling_ceil(SpatialMaxPooling(2, 2, 2, 2)).set_name("pool1")(relu1_2)

    relu2_1 = add_conv_relu(pool1, 64, 128, 3, 1, 1, "2_1")
    relu2_2 = add_conv_relu(relu2_1, 128, 128, 3, 1, 1, "2_2")
    pool2 = max_pooling_ceil(SpatialMaxPooling(2, 2, 2, 2)).set_name("pool2")(relu2_2)

    relu3_1 = add_conv_relu(pool2, 128, 256, 3, 1, 1, "3_1")
    relu3_2 = add_conv_relu(relu3_1, 256, 256, 3, 1, 1, "3_2")
    relu3_3 = add_conv_relu(relu3_2, 256, 256, 3, 1, 1, "3_3")
    pool3 = max_pooling_ceil(SpatialMaxPooling(2, 2, 2, 2)).set_name("pool3")(relu3_3)

    relu4_1 = add_conv_relu(pool3, 256, 512, 3, 1, 1, "4_1")
    relu4_2 = add_conv_relu(relu4_1, 512, 512, 3, 1, 1, "4_2")
    relu4_3 = add_conv_relu(relu4_2, 512, 512, 3, 1, 1, "4_3")
    pool4 = max_pooling_ceil(SpatialMaxPooling(2, 2, 2, 2)).set_name("pool4")(relu4_3)

    relu5_1 = add_conv_relu(pool4, 512, 512, 3, 1, 1, "5_1")
    relu5_2 = add_conv_relu(relu5_1, 512, 512, 3, 1, 1, "5_2")
    relu5_3 = add_conv_relu(relu5_2, 512, 512, 3, 1, 1, "5_3")
    return relu5_3


def vggfrcnn(class_num, post_process_param):
    data = Input()
    imInfo = Input()
    gt = Input()

    vgg = vgg16(data)
    rpn_conv_3x3 = SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)\
        .set_name("rpn_conv/3x3")(vgg)
    relu3x3 = ReLU(True).set_name("rpn_relu/3x3")(rpn_conv_3x3)
    rpn_cls_score = SpatialConvolution(512, 18, 1, 1, 1, 1)\
        .set_name("rpn_cls_score")(relu3x3)
    rpn_cls_score_reshape = InferReshape((0, 2, -1, 0))\
        .set_name("rpn_cls_score_reshape")(rpn_cls_score)
    rpn_cls_prob = SoftMax().set_name("rpn_cls_prob")(rpn_cls_score_reshape)
    rpn_cls_prob_reshape = InferReshape((1, 2 * anchor_num, -1, 0))\
        .set_name("rpn_cls_prob_reshape")(rpn_cls_prob)
    rpn_bbox_pred = SpatialConvolution(512, 36, 1, 1, 1, 1)\
        .set_name("rpn_bbox_pred")(relu3x3)
    proposal = Proposal(rpn_pre_nms_top_n_test, rpn_post_nms_top_n_test, ratios, scales)\
        .set_name("proposal")([rpn_cls_prob_reshape, rpn_bbox_pred, imInfo])

    roi_data = ProposalTarget(128, class_num).set_name("roi-data").set_debug(debug)([proposal, gt])
    roi = SelectTable(1).set_name("roi")(roi_data)
    pool = 7
    roi_pooling = RoiPooling(pool, pool, 0.0625)\
        .set_name("pool5")([vgg, roi])
    reshape = InferReshape((-1, 512 * pool * pool))\
        .set_name("pool5_reshape")(roi_pooling)
    fc6 = Linear(512 * pool * pool, 4096).set_name("fc6")(reshape)
    relu6 = ReLU()(fc6)
    dropout6 = Dropout().set_name("drop6")(relu6)
    if not debug:
        fc7 = Linear(4096, 4096).set_name("fc7")(dropout6)
    else:
        fc7 = Linear(4096, 4096).set_name("fc7")(relu6)
    relu7 = ReLU()(fc7)
    dropout7 = Dropout().set_name("drop7")(relu7)
    if not debug:
        cls_score = Linear(4096, class_num).set_name("cls_score")(dropout7)
    else:
        cls_score = Linear(4096, class_num).set_name("cls_score")(relu7)

    softmax = SoftMax().set_name("cls_prob")
    evaluate_only = EvaluateOnly(softmax)
    evaluate_only.add(softmax)
    cls_prob = evaluate_only(cls_score)

    if not debug:
        bbox_pred = BboxPred(4096, class_num * 4, n_class = class_num).set_name("bbox_pred")(dropout7)
    else:
        bbox_pred = BboxPred(4096, class_num * 4, n_class = class_num).set_name("bbox_pred")(relu7)

    # Training part
    rpn_data = AnchorTarget(ratios, scales)\
        .set_name("rpn-data")\
        .set_debug(debug)([rpn_cls_score, gt, imInfo, data])

    detectionOut = DetectionOutputFrcnn(post_process_param.nms_thresh, post_process_param.n_classes,
      post_process_param.bbox_vote, post_process_param.max_per_image, post_process_param.thresh)(
      [imInfo, roi_data, bbox_pred, cls_prob,
      rpn_cls_score_reshape, rpn_bbox_pred, rpn_data])
    vggfrcnn = Model([data, imInfo, gt], detectionOut)
    vggfrcnn.value.setScaleB(2.0)
    vggfrcnn.stop_gradient(skip_layers)
    return vggfrcnn
  