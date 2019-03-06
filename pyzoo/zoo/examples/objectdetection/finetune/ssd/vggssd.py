from zoo.models.image.objectdetection.object_detector import *


def getLocConfComponent(prev, name, nInput, nOutput, typeName):
    conv = SpatialConvolution(nInput, nOutput, 3, 3, 1, 1, 1, 1) \
        .set_init_method(weight_init_method=Xavier(), bias_init_method=Zeros()) \
        .set_name("%s_mbox_%s" % (name, typeName))(prev)
    trans = Transpose(((2, 3), (3, 4))) \
        .set_name("%s_mbox_%s_perm" % (name, typeName))(conv)
    return InferReshape((0, -1)).set_name("%s_mbox_%s_flat" % (name, typeName))(trans)


def getPriorBox(conv, name, params):
    param = params(name)
    return PriorBox(min_sizes=param.minSizes, max_sizes=param.maxSizes,
             aspect_ratios=param.aspectRatios, is_flip=param.isFlip, is_clip=param.isClip,
             variances=param.variances, step=param.step, offset=0.5,
             img_h=param.resolution, img_w=param.resolution)\
        .set_name("%s_mbox_priorbox" % name)(conv)

def getConcatOutput(conv, name, params, nInput, num_classes):
    con_flat = getLocConfComponent(conv, name, nInput, params(name).nPriors * num_classes, "conf")
    loc_flat = getLocConfComponent(conv, name, nInput, params(name).nPriors * 4, "loc")
    prior_box = getPriorBox(conv, name, params)
    return (con_flat, loc_flat, prior_box)

def ssg_graph(num_classes, resolution, input, base_part1, base_part2,
              params, is_last_pool, norm_scale, param, w_regularizer=None,
              b_regularizer=None):
    if w_regularizer is None:
        _w_regularizer = L2Regularizer(0.0005)
    else:
        _w_regularizer = w_regularizer

    conv4_3_norm = NormalizeScale(2, scale=norm_scale,
                                  size=(1, params("conv4_3_norm").nInput, 1, 1),
                                  w_regularizer=L2Regularizer(0.0005)) \
        .set_name("conv4_3_norm")(base_part1)

    base2 = base_part2
    fc6 = SpatialDilatedConvolution(params("fc7").nInput, 1024, 3, 3, 1, 1, 6, 6, 6, 6) \
        .set_init_method(weight_init_method=Xavier(), bias_init_method=Zeros()) \
        .set_name("fc6")(base2)

    relu6 = ReLU(True).set_name("relu6")(fc6)
    fc7 = SpatialConvolution(1024, 1024, 1, 1) \
        .set_init_method(weight_init_method=Xavier(), bias_init_method=Zeros()) \
        .set_name("fc7")(relu6)
    relu7 = ReLU(True).set_name("relu7")(fc7)

    norm4_3_out = getConcatOutput(conv4_3_norm, "conv4_3_norm", params,
                                 params("conv4_3_norm").nInput, num_classes)
    fc7_out = getConcatOutput(relu7, "fc7", params, 1024, num_classes)

    relu6_1 = add_conv_relu(relu7, 1024, 256, 1, 1, 0, "6_1")
    relu6_2 = add_conv_relu(relu6_1, 256, 512, 3, 2, 1, "6_2")
    c6_out = getConcatOutput(relu6_2, "conv6_2", params, params("conv6_2").nInput, num_classes)

    relu7_1 = add_conv_relu(relu6_2, 512, 128, 1, 1, 0, "7_1")
    relu7_2 = add_conv_relu(relu7_1, 128, 256, 3, 2, 1, "7_2")
    c7_out = getConcatOutput(relu7_2, "conv7_2", params, params("conv7_2").nInput, num_classes)

    relu8_1 = add_conv_relu(relu7_2, 256, 128, 1, 1, 0, "8_1")
    if (is_last_pool or resolution == 512):
        relu8_2 = add_conv_relu(relu8_1, 128, 256, 3, 2, 1, "8_2")
    else:
        relu8_2 = add_conv_relu(relu8_1, 128, 256, 3, 1, 0, "8_2")

    c8_out = getConcatOutput(relu8_2, "conv8_2", params, params("conv8_2").nInput, num_classes)
    if is_last_pool:
        # addFeatureComponentPool6(com8, params, num_classes)
        pool6 = SpatialAveragePooling(3, 3).set_name("pool6")(relu8_2)
        (c9_out, relu9_2) = (getConcatOutput(pool6, "pool6", params, params("pool6").nInput, num_classes), pool6)
    else:
        relu9_1 = add_conv_relu(relu8_2, 256, 128, 1, 1, 0, "9_1")
        if (resolution == 512):
             relu9_2 = add_conv_relu(relu9_1, 128, 256, 3, 2, 1, "9_2")
        else:
            relu9_2 = add_conv_relu(relu9_1, 128, 256, 3, 1, 0, "9_2")
        (c9_out, relu9_2) = (getConcatOutput(relu9_2, "conv9_2", params, params("conv9_2").nInput, num_classes), relu9_2)


    if (resolution == 512):
        relu10_1 = add_conv_relu(relu9_2, (256, 128, 1, 1, 0), "10_1")
        relu10_2 = add_conv_relu(relu10_1, 128, 256, 4, 1, 1, "10_2")
        c10_out = getConcatOutput(relu10_2, "conv10_2", params, params("conv10_2").nInput, num_classes)
    else:
        c10_out = None

    if (resolution == 300):
        conf = JoinTable(1, 1)(norm4_3_out._1, fc7_out._1, c6_out[0], c7_out[0], c8_out[0], c9_out[0])
        loc = JoinTable(1, 1)(norm4_3_out._2, fc7_out._2, c6_out._2, c7_out._2, c8_out._2, c9_out._2)
        priors = JoinTable(2, 2)(norm4_3_out._3, fc7_out._3, c6_out._3, c7_out._3, c8_out._3, c9_out._3)
    else:
        conf = JoinTable(1, 1)\
            (norm4_3_out._1, fc7_out._1, c6_out._1, c7_out._1, c8_out._1, c9_out._1, c10_out._1)
        loc = JoinTable(1, 1)\
            (norm4_3_out._2, fc7_out._2, c6_out._2, c7_out._2, c8_out._2, c9_out._2, c10_out._2)
        priors = JoinTable(2, 2)\
            (norm4_3_out._3, fc7_out._3, c6_out._3, c7_out._3, c8_out._3, c9_out._3, c10_out._3)


    model = Model(input, (loc, conf, priors))
    model.value.setScaleB(2)
#     stopGradient(model)
# val
# ssd = Sequential[T]()
# ssd.add(model)
# ssd.add(DetectionOutputSSD[T](param))
# setRegularizer(model, _wRegularizer, bRegularizer)
# ssd

