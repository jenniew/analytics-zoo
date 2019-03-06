#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from optparse import OptionParser

from bigdl.optim.optimizer import *
from zoo.common.nncontext import init_nncontext
from zoo.examples.objectdetection.finetune.fastrcnn.vggfrcnn import vggfrcnn
from zoo.feature.common import BigDLDataSet, ChainedPreprocessing
from zoo.models.image.objectdetection.object_detector import *

# parser = argparse.ArgumentParser()
# parser.add_argument('model_path', help="Path where the model is stored")
# parser.add_argument('model_path', help="Path where the model is stored")
# parser.add_argument('img_path', help="Path where the images are stored")
# parser.add_argument('output_path', help="Path to store the detection results")
# parser.add_argument("--partition_num", type=int, default=1, help="The number of partitions")
parser = OptionParser()
parser.add_option("-f", "--trainFolder", dest="train_folder", default="./",
                  help="url of hdfs folder store the train hadoop sequence files")
parser.add_option("-v", "--valFolder", dest="val_folder", default="./",
                  help="url of hdfs folder store the validation hadoop sequence files")
parser.add_option("--preTrainModel", dest="pretrain_model", default="./",
                  help="pretrain model location")
parser.add_option("--state", dest="state_snapshot", help="state snapshot location")
parser.add_option("--checkpoint", dest="checkpoint", help="where to cache the model")
parser.add_option("--checkIter", type=int, dest="check_iter", default=200,
                  help="checkpoint iteration")
parser.add_option("-e", "--maxEpoch", type=int, dest="max_epoch", default=50, help="epoch number")
parser.add_option("-l", "--learningRate", type=float, dest="learning_rate", default=0.0001,
                  help="initial learning rate")
parser.add_option("--step", type=int, dest="step", default=50000,
                  help="step to decay learning rate")
parser.add_option("--learningRateDecay", type=float, dest="learning_rate_decay", default=0.0005,
                  help="learning rate decay")
parser.add_option("--optim", dest="optim", default="sgd", help="optimization method")
parser.add_option("--class", dest="class_name_file", default="", help="class name file")
parser.add_option("-b", "--batchSize", type=int, dest="batch_size", default=4,
                  help="batch size, has to be same with total cores")
parser.add_option("--name", dest="job_name", default="Analytics Zoo Fasterrcnn Fine Tune Example",
                  help="job name")
parser.add_option("--summary", dest="summary_dir", help="train validate summary directory")
parser.add_option("-p", "--partition", type=int, dest="n_partition", default=1,
                  help="number of partitions")


def train(options):
    with open(options.class_name_file) as fp:
        class_names = [line.rstrip('\n') for line in fp]

    # prepare pre and post params
    post_param = PostProcessParam(len(class_names), False, 0.3, 100, 0.05)
    pre_param_train = PreProcessParam(options.batch_size, (400, 500, 600, 700))
    pre_param_val = PreProcessParam(options.batch_size, n_partition=options.batch_size)

    # load train data
    train_data = load_roi_seq_files(options.train_folder, sc, options.n_partition)
    train_transform = ChainedPreprocessing([ImageBytesToMat(),
                                            ImageRoiNormalize(),
                                            ImageRandomAspectScale(pre_param_train.scales,
                                                                   pre_param_train.scale_multiple_of),
                                            ImageRoiResize(),
                                            ImageRandomPreprocessing(
                                                ChainedImagePreprocessing([ImageHFlip(),ImageRoiHFlip(False)]), 0.5),
                                            ImageChannelNormalize(pre_param_train.pixel_mean_rgb[0],
                                                                  pre_param_train.pixel_mean_rgb[1],
                                                                  pre_param_train.pixel_mean_rgb[2]),
                                            ImageMatToFloats(valid_height=600, valid_width=600),
                                            FrcnnToBatch(options.batch_size, True)])
    train_set = BigDLDataSet.rdd(train_data).transform(train_transform)

    # load val data
    val_data = load_roi_seq_files(options.val_folder, sc, options.n_partition)
    val_transformer = ChainedPreprocessing([ImageBytesToMat(),
                                            ImageRoiNormalize(),
                                            ImageAspectScale(pre_param_val.scales[0],
                                                             pre_param_val.scale_multiple_of),
                                            ImageChannelNormalize(pre_param_val.pixel_mean_rgb[0],
                                                                  pre_param_val.pixel_mean_rgb[1],
                                                                  pre_param_val.pixel_mean_rgb[2]),
                                            ImageMatToFloats(valid_height=100, valid_width=100),
                                            FrcnnToBatch(options.batch_size, True)])
    val_set = BigDLDataSet.rdd(val_data).transform(val_transformer)

    # optim method
    if options.state_snapshot:
        optim = OptimMethod.load(options.state_snapshot)
    else:
        if options.optim == "sgd":
            learning_rate_schedule = Step(options.step, 0.1)
            optim = SGD(learningrate=options.learning_rate, momentum=0.9, dampening=0.0,
                        leaningrate_schedule=learning_rate_schedule, weightdecay=0.0005)
        elif options.optim == "adam":
            optim = Adam(learningrate=options.learning_rate, learningrate_decay=options.learning_rate_decay)

    # validation method
    val_method = MeanAveragePrecision(use_07_metric=True, classes=class_names, normalized=False)



    # load pretrained model
    model = vggfrcnn(len(class_names), post_param)
    pretrain = Model.loadModel(options.pretrain_model)
    # model = vggfrcnn(len(class_names), post_param)
    load_model_weights(pretrain, model, False)

    # load train data
    # train_data = load_roi_seq_files(options.train_foler, sc, options.n_partition)
    # train_transform = ChainedPreprocessing([ImageBytesToMat(),
    #                                         ImageRoiNormalize(),
    #                                         ImageRandomAspectScale(pre_param_train.scales,
    #                                                                pre_param_train.scale_multiple_of),
    #                                         ImageRoiResize(),
    #                                         ImageRandomPreprocessing(
    #                                             ChainedPreprocessing([ImageHFlip(),
    #                                                                   ImageRoiHFlip(False)]), 0.5),
    #                                         ImageChannelNormalize(pre_param_train.pixel_mean_rgb[0],
    #                                                               pre_param_train.pixel_mean_rgb[1],
    #                                                               pre_param_train.pixel_mean_rgb[3]),
    #                                         ImageMatToFloats(valid_height=600, valid_width=600),
    #                                         FrcnnToBatch(options.batch_size, True)])
    # train_set = DataSet.rdd(train_data).transform(train_transform)


    # optimizer
    optimizer = Optimizer.create(model=model,
                                 training_set=train_set,
                                 criterion=FrcnnCriterion(),
                                 optim_method=optim,
                                 end_trigger=MaxEpoch(options.max_epoch))

    if options.checkpoint:
        optimizer.set_checkpoint(checkpoint_path=options.checkpoint,
                                 checkpoint_trigger=SeveralIteration(options.check_iter),
                                 isOverWrite=True)

    if options.summary_dir:
        train_summary = TrainSummary(options.summary_dir, options.job_name)
        val_summary = ValidationSummary(options.summary_dir, options.job_name)
        train_summary.set_summary_trigger(name="LearningRate", trigger=SeveralIteration(1))
        optimizer.set_train_summary(train_summary)
        optimizer.set_val_summary(val_summary)

    optimizer.set_validation(val_rdd=val_set, trigger=SeveralIteration(options.check_iter), val_method=val_method)

    optimizer.optimize()

    model.saveModel("./final.model")


if __name__ == "__main__":
    (options, args) = parser.parse_args(sys.argv)
    sc = init_nncontext(options.job_name)
    train(options)
