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

from bigdl.nn.layer import Model
from bigdl.optim.optimizer import *
from zoo.common.nncontext import init_nncontext
from zoo.examples.objectdetection.finetune.fastrcnn import vggfrcnn
from zoo.models.image.objectdetection.ssd import *
from zoo.models.image.objectdetection.object_detector import *
from zoo.feature.common import *


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
parser.add_option("-r", "--resolution", type=int, dest="resolution", default=300,
                  help="input resolution 300 or 512")
parser.add_option("--model", dest="model_snapshot", help="model snapshot location")
parser.add_option("--dataset", dest="dataset", default="pascal",
                  help="which dataset of the model will be used")
parser.add_option("--state", dest="state_snapshot", help="state snapshot location")
parser.add_option("--checkpoint", dest="checkpoint", help="where to cache the model")
parser.add_option("--overwriteCheckpoint", dest="overwrite_checkpoint", default=False,
                  help="whether to overwrite checkpoint files")
parser.add_option("-e", "--maxEpoch", type=int, dest="max_epoch", default=20, help="epoch number")
parser.add_option("-l", "--learningRate", type=float, dest="learning_rate", default=0.0001,
                  help="initial learning rate")
parser.add_option("--learningRateDecay", type=float, dest="learning_rate_decay", default=0.0005,
                  help="learning rate decay")
parser.add_option("--class", dest="class_name_file", default="", help="class name file")
parser.add_option("-b", "--batchSize", type=int, dest="batch_size", default=4,
                  help="batch size, has to be same with total cores")
parser.add_option("--name", dest="job_name", default="Analytics Zoo SSD Fine Tune Example",
                  help="job name")
parser.add_option("--summary", dest="summary_dir", help="train validate summary directory")
parser.add_option("-p", "--partition", type=int, dest="n_partition", default=1,
                  help="number of partitions")
parser.add_option("--saveModelPath", dest="save_model_path", default="./final.model",
                    help="where to save trained model")
parser.add_option("--overwriteModel", dest="overwrite_model", default=False,
                  help="whether to overwrite model file")


def train(args):
    class_names = [line.rstrip('\n') for line in args.class_name_file]

    # load train data
    train_data = load_roi_seq_files(args.train_foler, sc, args.n_partition)
    train_transformer = ChainedPreprocessing([RoiRecordToFeature(),
                                              ImageBytesToMat(),
                                              ImageRoiNormalize(),
                                              ImageColorJitter(),
                                              ImageRandomPreprocessing(
                                                  ChainedImagePreprocessing([ImageExpand(), ImageRoiProject()]), 0.5),
                                              ImageRandomSampler(),
                                              ImageResize(args.resolution, args.resolution, -1),
                                              ImageRandomPreprocessing(
                                                  ChainedImagePreprocessing([ImageHFlip(), ImageRoiHFlip()]), 0.5),
                                              ImageChannelNormalize(123, 117, 104),
                                              ImageMatToFloats(valid_height=args.resolution,
                                                               valid_width=args.resolution),
                                              RoiImageToSSDBatch(args.batch_size)])
    train_set = FeatureSet.rdd(train_data).transform(train_transformer)

    # load val data
    val_data = load_roi_seq_files(args.val_foler, sc, args.n_partition)
    val_transformer = ChainedPreprocessing([RoiRecordToFeature(),
                                            ImageBytesToMat(),
                                            ImageRoiNormalize(),
                                            ImageResize(args.resolution, args.resolution, -1),
                                            ImageChannelNormalize(123, 117, 104),
                                            ImageMatToFloats(valid_height=args.resolution, valid_width=args.resolution),
                                            RoiImageToSSDBatch(args.batch_size)])
    val_set = FeatureSet.rdd(val_data).transform(val_transformer)

    # load pretrained model
    model = SSDVGG(len(class_names), args.resolution, args.dataset)
    pretrained = ImageModel._do_load(args.model_snapshot)
    load_model_weights(pretrained, model, False)

    # optim method
    if args.state_snapshot:
        optim = OptimMethod.load(args.state_snapshot)
    else:
        optim = Adam(learningrate=args.learning_rate, learningrate_decay=args.learning_rate_decay)

    # validation method
    val_method = MeanAveragePrecision(use_07_metric=True, normalized=False, class_names=class_names)

    # optimizer
    optimizer = Optimizer.create(model=model,
                                 training_set=train_set,
                                 criterion=MultiBoxLoss(MultiBoxLossParam(n_classes=len(class_names))),
                                 optim_method=optim,
                                 end_trigger=MaxEpoch(args.max_epoch))

    if args.checkpoint:
        optimizer.set_checkpoint(checkpoint_path=args.checkpoint,
                                 checkpoint_trigger=EveryEpoch(),
                                 isOverWrite=args.overwrite_checkpoint)

    if args.summary_dir:
        train_summary = TrainSummary(args.summary_dir, args.job_name)
        val_summary = ValidationSummary(args.summary_dir, args.job_name)
        train_summary.set_summary_trigger(name="LearningRate", trigger=SeveralIteration(1))
        optimizer.set_train_summary(train_summary)
        optimizer.set_val_summary(val_summary)

    optimizer.set_validation(val_rdd=val_set, trigger=EveryEpoch(), val_method=val_method)

    optimizer.optimize()

    model.saveModel(args.save_model_path, args.overwrite_model)


if __name__ == "__main__":
    (options, args) = parser.parse_args(sys.argv)
    # initial zoo context
    sc = init_nncontext(args.job_name)
    train(args)
