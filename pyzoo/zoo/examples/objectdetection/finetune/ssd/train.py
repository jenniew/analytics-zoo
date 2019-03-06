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
from zoo.models.image.objectdetection import *

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
parser.add_option("--preTrainModel", dest="pretrain_model", default="./",
                  help="pretrain model location")
parser.add_option("--state", dest="state_snapshot", help="state snapshot location")
parser.add_option("--checkpoint", dest="checkpoint", help="where to cache the model")
parser.add_option("--checkIter", type=int, dest="check_iter", default=200,
                  help="checkpoint iteration")
parser.add_option("--overwrite", dest="overwrite_checkpoint", default=False,
                  help="overwrite checkpoint files")
parser.add_option("-e", "--maxEpoch", type=int, dest="max_epoch", default=20, help="epoch number")
parser.add_option("-l", "--learningRate", type=float, dest="learning_rate", default=0.001,
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


def train(args):
    class_names = [line.rstrip('\n') for line in args.class_name_file]

    # prepare pre and post params
    pre_param_val = PreProcessParam(args.batch_size, n_partition=args.batch_size)

    # load pretrained model
    pretrain = Model.load(args.pretrain_model)
    model = vggfrcnn(len(class_names), post_param)
    load_model_weights(pretrain, model, False)

    # load train data
    train_data = roi_seq_files_to_image_frame(args.train_foler, sc, args.n_partition)
    train_transformer = Pipeline(BytesToMat(),
                                 RoiNormalize(),
                                 ColorJitter(),
                                 RandomTransformer(Pipeline(Expand(), RoiProject()), 0.5),
                                 RandomSampler(),
                                 Resize(args.resolution, args.resolution, -1),
                                 RandomTransformer(Pipeline(HFlip(), RoiHFlip()), 0.5),
                                 ChannelNormalize(123, 117, 104),
                                 MatToFloats(valid_height=args.resolution, valid_width=args.resolution),
                                 RoiImageToBatch(args.batch_size))
    train_set = DataSet.image_frame(train_data).transform(train_transformer)

    # load val data
    val_data = roi_seq_files_to_image_frame(args.val_foler, sc, args.n_partition)
    val_transformer = Pipeline(BytesToMat(),
                               RoiNormalize(),
                               Resize(args.resolution, args.resolution, -1),
                               ChannelNormalize(123, 117, 104),
                               MatToFloats(valid_height=100, valid_width=100),
                               RoiImageToBatch(args.batch_size))
    val_set = DataSet.image_frame(val_data).transform(val_transformer)

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

    model.saveModel("./final.model")


if __name__ == "__main__":
    (options, args) = parser.parse_args(sys.argv)
    sc = init_nncontext(args.job_name)
    train(args)
