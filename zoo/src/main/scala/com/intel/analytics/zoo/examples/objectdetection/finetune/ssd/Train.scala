/*
 * Copyright 2018 Analytics Zoo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.zoo.examples.objectdetection.finetune.ssd

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.MiniBatch
import com.intel.analytics.bigdl.optim.{Optimizer, _}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.LoggerFilter
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.{DistributedFeatureSet, FeatureSet}
import com.intel.analytics.zoo.models.image.common.ImageModel
import com.intel.analytics.zoo.models.image.objectdetection.common.ModuleUtil
import com.intel.analytics.zoo.models.image.objectdetection.common.evaluation.MeanAveragePrecision
import com.intel.analytics.zoo.models.image.objectdetection.common.loss.{MultiBoxLoss, MultiBoxLossParam}
import com.intel.analytics.zoo.models.image.objectdetection.ssd.{SSDDataSet, SSDMiniBatch, SSDVGG}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import scopt.OptionParser

import scala.io.Source

object Train {

  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  case class TrainParams(
    trainFolder: String = "./",
    valFolder: String = "./",
    resolution: Int = 300,
    dataset: String = "pascal",
    checkpoint: Option[String] = None,
    modelSnapshot: Option[String] = None,
    stateSnapshot: Option[String] = None,
    className: String = "",
    batchSize: Int = 4,
    learningRate: Double = 0.0001,
    learningRateDecay: Double = 0.0005,
    overWriteCheckpoint: Boolean = false,
    maxEpoch: Int = 20,
    jobName: String = "Analytics Zoo SSD Train Messi Example",
    summaryDir: Option[String] = None,
    nPartition: Option[Int] = None,
    saveModelPath: String = "./final.model",
    overWriteModel: Boolean = false
  )

  val trainParser = new OptionParser[TrainParams]("Analytics Zoo SSD Example") {
    opt[String]('f', "trainFolder")
      .text("url of hdfs folder store the train hadoop sequence files")
      .action((x, c) => c.copy(trainFolder = x))
    opt[String]('v', "valFolder")
      .text("url of hdfs folder store the validation hadoop sequence files")
      .action((x, c) => c.copy(valFolder = x))
      opt[Int]('r', "resolution")
      .text("input resolution 300 or 512")
      .action((x, c) => c.copy(resolution = x))
      .required()
    opt[String]('d', "dataset")
      .text("which dataset of the model will be used")
      .action((x, c) => c.copy(dataset = x))
    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelSnapshot = Some(x)))
      .required()
    opt[String]("state")
      .text("state snapshot location")
      .action((x, c) => c.copy(stateSnapshot = Some(x)))
    opt[String]("checkpoint")
      .text("where to cache the model")
      .action((x, c) => c.copy(checkpoint = Some(x)))
    opt[Int]('e', "maxEpoch")
      .text("epoch numbers")
      .action((x, c) => c.copy(maxEpoch = x))
      .required()
    opt[Double]('l', "learningRate")
      .text("inital learning rate")
      .action((x, c) => c.copy(learningRate = x))
    opt[Double]("learningRateDecay")
      .text("learning rate decay")
      .action((x, c) => c.copy(learningRateDecay = x))
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
    opt[String]("class")
      .text("class file")
      .action((x, c) => c.copy(className = x))
      .required()
    opt[Unit]("overwriteCheckpoint")
      .text("overwrite checkpoint files")
      .action((_, c) => c.copy(overWriteCheckpoint = true))
    opt[String]("name")
      .text("job name")
      .action((x, c) => c.copy(jobName = x))
    opt[String]("summary")
      .text("train validate summary")
      .action((x, c) => c.copy(summaryDir = Some(x)))
    opt[Int]('p', "partition")
      .text("number of partitions")
      .action((x, c) => c.copy(nPartition = Some(x)))
    opt[String]('s', "saveModelPath")
      .text("where to save trained model")
      .action((x, c) => c.copy(saveModelPath = x))
    opt[Unit]("overwriteModel")
      .text("overwrite model file")
      .action((_, c) => c.copy(overWriteModel = true))
  }

  val logger = Logger.getLogger(getClass.getName)

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, TrainParams()).map(param => {
      // initial zoo context
      val conf = new SparkConf().setAppName(param.jobName)
      val sc = NNContext.initNNContext(conf)

      // load data
      val classes = Source.fromFile(param.className).getLines().toArray
      val trainSet = SSDDataSet.loadSSDTrainSet(param.trainFolder, sc, param.resolution,
        param.batchSize, param.nPartition)
      val valSet = SSDDataSet.loadSSDValSet(param.valFolder, sc, param.resolution, param.batchSize,
        param.nPartition)

      // create ssd model and load weights from pretrained model
      val model = SSDVGG[Float](classes.length, param.resolution, param.dataset)
      val m = ImageModel.loadModel(param.modelSnapshot.get, modelType = "objectdetection")
      ModuleUtil.loadModelWeights(m, model, false)

      // create optimizer and optimize
      val optimMethod = if (param.stateSnapshot.isDefined) {
        OptimMethod.load[Float](param.stateSnapshot.get)
      } else {
        new Adam[Float](learningRate = param.learningRate,
          learningRateDecay = param.learningRateDecay)
      }
      optimize(model, trainSet, valSet, param, optimMethod,
        Trigger.maxEpoch(param.maxEpoch), classes)
      model.saveModel(param.saveModelPath, overWrite = param.overWriteModel)
    })
  }

  private def optimize(model: SSDVGG[Float],
                       trainSet: FeatureSet[SSDMiniBatch],
                       valSet: FeatureSet[SSDMiniBatch],
                       param: TrainParams,
                       optimMethod: OptimMethod[Float],
                       endTrigger: Trigger,
                       classes: Array[String]): SSDVGG[Float] = {
    val optimizer = Optimizer(
      model = model,
      dataset = trainSet.toDataSet(),
      criterion = new MultiBoxLoss[Float](MultiBoxLossParam(nClasses = classes.length))
    )

    if (param.checkpoint.isDefined) {
      optimizer.setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
    }

    if (param.overWriteCheckpoint) {
      optimizer.overWriteCheckpoint()
    }

    if (param.summaryDir.isDefined) {
      val trainSummary = TrainSummary(param.summaryDir.get, param.jobName)
      val validationSummary = ValidationSummary(param.summaryDir.get, param.jobName)
      trainSummary.setSummaryTrigger("LearningRate", Trigger.severalIteration(1))
      optimizer.setTrainSummary(trainSummary)
      optimizer.setValidationSummary(validationSummary)
    }
    optimizer
      .setOptimMethod(optimMethod)
      .setValidation(Trigger.everyEpoch,
        valSet.toDataSet().asInstanceOf[DataSet[MiniBatch[Float]]],
        Array(new MeanAveragePrecision(true, normalized = true, classes = classes)))
      .setEndWhen(endTrigger)
      .optimize().asInstanceOf[SSDVGG[Float]]
  }
}
