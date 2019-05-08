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

package com.intel.analytics.zoo.models.image.objectdetection.common.dataset

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.zoo.feature.image.{ImageSet, LocalImageSet}
import com.intel.analytics.zoo.models.image.objectdetection.common.dataset.OpenImages.{loadLabelName, loadLableMap}
import org.apache.log4j.Logger

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

/**
  * Parse the open image dataset, load images and annotations
  *
  * @param imageSet   train, validation, test, etc
  * @param devkitPath dataset folder
  */
class OpenImages(val imageSet: String, devkitPath: String) extends Imdb {

  override def getRoidb(readImage: Boolean = true): LocalImageSet = {
    val annotationFileName = imageSet + "-annotations-bbox.csv"
    val src = Source.fromFile(devkitPath + "/" + annotationFileName)
    val content = src.getLines().drop(1).map(_.split(","))
    val validBoxes = new ArrayBuffer[Float]()
    val validClasses = new ArrayBuffer[Float]()
    val validDifficults = new ArrayBuffer[Float]()
    var preImageID = ""
    var image: Array[Byte] = null
    var imagePath: String = ""
    val array = new ArrayBuffer[ImageFeature]()
    while (content.hasNext) {
      val record = content.next
      val imageID = record(0)
      if (!imageID.equals(preImageID)) {
        if (!preImageID.isEmpty) {
          // create image feature for the previous image
          val label = OpenImages.createRoiLabel(validClasses, validDifficults, validBoxes)
          array.append(ImageFeature(image, label, imagePath))
        }
        // add new image
        imagePath = devkitPath + "/images/" + imageID + ".jpg"
        image = if(readImage) loadImage(imagePath) else null
        preImageID = imageID
        validBoxes.clear()
        validClasses.clear()
        validDifficults.clear()
      }
      val x1 = record(4).toFloat
      val y1 = record(6).toFloat
      val x2 = record(5).toFloat
      val y2 = record(7).toFloat

      validBoxes.append(x1)
      validBoxes.append(y1)
      validBoxes.append(x2)
      validBoxes.append(y2)
      val clsInd = labelMap(record(2)) + 1
      validClasses.append(clsInd.toFloat)
      val difficult = record(3).toFloat
      validDifficults.append(difficult)
    }

    // add last imagefeature
    val label = OpenImages.createRoiLabel(validClasses, validDifficults, validBoxes)
    array.append(ImageFeature(image, label, imagePath))

    ImageSet.array(array.toArray)
  }

  // label -> label index
  val labelMap = loadLableMap(devkitPath + "/class-descriptions-boxable.csv")

  // label index -> label name
  val labelName = loadLabelName(devkitPath + "/class-descriptions-boxable.csv")

}

object OpenImages {

  val logger = Logger.getLogger(getClass.getName)

  // label -> label index
  def loadLableMap(path: String): Map[String, Int] = {
    Source.fromFile(path).getLines().zipWithIndex.map(x => (x._1.split(",")(0), x._2)).toMap
  }

  // label index -> label name
  def loadLabelName(path: String): Map[Int, String] = {
    Source.fromFile(path).getLines().zipWithIndex.map(x => (x._2, x._1.split(",")(1))).toMap
  }

  def createRoiLabel(classesBuf: ArrayBuffer[Float],
                     difficultsBuf: ArrayBuffer[Float],
                     boxesBuf: ArrayBuffer[Float]): RoiLabel = {
    val clses = new Array[Float](classesBuf.length)
    classesBuf.copyToArray(clses, 0, classesBuf.length)
    val diffs = new Array[Float](difficultsBuf.length)
    difficultsBuf.copyToArray(diffs, 0, difficultsBuf.length)
    RoiLabel(Tensor(Storage(clses ++ diffs )).resize(2, classesBuf.length),
      Tensor(Storage(boxesBuf.toArray)).resize(boxesBuf.length / 4, 4))
  }
}
