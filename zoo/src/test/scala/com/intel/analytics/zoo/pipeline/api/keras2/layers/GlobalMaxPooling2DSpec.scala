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

package com.intel.analytics.zoo.pipeline.api.keras2.layers

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Keras2Test, KerasBaseSpec}
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest

import scala.util.Random

class GlobalMaxPooling2DSpec extends KerasBaseSpec {

  "GlobalMaxPooling2D NCHW" should "be the same as Keras" taggedAs(Keras2Test) in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[4, 24, 32])
        |input = np.random.random([2, 4, 24, 32])
        |output_tensor = GlobalMaxPooling2D(data_format="channels_first")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = GlobalMaxPooling2D[Float](inputShape = Shape(4, 24, 32))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 4))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

  "GlobalMaxPooling2D NHWC" should "be the same as Keras" taggedAs(Keras2Test) in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[16, 16, 2])
        |input = np.random.random([3, 16, 16, 2])
        |output_tensor = GlobalMaxPooling2D(data_format="channels_last")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = GlobalMaxPooling2D[Float](dataFormat = "channels_last",
      inputShape = Shape(16, 16, 2))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 2))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

}

class GlobalMaxPooling2DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = GlobalMaxPooling2D[Float](inputShape = Shape(4, 24, 32))
    layer.build(Shape(2, 4, 24, 32))
    val input = Tensor[Float](2, 4, 24, 32).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}

