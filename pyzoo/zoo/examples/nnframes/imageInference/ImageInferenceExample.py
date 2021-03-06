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

from bigdl.nn.layer import Model
from bigdl.util.common import *
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

from zoo.common.nncontext import *
from zoo.pipeline.nnframes.nn_classifier import *
from zoo.pipeline.nnframes.nn_image_reader import *
from zoo.feature.common import *
from zoo.feature.image.imagePreprocessing import *

def inference(image_path, model_path, sc):

    imageDF = NNImageReader.readImages(image_path, sc)
    getName = udf(lambda row: row[0], StringType())
    transformer = ChainedPreprocessing(
        [RowToImageFeature(), Resize(256, 256), CenterCrop(224, 224),
         ChannelNormalize(123.0, 117.0, 104.0), MatToTensor(), ImageFeatureToTensor()])

    model = Model.loadModel(model_path)
    classifier_model = NNClassifierModel(model, transformer)\
        .setFeaturesCol("image").setBatchSize(4)
    predictionDF = classifier_model.transform(imageDF).withColumn("name", getName(col("image")))
    return predictionDF


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Need parameters: <modelPath> <imagePath>")
        exit(-1)

    sc = get_nncontext()

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    predictionDF = inference(image_path, model_path, sc)
    predictionDF.select("name", "prediction").orderBy("name").show(20, False)
