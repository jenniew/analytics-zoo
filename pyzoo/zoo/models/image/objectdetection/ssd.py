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

import sys

from pyspark import RDD

from zoo.models.image.common.image_model import ImageModel
from bigdl.util.common import callBigDlFunc


if sys.version >= '3':
    long = int
    unicode = str


class SSD(ImageModel):
    """
    The base class for SSD models in Analytics Zoo.
    """


class SSDVGG(SSD):
    """
    SSD model based on VGG16.
    """
    def __init__(self, class_num, resolution=300,
                         dataset="pascal", sizes=None,
                         post_process_param=None, bigdl_type="float"):
        super(SSDVGG, self).__init__(None, bigdl_type, class_num, resolution, dataset,
                                     sizes, post_process_param)
