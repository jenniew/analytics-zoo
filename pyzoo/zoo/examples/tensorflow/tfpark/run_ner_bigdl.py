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

import os
import sys
import time
import pickle
import numpy as np
import tensorflow as tf
from optparse import OptionParser

# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Bidirectional, LSTM, Dense
from bert_ner_utils import *

from bigdl.nn.criterion import CrossEntropyCriterion
from bigdl.optim.optimizer import Loss
from zoo.pipeline.api.keras.objectives import TimeDistributedCriterion
from zoo.common import Sample
from zoo.common.nncontext import init_nncontext
from zoo.pipeline.api.keras.models import Model
from zoo.pipeline.api.keras.layers import Bidirectional, LSTM, Dense, InputLayer, Input
from zoo.tfpark.text.estimator import BERTFeatureExtractor, bert_input_fn
from zoo.pipeline.api.keras.optimizers import AdamWeightDecay, Adam
from zoo.feature.common import *


def feature_to_input(feature):
    res = dict()
    res["input_ids"] = np.array(feature.input_ids)
    res["input_mask"] = np.array(feature.mask)
    res["token_type_ids"] = np.array(feature.segment_ids)
    return res, np.array(feature.label_ids) + 1


def generate_input_rdd(examples, label_list, max_seq_length, tokenizer, type="train"):
    features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer)
    features = [feature_to_input(feature) for feature in features]
    if type == "test":
        return sc.parallelize(features).map(lambda x: x[0])
    else:
        return sc.parallelize(features)


def build_model():
    input1 = Input(shape=(options.max_seq_length, 768))
    lstm1 = Bidirectional(LSTM(768, return_sequences=True))(input1)
    lstm2 = Bidirectional(LSTM(768, return_sequences=True))(lstm1)
    fc = Dense(len(label_list), activation="softmax")(lstm2)
    model = Model(input1, fc)
    model.summary()
    return model


if __name__ == '__main__':
    start_time = time.time()
    parser = OptionParser()
    parser.add_option("--bert_base_dir", dest="bert_base_dir")
    parser.add_option("--data_dir", dest="data_dir")
    parser.add_option("--output_dir", dest="output_dir")
    parser.add_option("--batch_size", dest="batch_size", type=int, default=32)
    parser.add_option("--max_seq_length", dest="max_seq_length", type=int, default=128)
    parser.add_option("-e", "--nb_epoch", dest="nb_epoch", type=int, default=3)
    parser.add_option("-l", "--learning_rate", dest="learning_rate", type=float, default=2e-5)
    parser.add_option("--do_train", dest="do_train", type=int, default=1)
    parser.add_option("--do_eval", dest="do_eval", type=int, default=1)
    parser.add_option("--do_predict", dest="do_predict", type=int, default=1)
    parser.add_option("--mem_type", dest="mem_type", type=str, default="DRAM")

    (options, args) = parser.parse_args(sys.argv)
    sc = init_nncontext("BERT NER Example")

    processor = NerProcessor()
    label_list = processor.get_labels()
    # Recommended to use cased model for NER
    tokenizer = tokenization.FullTokenizer(os.path.join(options.bert_base_dir, "vocab.txt"), do_lower_case=False)
    estimator = BERTFeatureExtractor(
        bert_config_file=os.path.join(options.bert_base_dir, "bert_config.json"),
        init_checkpoint=os.path.join(options.bert_base_dir, "bert_model.ckpt"))
    model = build_model()

    # Training
    if options.do_train:
        # prepare train data
        train_examples = processor.get_train_examples(options.data_dir)
        train_rdd = generate_input_rdd(train_examples, label_list, options.max_seq_length, tokenizer, "train")
        train_input_fn = bert_input_fn(train_rdd, options.max_seq_length, options.batch_size)
        train_rdd_bert = estimator.predict(train_input_fn).zip(train_rdd.map(lambda x: x[1]))
        train_rdd_bert = train_rdd_bert.map(lambda x: Sample.from_ndarray(x[0], x[1]))
        train_dataset = FeatureSet.rdd(train_rdd_bert, memory_type=options.mem_type)
        # train_feature = sc.parallelize(np.random.random((20, 128, 768)))
        # train_lable = sc.parallelize(np.random.randint(1, len(label_list), (20, 128, 1)))
        # train_rdd = train_feature.zip(train_lable).map(lambda x: Sample.from_ndarray(x[0], x[1]))
        # train_dataset = FeatureSet.rdd(train_rdd, memory_type=options.mem_type)

        # prepare validation data
        val_examples = processor.get_dev_examples(options.data_dir)
        val_rdd = generate_input_rdd(val_examples, label_list, options.max_seq_length, tokenizer, "eval")
        val_input_fn = bert_input_fn(val_rdd, options.max_seq_length, options.batch_size)
        val_rdd_bert = estimator.predict(val_input_fn).zip(val_rdd.map(lambda x: x[1]))
        val_rdd_bert = val_rdd_bert.map(lambda x: Sample.from_ndarray(x[0], x[1]))
        val_dataset = FeatureSet.rdd(val_rdd_bert, memory_type=options.mem_type)

        # eval_feature = sc.parallelize(np.random.random((20, 128, 768)))
        # eval_lable = sc.parallelize(np.random.randint(1, len(label_list), (20, 128, 1)).astype("float32"))
        # eval_rdd = eval_feature.zip(eval_lable).map(lambda x: Sample.from_ndarray(x[0], x[1]))
        # eval_dataset = FeatureSet.rdd(eval_rdd, memory_type=options.mem_type)

        steps = len(train_examples) * options.nb_epoch // options.batch_size
        # optimizer = AdamWeightDecay(lr=options.learning_rate, warmup_portion=0.1, total=steps)
        optimizer = Adam(lr=options.learning_rate)
        model.compile(optimizer=optimizer,
                      loss=TimeDistributedCriterion(CrossEntropyCriterion(), size_average=False, dimension=1),
                      metrics=[Loss(TimeDistributedCriterion(CrossEntropyCriterion(), size_average=False, dimension=1))])
        # model.compile(optimizer=optimizer, loss=ClassNLLCriterion())
        model.set_gradient_clipping_by_l2_norm(1.0)
        # train_rdd_bert = train_rdd.map(lambda x: Sample.from_ndarray(x[0], x[1]))

        train_start_time = time.time()
        model.fit(train_dataset, nb_epoch=options.nb_epoch, batch_size=options.batch_size, validation_data=val_dataset)
        train_end_time = time.time()
        print("Train time: %s minutes" % ((train_end_time - train_start_time) / 60))

    # Evaluation
    # Confusion matrix is not supported and thus use sklearn classification_report for evaluation
    if options.do_eval:
        eval_examples = processor.get_dev_examples(options.data_dir)
        eval_rdd = generate_input_rdd(eval_examples, label_list, options.max_seq_length, tokenizer, "eval")
        eval_input_fn = bert_input_fn(eval_rdd, options.max_seq_length, options.batch_size)
        eval_rdd_bert = estimator.predict(eval_input_fn).zip(eval_rdd.map(lambda x: x[1]))
        # eval_rdd_bert = sc.parallelize(eval_rdd_bert.collect())
        # eval_rdd_bert = sc.parallelize(np.random.uniform(low=-50, high=13.3, size=(20, 128, 768)))
        eval_dataset = eval_rdd_bert.map(lambda x: Sample.from_ndarray(x[0], x[1]))
        # eval_dataset = FeatureSet.rdd(eval_rdd, memory_type=options.mem_type)
        # model.evaluate(eval_dataset, batch_size=8)
        result = model.predict(eval_dataset)
        print(result.take(5))
        # pred = result.collect()
        predictions = np.concatenate([np.argmax(r, axis=-1) for r in result.collect()])
        truths = np.concatenate([r[1] for r in eval_rdd.collect()])
        mask = np.concatenate([r[0]["input_mask"] for r in eval_rdd.collect()])
        from sklearn.metrics import classification_report
        print(classification_report(truths, predictions, sample_weight=mask,
                                    labels=range(len(label_list)), target_names=label_list))

    # Inference
    if options.do_predict:
        test_examples = processor.get_test_examples(options.data_dir)
        test_rdd = generate_input_rdd(test_examples, label_list, options.max_seq_length, tokenizer, "test")
        # print(test_rdd.take(1)[0].shape)
        test_input_fn = bert_input_fn(test_rdd, options.max_seq_length, options.batch_size)
        test_rdd_bert = estimator.predict(test_input_fn).zip(test_rdd.map(lambda x: x[1]))
        test_dataset = test_rdd_bert.map(lambda x: Sample.from_ndarray(x[0], x[1]))
        a = test_dataset.take(2)

        pred_start_time = time.time()
        predictions = model.predict(test_dataset)
        predictions.collect()
        pred_end_time = time.time()
        print("Inference time: %s minutes" % ((pred_end_time - pred_start_time) / 60))
        print("Inference throughput: %s records/s" % (len(test_examples) / (pred_end_time - pred_start_time)))
        for prediction in predictions.take(5):
            print(prediction)

    end_time = time.time()
    print("Time elapsed: %s minutes" % ((end_time - start_time) / 60))
    print("Finished")
