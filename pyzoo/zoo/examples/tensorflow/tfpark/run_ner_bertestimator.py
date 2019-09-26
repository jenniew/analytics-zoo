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

import time
import pickle
from optparse import OptionParser

from zoo.common.nncontext import *
from zoo.tfpark.text.estimator import BERTNER, bert_input_fn
from zoo.pipeline.api.keras.optimizers import AdamWeightDecay
from bert_ner_utils import *


def feature_to_input(feature):
    res = dict()
    res["input_ids"] = np.array(feature.input_ids)
    res["input_mask"] = np.array(feature.mask)
    res["token_type_ids"] = np.array(feature.segment_ids)
    return res, np.array(feature.label_ids)


def generate_input_rdd(examples, label_list, max_seq_length, tokenizer, type="train"):
    features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer)
    features = [feature_to_input(feature) for feature in features]
    if type == "test":
        return sc.parallelize(features).map(lambda x: x[0])
    else:
        return sc.parallelize(features)


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

    (options, args) = parser.parse_args(sys.argv)
    sc = init_nncontext("BERT NER Example")

    processor = NerProcessor()
    label_list = processor.get_labels()
    # Recommended to use cased model for NER
    tokenizer = tokenization.FullTokenizer(os.path.join(options.bert_base_dir, "vocab.txt"), do_lower_case=False)
    estimator = BERTNER(len(label_list),
                        bert_config_file=os.path.join(options.bert_base_dir, "bert_config.json"),
                        init_checkpoint=os.path.join(options.bert_base_dir, "bert_model.ckpt"),
                        model_dir=options.output_dir)

    # Training
    if options.do_train:
        train_examples = processor.get_train_examples(options.data_dir)
        steps = len(train_examples) * options.nb_epoch // options.batch_size
        optimizer = AdamWeightDecay(lr=options.learning_rate, warmup_portion=0.1, total=steps)
        estimator.set_optimizer(optimizer)
        estimator.set_gradient_clipping_by_l2_norm(1.0)
        train_rdd = generate_input_rdd(train_examples, label_list, options.max_seq_length, tokenizer, "train")
        train_input_fn = bert_input_fn(train_rdd, options.max_seq_length, options.batch_size,
                                       label_size=options.max_seq_length)
        train_start_time = time.time()
        estimator.train(train_input_fn, steps=steps)
        train_end_time = time.time()
        print("Train time: %s minutes" % ((train_end_time - train_start_time) / 60))

    # Evaluation
    # Confusion matrix is not supported and thus use sklearn classification_report for evaluation
    if options.do_eval:
        eval_examples = processor.get_dev_examples(options.data_dir)
        eval_rdd = generate_input_rdd(eval_examples, label_list, options.max_seq_length, tokenizer, "eval")
        eval_input_fn = bert_input_fn(eval_rdd, options.max_seq_length, options.batch_size)
        result = estimator.predict(eval_input_fn).zip(eval_rdd).collect()
        predictions = np.concatenate([r[0] for r in result])
        truths = np.concatenate([r[1][1] for r in result])
        mask = np.concatenate([r[1][0]["input_mask"] for r in result])
        # label_map = {}
        # for (i, label) in enumerate(label_list):
        #     label_map[label] = i
        # with open(os.path.join(options.output_dir, "label2id.pkl"), 'rb') as rf:
        #     label2id = pickle.load(rf)
        #     id2label = {value: key for key, value in label2id.items()}
        # sorted_ids = sorted(id2label.keys())
        # labels = [id2label[id] for id in sorted_ids]
        # sorted_ids = range(len(label_list))
        # labels = [id2label[id] for id in sorted_ids]
        from sklearn.metrics import classification_report
        print(classification_report(truths, predictions, sample_weight=mask,
                                    labels=range(len(label_list)), target_names=label_list))

    # Inference
    if options.do_predict:
        test_examples = processor.get_test_examples(options.data_dir)
        test_rdd = generate_input_rdd(test_examples, label_list, options.max_seq_length, tokenizer, "test")
        test_input_fn = bert_input_fn(test_rdd, options.max_seq_length, options.batch_size)
        predictions = estimator.predict(test_input_fn)
        pred_start_time = time.time()
        predictions.collect()
        pred_end_time = time.time()
        print("Inference time: %s minutes" % ((pred_end_time - pred_start_time) / 60))
        print("Inference throughput: %s records/s" % (len(test_examples) / (pred_end_time - pred_start_time)))
        for prediction in predictions.take(5):
            print(prediction)

    end_time = time.time()
    print("Time elapsed: %s minutes" % ((end_time - start_time) / 60))
    print("Finished")
