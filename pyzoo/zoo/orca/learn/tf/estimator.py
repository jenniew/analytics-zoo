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

from bigdl.optim.optimizer import MaxEpoch

from zoo.orca.learn.tf.utils import *
from zoo.tfpark import KerasModel
from zoo.tfpark import TFOptimizer, TFNet, ZooOptimizer
from zoo.tfpark.tf_optimizer import StatelessMetric
from zoo.tfpark.utils import evaluate_metrics
from zoo.util import nest


class Estimator(object):
    def fit(self, data, epochs, **kwargs):
        pass

    def predict(self, data, **kwargs):
        pass

    def evaluate(self, data, **kwargs):
        pass

    def load(self, path, version):
        """
        Load specified Orca checkpoint
        :param path: checkpoint path
        :param version: checkpoint version
        """
        self.load_checkpoint = True
        self.checkpoint_path = path
        self.checkpoint_version = version

    def load_latest_checkpoint(self, path):
        """
        Load latest Orca checkpoint under specified directory.
        :param path: checkpoint directory
        """
        ckpt_path, version = find_latest_checkpoint(path)
        if ckpt_path is None:
            raise Exception("Cannot find checkpoint")
        self.load(ckpt_path, version)

    def set_tensorboard(self, log_dir, app_name):
        """
        Set summary information during the training process for visualization purposes.
        Saved summary can be viewed via TensorBoard.
        In order to take effect, it needs to be called before fit.

        Training summary will be saved to 'log_dir/app_name/train'
        and validation summary (if any) will be saved to 'log_dir/app_name/validation'.

        # Arguments
        :param log_dir: The base directory path to store training and validation logs.
        :param app_name: The name of the application.
        """
        self.log_dir = log_dir
        self.app_name = app_name

    def get_train_summary(self, tag=None):
        """
        Get the scalar from model train summary
        Return list of summary data of [iteration_number, scalar_value, timestamp]
        # Arguments
        tag: The string variable represents the scalar wanted
        """
        if self.tf_optimizer:
            return self.tf_optimizer.estimator.get_train_summary(tag)

        return None

    def get_validation_summary(self, tag=None):
        """
        Get the scalar from model validation summary
        Return list of summary data of [iteration_number, scalar_value, timestamp]

        Note: The metric and tag may not be consistent
        Please look up following form to pass tag parameter
        Left side is your metric during compile
        Right side is the tag you should pass
        'Accuracy'                  |   'Top1Accuracy'
        'BinaryAccuracy'            |   'Top1Accuracy'
        'CategoricalAccuracy'       |   'Top1Accuracy'
        'SparseCategoricalAccuracy' |   'Top1Accuracy'
        'AUC'                       |   'AucScore'
        'HitRatio'                  |   'HitRate@k' (k is Top-k)
        'Loss'                      |   'Loss'
        'MAE'                       |   'MAE'
        'NDCG'                      |   'NDCG'
        'TFValidationMethod'        |   '${name + " " + valMethod.toString()}'
        'Top5Accuracy'              |   'Top5Accuracy'
        'TreeNNAccuracy'            |   'TreeNNAccuracy()'
        'MeanAveragePrecision'      |   'MAP@k' (k is Top-k) (BigDL)
        'MeanAveragePrecision'      |   'PascalMeanAveragePrecision' (Zoo)
        'StatelessMetric'           |   '${name}'
        # Arguments
        tag: The string variable represents the scalar wanted
        """
        if self.tf_optimizer:
            for val_method in self.tf_optimizer.tf_model.val_methods:
                if isinstance(val_method, StatelessMetric):
                    if tag == val_method.name:
                        return self.tf_optimizer.estimator.get_validation_summary(tag)
                else:
                    if tag == str(val_method.val_method):
                        return self.tf_optimizer.estimator.\
                            get_validation_summary("{} {}".format(val_method.name, tag))
                continue
        return None

    @staticmethod
    def from_graph(*, inputs, outputs=None,
                   labels=None, loss=None, optimizer=None,
                   clip_norm=None, clip_value=None,
                   metrics=None, updates=None,
                   sess=None, model_dir=None, backend="spark"):
        assert backend == "spark", "only spark backend is supported for now"
        return TFOptimizerWrapper(inputs=inputs,
                                  outputs=outputs,
                                  labels=labels,
                                  loss=loss,
                                  optimizer=optimizer,
                                  clip_norm=clip_norm,
                                  clip_value=clip_value,
                                  metrics=metrics, updates=updates,
                                  sess=sess,
                                  model_dir=model_dir
                                  )

    @staticmethod
    def from_keras(keras_model, metrics=None, model_dir=None, backend="spark"):
        assert backend == "spark", "only spark backend is supported for now"
        return TFKerasWrapper(keras_model, metrics, model_dir)


class TFOptimizerWrapper(Estimator):
    def __init__(self, *, inputs, outputs, labels, loss,
                 optimizer, clip_norm, clip_value,
                 metrics,
                 updates, sess,
                 model_dir
                 ):
        self.inputs = inputs
        self.outputs = outputs
        self.labels = labels
        self.loss = loss
        if optimizer is not None:
            assert isinstance(optimizer, tf.train.Optimizer), \
                "optimizer is of type {}, ".format(type(optimizer)) + \
                "it should be an instance of tf.train.Optimizer"
            self.optimizer = ZooOptimizer(optimizer)
            if clip_norm or clip_value:
                gvs = self.optimizer.compute_gradients(self.loss)
                if clip_norm:
                    gvs = [(tf.clip_by_norm(g_v[0], clip_norm), g_v[1]) for g_v in gvs]
                if clip_value:
                    if isinstance(clip_value, tuple):
                        assert len(clip_value) == 2 and clip_value[0] < clip_value[1], \
                            "clip value should be (clip_min, clip_max)"
                        gvs = [(tf.clip_by_value(g_v[0], clip_value[0], clip_value[1]), g_v[1])
                               for g_v in gvs]
                    if isinstance(clip_value, (int, float)):
                        assert clip_value > 0, "clip value should be larger than 0"
                        gvs = [(tf.clip_by_value(g_v[0], -clip_value, clip_value), g_v[1])
                               for g_v in gvs]
                    else:
                        raise Exception("clip_value should be a tuple or one number")
                self.train_op = self.optimizer.apply_gradients(gvs)
            else:
                self.train_op = self.optimizer.minimize(self.loss)
        else:
            self.optimizer = None
            self.train_op = None
        self.metrics = metrics
        self.updates = updates
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        self.model_dir = model_dir
        self.load_checkpoint = False
        self.tf_optimizer = None
        self.log_dir = None
        self.app_name = None

    def fit(self, data,
            epochs=1,
            batch_size=32,
            feature_cols=None,
            labels_cols=None,
            validation_data=None,
            hard_code_batch_size=False,
            session_config=None,
            feed_dict=None,
            checkpoint_trigger=None
            ):

        assert self.labels is not None, \
            "labels is None; it should not be None in training"
        assert self.loss is not None, \
            "loss is None; it should not be None in training"
        assert self.optimizer is not None, \
            "optimizer is None; it should not be None in training"

        if isinstance(data, DataFrame):
            assert feature_cols is not None, \
                "feature columns is None; it should not be None in training"
            assert labels_cols is not None, \
                "label columns is None; it should not be None in training"

        dataset = to_dataset(data, batch_size=batch_size, batch_per_thread=-1,
                             validation_data=validation_data,
                             feature_cols=feature_cols, labels_cols=labels_cols,
                             hard_code_batch_size=hard_code_batch_size,
                             sequential_order=False, shuffle=True
                             )

        if feed_dict is not None:
            tensor_with_value = {key: (value, value) for key, value in feed_dict.items()}
        else:
            tensor_with_value = None

        self.tf_optimizer = TFOptimizer.from_train_op(
            train_op=self.train_op,
            loss=self.loss,
            inputs=self.inputs,
            labels=self.labels,
            dataset=dataset,
            metrics=self.metrics,
            updates=self.updates, sess=self.sess,
            tensor_with_value=tensor_with_value,
            session_config=session_config,
            model_dir=self.model_dir)

        if self.load_checkpoint:
            self.tf_optimizer.load_checkpoint(self.checkpoint_path, self.checkpoint_version)

        if self.log_dir and self.app_name:
            self.tf_optimizer.estimator.set_tensorboad(self.log_dir, self.app_name)

        self.tf_optimizer.optimize(end_trigger=MaxEpoch(epochs),
                                   checkpoint_trigger=checkpoint_trigger)
        return self

    def predict(self, data, batch_size=4,
                feature_cols=None,
                hard_code_batch_size=False
                ):

        assert self.outputs is not None, \
            "output is None, it should not be None in prediction"
        if isinstance(data, DataFrame):
            assert feature_cols is not None, \
                "feature columns is None; it should not be None in prediction"

        dataset = to_dataset(data, batch_size=-1, batch_per_thread=batch_size,
                             validation_data=None,
                             feature_cols=feature_cols, labels_cols=None,
                             hard_code_batch_size=hard_code_batch_size,
                             sequential_order=True,
                             shuffle=False
                             )

        flat_inputs = nest.flatten(self.inputs)
        flat_outputs = nest.flatten(self.outputs)
        tfnet = TFNet.from_session(sess=self.sess, inputs=flat_inputs, outputs=flat_outputs)
        predicted_rdd = tfnet.predict(dataset)
        if isinstance(data, DataFrame):
            return convert_predict_to_dataframe(data, predicted_rdd)
        elif isinstance(data, SparkXShards):
            return convert_predict_to_xshard(data, predicted_rdd)
        else:
            return predicted_rdd

    def evaluate(self, data, batch_size=32,
                 feature_cols=None,
                 labels_cols=None,
                 hard_code_batch_size=False
                 ):

        assert self.metrics is not None, \
            "metrics is None, it should not be None in evaluate"

        if isinstance(data, DataFrame):
            assert feature_cols is not None, \
                "feature columns is None; it should not be None in evaluation"
            assert labels_cols is not None, \
                "label columns is None; it should not be None in evaluation"

        dataset = to_dataset(data, batch_size=-1, batch_per_thread=batch_size,
                             validation_data=None,
                             feature_cols=feature_cols, labels_cols=labels_cols,
                             hard_code_batch_size=hard_code_batch_size,
                             sequential_order=True,
                             shuffle=False
                             )

        flat_inputs = nest.flatten(self.inputs)
        flat_labels = nest.flatten(self.labels)

        return evaluate_metrics(flat_inputs + flat_labels,
                                sess=self.sess,
                                dataset=dataset, metrics=self.metrics)


class TFKerasWrapper(Estimator):
    def __init__(self, keras_model, metrics, model_dir):
        self.model = KerasModel(keras_model, model_dir)
        self.load_checkpoint = False
        self.metrics = metrics
        self.tf_optimizer = None
        self.log_dir = None
        self.app_name = None

    def fit(self, data,
            epochs=1,
            batch_size=32,
            feature_cols=None,
            labels_cols=None,
            validation_data=None,
            hard_code_batch_size=False,
            session_config=None,
            checkpoint_trigger=None
            ):

        if isinstance(data, DataFrame):
            assert feature_cols is not None, \
                "feature columns is None; it should not be None in training"
            assert labels_cols is not None, \
                "label columns is None; it should not be None in training"

        dataset = to_dataset(data, batch_size=batch_size, batch_per_thread=-1,
                             validation_data=validation_data,
                             feature_cols=feature_cols, labels_cols=labels_cols,
                             hard_code_batch_size=hard_code_batch_size,
                             sequential_order=False, shuffle=True
                             )

        self.tf_optimizer = TFOptimizer.from_keras(self.model.model, dataset,
                                                   model_dir=self.model.model_dir,
                                                   session_config=session_config,
                                                   metrics=self.metrics)

        if self.load_checkpoint:
            self.tf_optimizer.load_checkpoint(self.checkpoint_path, self.checkpoint_version)

        if self.log_dir and self.app_name:
            self.tf_optimizer.estimator.set_tensorboad(self.log_dir, self.app_name)

        self.tf_optimizer.optimize(MaxEpoch(epochs), checkpoint_trigger=checkpoint_trigger)

        return self

    def predict(self, data, batch_size=4,
                feature_cols=None,
                hard_code_batch_size=False
                ):

        if isinstance(data, DataFrame):
            assert feature_cols is not None, \
                "feature columns is None; it should not be None in prediction"

        dataset = to_dataset(data, batch_size=-1, batch_per_thread=batch_size,
                             validation_data=None,
                             feature_cols=feature_cols, labels_cols=None,
                             hard_code_batch_size=hard_code_batch_size,
                             sequential_order=True, shuffle=False
                             )

        predicted_rdd = self.model.predict(dataset, batch_size)
        if isinstance(data, DataFrame):
            return convert_predict_to_dataframe(data, predicted_rdd)
        elif isinstance(data, SparkXShards):
            return convert_predict_to_xshard(data, predicted_rdd)
        else:
            return predicted_rdd

    def evaluate(self, data, batch_size=4,
                 feature_cols=None,
                 labels_cols=None,
                 hard_code_batch_size=False
                 ):

        if isinstance(data, DataFrame):
            assert feature_cols is not None, \
                "feature columns is None; it should not be None in evaluation"
            assert labels_cols is not None, \
                "label columns is None; it should not be None in evaluation"

        dataset = to_dataset(data, batch_size=-1, batch_per_thread=batch_size,
                             validation_data=None,
                             feature_cols=feature_cols, labels_cols=labels_cols,
                             hard_code_batch_size=hard_code_batch_size,
                             sequential_order=True, shuffle=False
                             )

        return self.model.evaluate(dataset, batch_per_thread=batch_size)
