import frame_level_models
import losses
import tensorflow as tf
import video_level_models
import train as ytrain
import utils
import readers

from tensorflow import flags
from tensorflow import gfile
from tensorflow import app


from zoo.common.nncontext import init_nncontext
from zoo.pipeline.api.net import TFOptimizer
from zoo.pipeline.api.net import TFDataset
from bigdl.optim.optimizer import MaxEpoch, TrainSummary, Adam

slim = tf.contrib.slim
FLAGS = flags.FLAGS


def find_class_by_name(name, modules):
    """Searches the provided modules for the named class and returns it."""
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)


def mapper(filename):
    return (example for example in tf.python_io.tf_record_iterator(filename))


def read2(example, is_frame_level, reader):
    num_features = len(reader.feature_names)
    max_quantized_value = 2
    min_quantized_value = -2
    assert num_features > 0, "No feature selected: feature_names is empty!"
    if is_frame_level:
        sess = tf.InteractiveSession()
        contexts, features = tf.parse_single_sequence_example(
            example,
            context_features={"id": tf.FixedLenFeature(
                [], tf.string),
                "labels": tf.VarLenFeature(tf.int64)},
            sequence_features={
                feature_name: tf.FixedLenSequenceFeature([], dtype=tf.string)
                for feature_name in reader.feature_names
            })

        # read ground truth labels
        labels = (
            tf.sparse_to_dense(contexts["labels"].values, (reader.num_classes,), 1,
                               validate_indices=False)).eval()

        # loads (potentially) different types of features and concatenates them
        num_features = len(reader.feature_names)
        assert num_features > 0, "No feature selected: feature_names is empty!"

        assert len(reader.feature_names) == len(reader.feature_sizes), \
            "length of feature_names (={}) != length of feature_sizes (={})".format( \
                len(reader.feature_names), len(reader.feature_sizes))

        num_frames = -1  # the number of frames in the video
        feature_matrices = [None] * num_features  # an array of different features
        for feature_index in range(num_features):
            feature_matrix, num_frames_in_this_feature = reader.get_video_matrix(
                features[reader.feature_names[feature_index]],
                reader.feature_sizes[feature_index],
                reader.max_frames,
                max_quantized_value,
                min_quantized_value)
            if num_frames == -1:
                num_frames = num_frames_in_this_feature
            else:
                tf.assert_equal(num_frames, num_frames_in_this_feature)

            feature_matrices[feature_index] = feature_matrix

        # cap the number of frames at self.max_frames
        num_frames = tf.minimum(num_frames, reader.max_frames).eval()

        # concatenate different features
        video_matrix = tf.concat(feature_matrices, 1).eval()

        # batch_video_ids = tf.expand_dims(contexts["id"], 0).eval()
        # batch_video_matrix = tf.expand_dims(video_matrix, 0)
        # batch_labels = tf.expand_dims(labels, 0).eval()
        # batch_frames = tf.expand_dims(num_frames, 0).eval()

        sess.close()

        # print [model_input, num_frames], labels
        return ([video_matrix, num_frames], labels)


def build_graph(model, model_input_raw, vocab_size, num_frames, labels, iterations,
                label_loss_fn=losses.CrossEntropyLoss(),
                base_learning_rate=0.01,
                learning_rate_decay_examples=1000000,
                learning_rate_decay=0.95,
                regularization_penalty=1):
    learning_rate = tf.train.exponential_decay(
        base_learning_rate,
        iterations,
        learning_rate_decay_examples,
        learning_rate_decay,
        staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

    tf.summary.histogram("model/input_raw", model_input_raw)

    feature_dim = len(model_input_raw.get_shape()) - 1

    model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)

    with tf.name_scope("model"):
        result = model.create_model(
            model_input,
            num_frames=num_frames,
            vocab_size=vocab_size,
            labels=labels)

        for variable in slim.get_model_variables():
            tf.summary.histogram(variable.op.name, variable)

        predictions = result["predictions"]
        if "loss" in result.keys():
            label_loss = result["loss"]
        else:
            label_loss = label_loss_fn.calculate_loss(predictions, labels)
        tf.summary.scalar("label_loss", label_loss)

        if "regularization_loss" in result.keys():
            reg_loss = result["regularization_loss"]
        else:
            reg_loss = tf.constant(0.0)

        reg_losses = tf.losses.get_regularization_losses()
        if reg_losses:
            reg_loss += tf.add_n(reg_losses)

        if regularization_penalty != 0:
            tf.summary.scalar("reg_loss", reg_loss)

        # Adds update_ops (e.g., moving average updates in batch normalization) as
        # a dependency to the train_op.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if "update_ops" in result.keys():
            update_ops += result["update_ops"]
        if update_ops:
            with tf.control_dependencies(update_ops):
                barrier = tf.no_op(name="gradient_barrier")
                with tf.control_dependencies([barrier]):
                    label_loss = tf.identity(label_loss)

        # Incorporate the L2 weight penalties etc.
        final_loss = regularization_penalty * reg_loss + label_loss
        return final_loss

        # model = find_class_by_name(FLAGS.model, [frame_level_models, video_level_models])()
        # if isinstance(model, frame_level_models.NetVLADModelLF):
        #     graph = model.create_model(model_input, vocab_size, num_frames, FLAGS.iterations)


def main(unused_argv):
    reader = ytrain.get_reader()

    # model_exporter = export_model.ModelExporter(
    #     frame_features=FLAGS.frame_features, model=model, reader=reader)

    # files = gfile.Glob(FLAGS.train_data_pattern)
    # if not files:
    #     raise IOError("Unable to find training files. data_pattern='" +
    #                   FLAGS.train_data_pattern + "'.")
    # print("Number of training files: %s.", str(len(files)))

    sc = init_nncontext("Video Classification Example")
    # record_rdd = sc.parallelize(files).flatMap(
    #     lambda filename: (example for example in tf.python_io.tf_record_iterator(filename)))
    dataRDD = sc.newAPIHadoopFile(FLAGS.train_data_pattern, "org.tensorflow.hadoop.io.TFRecordFileInputFormat",
                                  keyClass="org.apache.hadoop.io.BytesWritable",
                                  valueClass="org.apache.hadoop.io.NullWritable")
    train_data = dataRDD.map(lambda record: read2(bytes(record[0]), True, reader))

    train_dataset = TFDataset.from_rdd(train_data,
                                       features=[(tf.float32, [300, 1152]), (tf.int32, [])],
                                       labels=(tf.int32, [4716]),
                                       batch_size=FLAGS.batch_size)

    [model_inputs, num_frames], labels = train_dataset.tensors

    model = find_class_by_name(FLAGS.model, [frame_level_models, video_level_models])()
    label_loss_fn = find_class_by_name(FLAGS.label_loss, [losses])()
    loss = build_graph(model, model_inputs, vocab_size=reader.num_classes, num_frames=num_frames,
                       labels=labels, iterations=FLAGS.iterations,
                       label_loss_fn=label_loss_fn,
                       base_learning_rate=FLAGS.base_learning_rate,
                       learning_rate_decay=FLAGS.learning_rate_decay,
                       learning_rate_decay_examples=FLAGS.learning_rate_decay_examples,
                       regularization_penalty=FLAGS.regularization_penalty
                       )

    if "AdamOptimizer" == FLAGS.optimizer:
        optim_method = Adam()
    optimizer = TFOptimizer.from_loss(loss, optim_method, clip_norm=FLAGS.clip_gradient_norm)
    optimizer.set_train_summary(TrainSummary("/model/youtube/frame/summary", "youtube"))
    optimizer.optimize(end_trigger=MaxEpoch(FLAGS.num_epochs))
    saver = tf.train.Saver()
    saver.save(optimizer.sess, FLAGS.train_dir)
    sc.stop()


if __name__ == "__main__":

    # Dataset flags.
    flags.DEFINE_string("train_dir", "/tmp/yt8m_model/",
                        "The directory to save the model files in.")
    flags.DEFINE_string(
        "train_data_pattern", "",
        "File glob for the training dataset. If the files refer to Frame Level "
        "features (i.e. tensorflow.SequenceExample), then set --reader_type "
        "format. The (Sequence)Examples are expected to have 'rgb' byte array "
        "sequence feature as well as a 'labels' int64 context feature.")
    flags.DEFINE_string("feature_names", "mean_rgb", "Name of the feature "
                                                     "to use for training.")
    flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")

    # Model flags.
    flags.DEFINE_bool(
        "frame_features", False,
        "If set, then --train_data_pattern must be frame-level features. "
        "Otherwise, --train_data_pattern must be aggregated video-level "
        "features. The model must also be set appropriately (i.e. to read 3D "
        "batches VS 4D batches.")
    flags.DEFINE_bool(
        "segment_labels", False,
        "If set, then --train_data_pattern must be frame-level features (but with"
        " segment_labels). Otherwise, --train_data_pattern must be aggregated "
        "video-level features. The model must also be set appropriately (i.e. to "
        "read 3D batches VS 4D batches.")
    flags.DEFINE_string(
        "model", "LogisticModel",
        "Which architecture to use for the model. Models are defined "
        "in models.py.")
    flags.DEFINE_bool(
        "start_new_model", False,
        "If set, this will not resume from a checkpoint and will instead create a"
        " new model instance.")
    flags.DEFINE_float("clip_gradient_norm", 1.0, "Norm to clip gradients to.")

    # Training flags.
    # flags.DEFINE_integer(
    #     "num_gpu", 1, "The maximum number of GPU devices to use for training. "
    #                   "Flag only applies if GPUs are installed")
    flags.DEFINE_integer("batch_size", 1024,
                         "How many examples to process per batch for training.")
    flags.DEFINE_string("label_loss", "CrossEntropyLoss",
                        "Which loss function to use for training the model.")
    flags.DEFINE_float(
        "regularization_penalty", 1.0,
        "How much weight to give to the regularization loss (the label loss has "
        "a weight of 1).")
    flags.DEFINE_float("base_learning_rate", 0.01,
                       "Which learning rate to start with.")
    flags.DEFINE_float(
        "learning_rate_decay", 0.95,
        "Learning rate decay factor to be applied every "
        "learning_rate_decay_examples.")
    flags.DEFINE_float(
        "learning_rate_decay_examples", 4000000,
        "Multiply current learning rate by learning_rate_decay "
        "every learning_rate_decay_examples.")
    flags.DEFINE_integer(
        "num_epochs", 5, "How many passes to make over the dataset before "
                         "halting training.")
    flags.DEFINE_integer(
        "max_steps", None,
        "The maximum number of iterations of the training loop.")
    flags.DEFINE_integer(
        "export_model_steps", 1000,
        "The period, in number of steps, with which the model "
        "is exported for batch prediction.")

    # Other flags.
    flags.DEFINE_integer("num_readers", 8,
                         "How many threads to use for reading input files.")
    flags.DEFINE_string("optimizer", "AdamOptimizer",
                        "What optimizer class to use.")

    # (options, args) = parser.parse_args(sys.argv)
    app.run()
