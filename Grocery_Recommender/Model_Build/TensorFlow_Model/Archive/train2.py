import os
import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes

SIGNATURE_NAME = 'predictions'

def model_fn(features, labels, mode, params):
   # Set the batch norm params
    batch_norm_momentum = 0.99

    # Create neural network input layer using our feature columns defined above
    net = tf.feature_column.input_layer(features=features,
                                        feature_columns=params['feature_columns'])

    he_init = tf.variance_scaling_initializer()

    if mode == tf.estimator.ModeKeys.TRAIN:
        batch_norm_layer = partial(
            tf.layers.batch_normalization,
            training=True,
            momentum=batch_norm_momentum)

    else:
        batch_norm_layer = partial(
            tf.layers.batch_normalization,
            training=False,
            momentum=batch_norm_momentum)

    dense_layer = partial(
        tf.layers.dense,
        kernel_initializer=he_init)

    # Create hidden and batch norm layers
    hidden1 = dense_layer(inputs=net, units=params['hidden_units'][0])
    bn1 = tf.nn.relu(batch_norm_layer(hidden1))
    hidden2 = dense_layer(inputs=bn1, units=params['hidden_units'][1])
    bn2 = tf.nn.relu(batch_norm_layer(hidden2))
    hidden3=dense_layer(inputs=bn2, units=params['hidden_units'][2])
    bn3 = tf.nn.relu(batch_norm_layer(hidden3))

    # Compute logits using the output of our last hidden layer
    logits = tf.layers.dense(inputs=bn3, units=1, activation=None)

    # Define operations
    if mode in (Modes.PREDICT, Modes.EVAL):
        
        logits = logits
        probabilities = tf.nn.sigmoid(logits)

    if mode in (Modes.TRAIN, Modes.EVAL):
        
        # Compute loss using sigmoid cross entropy since this is classification and our labels
        # and probabilities are mutually exclusive
        labels = tf.cast(logits, tf.float32)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                       logits=tf.squeeze(logits))

        predicted_class = tf.cast(tf.round(tf.nn.sigmoid(logits)), tf.float32)


        # Compute evaluation metrics of total accuracy and auc
        accuracy = tf.metrics.accuracy(labels=labels,
                                       predictions=predicted_class,
                                       name='acc_op')

        auc = tf.metrics.auc(labels=labels,
                             predictions=predicted_class,
                             name='auc')

        # Create scalar summaries to see in TensorBoard
        tf.summary.scalar(name='accuracy', tensor=accuracy[1])
        tf.summary.scalar(name='auc', tensor=auc[1])


    if mode == Modes.PREDICT:
        predictions = {
            'logits': logits,
            'probabilities': probabilities
        }
        export_outputs = {
            SIGNATURE_NAME: tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs)

    if mode == Modes.TRAIN:
        
        # Create a custom optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    
        # Create train op
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == Modes.EVAL:
        
        # Put eval metrics into a dictionary
        eval_metrics_ops = {'accuracy': accuracy,
                            'auc': auc}
        
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops)


def train_input_fn(training_dir, params):
    return read_dataset(filename=args['train_data_paths'],
                              mode=tf.estimator.ModeKeys.TRAIN,
                              batch_size=args['batch_size'])


def eval_input_fn(training_dir, params):
    return _input_fn(training_dir, 'test.tfrecords', batch_size=100)


def _input_fn(training_dir, training_filename, batch_size=100):
    test_file = os.path.join(training_dir, training_filename)
    filename_queue = tf.train.string_input_producer([test_file])

    image, label = read_and_decode(filename_queue)
    images, labels = tf.train.batch(
        [image, label], batch_size=batch_size,
        capacity=1000 + 3 * batch_size)

    return {INPUT_TENSOR_NAME: images}, labels