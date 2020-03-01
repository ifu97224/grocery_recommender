import argparse
import tensorflow as tf
import pandas as pd
import os
from functools import partial

def get_defaults_feature_target(df):
    """Function creates lists of feature columns, target label and defaults

    Parameters
    ----------
    df : pandas.DataFrame
        Pandas DataFrame containing examples of the feature columns and target

    Returns
    -------
    csv_feat_cols : list
        list of feature columns
    csv_label_col : list
        list of target columns
    defaults : str
        string containing the target label

    """

    csv_feat_cols = df.drop("TARGET", axis=1).columns.to_list()
    csv_label_col = "TARGET"

    # Get list of numerical features to create defaults
    int_float_features = df.select_dtypes(include=["int", "float64"]).columns.to_list()
    int_float_features = [elem for elem in int_float_features if not elem in ("TARGET")]

    # Get list of categorical features to create defaults
    cat_features = df.select_dtypes(include=["object"]).columns.to_list()
    cat_defaults = [["UN"] for i in range(0, len(cat_features))]

    # Set default values for each CSV column
    int_float_defaults = [[0.0] for i in range(0, len(int_float_features))]

    defaults = cat_defaults + int_float_defaults

    return csv_feat_cols, csv_label_col, defaults


def read_dataset(filename, mode, batch_size=512):
    def _input_fn():
        def decode_csv(value_column):
            columns = tf.io.decode_csv(records=value_column, record_defaults=defaults)
            features = dict(zip(csv_feat_cols, columns))
            # label = features.pop(csv_label_col)
            label = csv_label_col
            return features, label

        # Create list of files that match pattern
        # file_list = tf.gfile.Glob(filename=filename)

        # Create dataset from file list
        # dataset = tf.data.TextLineDataset(filenames=file_list).map(map_func=decode_csv)
        dataset = tf.data.TextLineDataset(filenames=filename).map(map_func=decode_csv)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None  # indefinitely
            dataset = dataset.shuffle(buffer_size=10 * batch_size)
        else:
            num_epochs = 1

        dataset = dataset.repeat(count=num_epochs).batch(batch_size=batch_size)
        return next(iter(dataset))

    return _input_fn

# Create feature columns to be used in model
def create_feature_columns(df):
    # Create the customer price sensitivity column
    categorical_price_sens_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key="CUST_PRICE_SENSITIVITY",
        vocabulary_list=("MM", "LA", "UM", "XX"),
        num_oov_buckets=1,
    )

    # Convert price sensitivity column into indicator column
    indicator_price_sens_column = tf.feature_column.indicator_column(
        categorical_column=categorical_price_sens_column
    )

    # Create the customer customer lifestage column
    categorical_cust_lifestage_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key="CUST_LIFESTAGE",
        vocabulary_list=("OT", "XX", "YF", "YA", "OA", "PE", "OF"),
        num_oov_buckets=1,
    )

    # Convert customer lifestage column into indicator column
    indicator_cust_lifestage_column = tf.feature_column.indicator_column(
        categorical_column=categorical_cust_lifestage_column
    )

    # Create all other numeric feature columns
    int_float_features = df.select_dtypes(include=["int", "float64"]).columns.to_list()
    int_float_features = [elem for elem in int_float_features if not elem in ("TARGET")]

    numeric_feat_cols = [
        tf.feature_column.numeric_column(key=feat) for feat in int_float_features
    ]

    feature_columns = [
        indicator_price_sens_column,
        indicator_cust_lifestage_column,
    ] + numeric_feat_cols

    return feature_columns


# Create the model function
def model_fn(features, mode, params):
    # Set the batch norm params
    batch_norm_momentum = 0.99

    # Create neural network input layer using our feature columns defined above
    net = tf.feature_column.input_layer(
        features=features, feature_columns=params["feature_columns"]
    )

    he_init = tf.variance_scaling_initializer()

    if mode == tf.estimator.ModeKeys.TRAIN:
        batch_norm_layer = partial(
            tf.layers.batch_normalization, training=True, momentum=batch_norm_momentum
        )

    else:
        batch_norm_layer = partial(
            tf.layers.batch_normalization, training=False, momentum=batch_norm_momentum
        )

    dense_layer = partial(tf.layers.dense, kernel_initializer=he_init)

    # Create hidden and batch norm layers
    hidden1 = dense_layer(inputs=net, units=params["hidden_units_1"])
    bn1 = tf.nn.relu(batch_norm_layer(hidden1))
    hidden2 = dense_layer(inputs=bn1, units=params["hidden_units_2"])
    bn2 = tf.nn.relu(batch_norm_layer(hidden2))
    hidden3 = dense_layer(inputs=bn2, units=params["hidden_units_3"])
    bn3 = tf.nn.relu(batch_norm_layer(hidden3))

    # Compute logits using the output of our last hidden layer
    logits = tf.layers.dense(inputs=bn3, units=1, activation=None)

    # If the mode is prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        # Create predictions dict
        predictions_dict = {"probabilities": tf.nn.sigmoid(logits), "logits": logits}

        # Create export outputs
        export_outputs = {
            "predict_export_outputs": tf.estimator.export.PredictOutput(
                outputs=predictions_dict
            )
        }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions_dict,
            loss=None,
            train_op=None,
            eval_metric_ops=None,
            export_outputs=export_outputs,
        )

    # Continue on with training and evaluation modes

    # Compute loss using sigmoid cross entropy since this is classification and our labels
    # and probabilities are mutually exclusive
    labels = tf.cast(logits, tf.float32)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=tf.squeeze(logits)
    )

    predicted_class = tf.cast(tf.round(tf.nn.sigmoid(logits)), tf.float32)

    # Compute evaluation metrics of total accuracy and auc
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=predicted_class, name="acc_op"
    )

    auc = tf.metrics.auc(labels=labels, predictions=predicted_class, name="auc")

    # Put eval metrics into a dictionary
    eval_metrics = {"accuracy": accuracy, "auc": auc}

    # Create scalar summaries to see in TensorBoard
    tf.summary.scalar(name="accuracy", tensor=accuracy[1])
    tf.summary.scalar(name="auc", tensor=auc[1])

    logging_hook = tf.train.LoggingTensorHook(
        {"loss": loss, "accuracy": accuracy[1], "auc": auc[1]}, every_n_iter=100
    )

    # If the mode is evaluation
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=None,
            loss=loss,
            train_op=None,
            eval_metric_ops=eval_metrics,
            export_outputs=None,
        )

    # Continue on with training mode (if mode is training)
    assert mode == tf.estimator.ModeKeys.TRAIN

    # Create a custom optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])

    # Create train op
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=None,
        loss=loss,
        training_hooks=[logging_hook],
        train_op=train_op,
        eval_metric_ops=None,
        export_outputs=None,
    )


def train_and_evaluate(args):
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.model_dir,
        params={
            "feature_columns": create_feature_columns(train_df),
            "hidden_units_1": args.hidden_units_1,
            "hidden_units_2": args.hidden_units_2,
            "hidden_units_3": args.hidden_units_3,
            "learning_rate": args.learning_rate,
        },
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=read_dataset(
            filename=train_data,
            mode=tf.estimator.ModeKeys.TRAIN,
            batch_size=args.batch_size,
        ),
        max_steps=args.train_steps,
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=read_dataset(
            filename=test_data,
            mode=tf.estimator.ModeKeys.EVAL,
            batch_size=args.batch_size,
        ),
        steps=None,
        start_delay_secs=args.start_delay_secs,
        throttle_secs=args.throttle_secs,
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--hidden_units_1", type=int)
    parser.add_argument("--hidden_units_2", type=int)
    parser.add_argument("--hidden_units_3", type=int)
    parser.add_argument("--train_steps", type=int)
    parser.add_argument("--start_delay_secs", type=int)
    parser.add_argument("--throttle_secs", type=int)

    args = parser.parse_args()
    print("Received arguments {}".format(args))

    training_data_directory = "/opt/ml/input/data/train"
    train_data = os.path.join(training_data_directory, "train_df.csv")
    train_df = pd.read_csv(train_data)

    testing_data_directory = "/opt/ml/input/data/test"
    test_data = os.path.join(testing_data_directory, "test_df.csv")
    
    csv_feat_cols, csv_label_col, defaults = get_defaults_feature_target(train_df)

    train_and_evaluate(args)