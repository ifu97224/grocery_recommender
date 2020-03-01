import argparse
import tensorflow as tf
import pandas as pd
import os
from functools import partial
from keras.callbacks import EarlyStopping
import boto3

def get_feature_target(df):
    """Function creates lists of feature columns, target label and defaults

    Parameters
    ----------
    df : pandas.DataFrame
        Pandas DataFrame containing examples of the feature columns and target

    Returns
    -------
    csv_cols : list
        list of feature columns
    csv_label_col : list
        list of target columns

    """

    csv_cols = df.columns.to_list()
    csv_label_col = "TARGET"

    return csv_cols, csv_label_col


# Create feature columns to be used in model
def create_feature_columns(df):
    """Function to create feature columns that will be used in the model

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the training data

    Returns
    -------
    feature_columns : list
        List of TensorFlow feature columns

    """
    
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

def get_dataset(file_path, **kwargs):
    """Function to import a csv file and create a dataset that can be fed into the TensorFlow model.
    Also acts as an iterator creating batches per the specified batch size

    Parameters
    ----------
    file_path : str
        Path to the training data csv

    Returns
    -------
        Dataset in TensorFlow format with the specified batch size

    """
    
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        args.batch_size,
        shuffle=True,
        shuffle_buffer_size=10 * args.batch_size,
        column_names=csv_cols,
        label_name=csv_label_col,
        num_epochs=1,
        **kwargs)

    return dataset

def train_and_evaluate(args):
    """Function to train the TensorFlow model and evaluate on a test set

    Parameters
    ----------
    args : dictionary
        Dictionary containing hyperparameter settings

    Returns
    -------
        model : tf.keras model object
            Fitted TensorFlow model object

    """

    # Import the training data
    training_dataset = get_dataset(train_data)
    test_dataset = get_dataset(test_data)

    # Set the batch norm params
    batch_norm_momentum = args.momentum
    batch_norm_layer = partial(tf.keras.layers.BatchNormalization, trainable=True, momentum=batch_norm_momentum)

    # Create the input layer based on the specified feature columns
    preprocessing_layer = tf.keras.layers.DenseFeatures(create_feature_columns(train_df))

    # Create the dense layer
    if args.initialization == 'RandomNormal':
        init = tf.keras.initializers.RandomNormal()
    elif args.initialization == 'RandomUniform':
        init = tf.keras.initializers.RandomUniform()
    elif args.initialization == 'he_normal':
        init = tf.keras.initializers.he_normal()
    elif args.initialization == 'he_uniform':
        init = tf.keras.initializers.he_uniform()
    
    dense_layer = partial(tf.keras.layers.Dense, kernel_initializer=init, activation='relu')
    
    # Drop out layer
    dropout_rate = args.dropout_rate
    dropout_layer = partial(tf.keras.layers.Dropout, rate=dropout_rate)
    
    # Create and compile the model
    model = tf.keras.Sequential()
    model.add(preprocessing_layer)
    
    for i in range(0, args.num_layers + 1):
        model.add(dense_layer(args.hidden_units)),
        model.add(batch_norm_layer()),
        model.add(dropout_layer())
    
    model.add(tf.keras.layers.Dense(1))

    optim_adam = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=optim_adam,
        metrics=['accuracy'])

    # Fit on the test set, evaluate on the training data
    early_stopping = EarlyStopping(patience=100, monitor='val_accuracy',min_delta=0.001, restore_best_weights=True, verbose=1)
    model.fit(training_dataset, epochs=args.epochs, validation_data=test_dataset, callbacks=[early_stopping])
        
    test_loss, test_acc = model.evaluate(test_dataset)
    print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_acc))
    
    return model
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--hidden_units", type=int)
    parser.add_argument("--model_dir", type = str)
    parser.add_argument("--num_layers", type = int)
    parser.add_argument("--dropout_rate", type=float)
    parser.add_argument("--momentum", type = float)
    parser.add_argument("--initialization", type = str)
    
    args = parser.parse_args()
    print("Received arguments {}".format(args))

    training_data_directory = "/opt/ml/input/data/train"
    train_data = os.path.join(training_data_directory, "train_df.csv")
    train_df = pd.read_csv(train_data)

    testing_data_directory = "/opt/ml/input/data/test"
    test_data = os.path.join(testing_data_directory, "test_df.csv")

    csv_cols, csv_label_col = get_feature_target(train_df)

    keras_model = train_and_evaluate(args)
    
    keras_model.save('/opt/ml/model/')