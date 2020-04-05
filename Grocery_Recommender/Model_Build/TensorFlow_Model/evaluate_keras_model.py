import tensorflow as tf
import pandas as pd
import numpy as np
import shutil
import os
import tarfile
import boto3
from urllib.parse import urlparse
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix

def tf_model_evaluate(model_data_s3_uri, valid_data_path, valid_filename):
    """Function to create predictions on a trained keras model and return a classification report
    showing model performance

    Parameters
    ----------
    model_data_s3_uri : str
        Path to the location holding the trained model
    valid_data_path : str
        Path to the location of the validation data
    valid_filename : str
        Name of the csv containing the validation data

    Returns
    -------
        Dictionary containing the classification report

    """
    
    # Download the trained model
    parsed_url = urlparse(model_data_s3_uri)
    bucket = parsed_url.netloc
    key = os.path.join(parsed_url.path[1:])
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, "./Saved_TensorFlow_Model/model.tar.gz")

    with tarfile.open("./Saved_TensorFlow_Model/model.tar.gz") as tar:
        tar.extractall(path="./Saved_TensorFlow_Model/")

    model = tf.keras.models.load_model("./Saved_TensorFlow_Model")

    valid_df = pd.read_csv(valid_data_path + "/{}".format(valid_filename))
    print("Validation DataFrame shape: {}".format(valid_df.shape))

    # Get the feature columns and label
    csv_cols = valid_df.columns.to_list()
    csv_label_col = "TARGET"

    # Download the validation data
    parsed_url = urlparse(valid_data_path)
    bucket = parsed_url.netloc
    key = os.path.join(parsed_url.path[1:], valid_filename)
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, "./{}".format(valid_filename))

    # Get the dataset into TensorFlow format and set up the iterator
    dataset = tf.data.experimental.make_csv_dataset(
        "./{}".format(valid_filename),
        1000,
        shuffle=False,
        column_names=csv_cols,
        label_name=csv_label_col,
        num_epochs=1,
    )

    # Get the predictions
    logits = model.predict(dataset)
    odds = np.exp(logits)
    prob = odds / (1 + odds)
    predictions = pd.DataFrame(np.where(prob > 0.5, 1, 0))

    y_valid = valid_df[["TARGET"]]

    # Create the classification report
    report_dict = classification_report(y_valid, predictions, output_dict=True)
    report_dict["accuracy"] = accuracy_score(y_valid, predictions)
    report_dict["roc_auc"] = roc_auc_score(y_valid, predictions)

    return print("Classification report:\n{}".format(report_dict))