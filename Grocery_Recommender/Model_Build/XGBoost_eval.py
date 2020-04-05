import json
import os
import io
import boto3
from urllib.parse import urlparse
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score


def evaluate_xgboost(tuned_model, val_data_location, pred_out, proba_cutoff, Y_val):
    """Function to evaluate the XGBoost model on the validation set

    Parameters
    ----------
    tuned_model : Sagemaker.tuner
        tuned Sagemaker hyperparameter object with best training job attached
    val_data_location : str
        path to the validation data
    pred_out : str
        path to the directory to save the predictions
    proba_cutoff : float
        cut-off value to use to flag positive target e.g. probability values >0.5 are identified as TARGET = 1
    Y_val : pandas.DataFrame
        DataFrame containing the target variable for the validation set

    Returns
    -------
        dictionary from the SKLearn classification_report

    """

    xgb_transformer = tuned_model.transformer(
        instance_count=1, instance_type="ml.m4.xlarge"
    )
    xgb_transformer.transform(
        val_data_location, content_type="text/csv", split_type="Line"
    )
    xgb_transformer.wait()

    def get_csv_output_from_s3(s3uri, file_name):
        parsed_url = urlparse(s3uri)
        bucket_name = parsed_url.netloc
        prefix = parsed_url.path[1:]
        s3 = boto3.resource("s3")
        obj = s3.Object(bucket_name, "{}/{}".format(prefix, file_name))
        return obj.get()["Body"].read().decode("utf-8")

    output = get_csv_output_from_s3(
        xgb_transformer.output_path, "{}.out".format("validation.csv")
    )
    
    Y_pred = pd.read_csv(io.StringIO(output), sep=",", header=None)
    Y_pred.columns = ["proba"]
    Y_pred.loc[Y_pred["proba"] > proba_cutoff, "prediction"] = 1
    Y_pred.fillna(0, inplace=True)
    pred = Y_pred[["prediction"]]

    print("Creating classification evaluation report")
    report_dict = classification_report(Y_val, pred, output_dict=True)
    report_dict["accuracy"] = accuracy_score(Y_val, pred)
    report_dict["roc_auc"] = roc_auc_score(Y_val, pred)

    return print("Classification report:\n{}".format(report_dict))