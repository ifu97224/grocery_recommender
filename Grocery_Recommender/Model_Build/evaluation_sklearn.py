import json
import os
import tarfile

import pandas as pd

from sklearn.externals import joblib
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

if __name__ == "__main__":
    model_path = os.path.join("/opt/ml/processing/model", "model.tar.gz")
    print("Extracting model from path: {}".format(model_path))
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    print("Loading model")
    model = joblib.load("model.joblib")

    print("Loading validation input data")
    valid_features_data = os.path.join("/opt/ml/processing/valid", "valid_features.csv")
    valid_labels_data = os.path.join("/opt/ml/processing/valid", "valid_labels.csv")

    X_valid = pd.read_csv(valid_features_data, header=None)
    y_valid = pd.read_csv(valid_labels_data, header=None)
    predictions = model.predict(X_valid)

    print("Creating classification evaluation report")
    report_dict = classification_report(y_valid, predictions, output_dict=True)
    report_dict["accuracy"] = accuracy_score(y_valid, predictions)
    report_dict["roc_auc"] = roc_auc_score(y_valid, predictions)

    print("Classification report:\n{}".format(report_dict))

    evaluation_output_path = os.path.join(
        "/opt/ml/processing/evaluation", "evaluation.json"
    )
    print("Saving classification report to {}".format(evaluation_output_path))

    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps(report_dict))