import os
import argparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.externals import joblib

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_features", type=str, default="auto")
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--tune", type=int, default=1)

    # Sagemaker specific arguments
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])

    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))

    # Train and fit on training set
    training_data_directory = "/opt/ml/input/data/train"
    train_features_data = os.path.join(training_data_directory, "train_features.csv")
    train_labels_data = os.path.join(training_data_directory, "train_labels.csv")
    print("Reading training input data")
    X_train = pd.read_csv(train_features_data, header=None)
    y_train = pd.read_csv(train_labels_data, header=None)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_features=args.max_features,
        max_depth=args.max_depth,
        random_state=0,
        class_weight="balanced",
        oob_score=True,
        n_jobs=-1,
    )

    print("Training Random Forest model")
    model.fit(X_train, y_train.values.ravel())

    # If tuning hyperparameters get the AUC on the test set
    tune_hyperparameters = args.tune
    if tune_hyperparameters == 1:
        # Predict and evaluate on test set
        test_data_directory = "/opt/ml/input/data/test"
        test_features_data = os.path.join(test_data_directory, "test_features.csv")
        test_labels_data = os.path.join(test_data_directory, "test_labels.csv")
        print("Reading test input data")

        X_test = pd.read_csv(test_features_data, header=None)
        y_test = pd.read_csv(test_labels_data, header=None)

        fpr_rfc, tpr_rfc, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        roc_auc_rfc = auc(fpr_rfc, tpr_rfc)
        print("ROC AUC RF: %0.5f" % roc_auc_rfc)

    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    model = joblib.load(os.path.join(model_dir, "model.joblib"))

    return model
