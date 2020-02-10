import argparse
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelBinarizer,
    KBinsDiscretizer,
)
from sklearn.compose import make_column_transformer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-test-split-ratio", type=float, default=0.3)
    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))

    input_train_data_path = os.path.join(
        "/opt/ml/processing/input_data_train", "train_df_final_features.csv"
    )
    print("Reading input training data from {}".format(input_train_data_path))
    train_df = pd.read_csv(input_train_data_path)

    # Replace all inf values with 0
    train_df = train_df.replace([np.inf, -np.inf], 0)

    input_test_data_path = os.path.join(
        "/opt/ml/processing/input_data_test", "test_df_final_features.csv"
    )
    print("Reading input test data from {}".format(input_test_data_path))

    # Calling the test data validation as the training data will be further split into train and test
    valid_df = pd.read_csv(input_test_data_path)

    # Replace all inf values with 0
    valid_df = train_df.replace([np.inf, -np.inf], 0)

    # Further splitting the training data into training and testing
    split_ratio = args.train_test_split_ratio
    print(
        "Further splitting the training data into train and test sets with ratio {}".format(
            split_ratio
        )
    )
    X_train, X_test, y_train, y_test = train_test_split(
        train_df.drop("TARGET", axis=1),
        train_df["TARGET"],
        test_size=split_ratio,
        random_state=0,
    )

    # Create the valiation sets
    X_valid = valid_df.drop("TARGET", axis=1)
    y_valid = valid_df["TARGET"]

    negative_examples, positive_examples = np.bincount(y_train)
    print(
        "Train data contains: {} observations of which {} are positive examples and {} are negative examples".format(
            train_df.shape, positive_examples, negative_examples
        )
    )

    negative_examples, positive_examples = np.bincount(y_test)
    print(
        "Test data contains: {} observations of which {} are positive examples and {} are negative examples".format(
            X_test.shape, positive_examples, negative_examples
        )
    )

    negative_examples, positive_examples = np.bincount(y_valid)
    print(
        "Validation data contains: {} observations of which {} are positive examples and {} are negative examples".format(
            valid_df.shape, positive_examples, negative_examples
        )
    )

    # Get list of categorical features, these will need to one-hot encoded
    cat_features = train_df.select_dtypes(include=["object"]).columns.to_list()
    cat_features = [
        elem for elem in cat_features if not elem in ("CUST_CODE", "PROD_CODE")
    ]

    # Get list of numerical features, these will be standardized
    int_float_features = train_df.select_dtypes(
        include=["int", "float64"]
    ).columns.to_list()
    int_float_features = [elem for elem in int_float_features if not elem in ("TARGET")]

    # Standardize numeric features and one-hot encode categorical features
    preprocess = make_column_transformer(
        (int_float_features, StandardScaler()),
        (cat_features, OneHotEncoder(sparse=False)),
    )

    print("Standardizing numeric features and one-hot encoding categorical features")
    train_features = preprocess.fit_transform(X_train)
    test_features = preprocess.transform(X_test)
    valid_features = preprocess.transform(X_valid)

    print(
        "Train data shape after standardizing and one-hot encoding: {}".format(
            train_features.shape
        )
    )
    print(
        "Test data shape after standardizing and one-hot encoding: {}".format(
            test_features.shape
        )
    )
    print(
        "Validation data shape after standardizing and one-hot encoding: {}".format(
            valid_features.shape
        )
    )

    train_features_output_path = os.path.join(
        "/opt/ml/processing/train", "train_features.csv"
    )
    train_labels_output_path = os.path.join(
        "/opt/ml/processing/train", "train_labels.csv"
    )

    test_features_output_path = os.path.join(
        "/opt/ml/processing/test", "test_features.csv"
    )
    test_labels_output_path = os.path.join("/opt/ml/processing/test", "test_labels.csv")

    valid_features_output_path = os.path.join(
        "/opt/ml/processing/valid", "valid_features.csv"
    )
    valid_labels_output_path = os.path.join(
        "/opt/ml/processing/valid", "valid_labels.csv"
    )

    print("Saving training features to {}".format(train_features_output_path))
    pd.DataFrame(train_features).to_csv(
        train_features_output_path, header=False, index=False
    )

    print("Saving test features to {}".format(test_features_output_path))
    pd.DataFrame(test_features).to_csv(
        test_features_output_path, header=False, index=False
    )

    print("Saving validation features to {}".format(valid_features_output_path))
    pd.DataFrame(valid_features).to_csv(
        valid_features_output_path, header=False, index=False
    )

    print("Saving training labels to {}".format(train_labels_output_path))
    y_train.to_csv(train_labels_output_path, header=False, index=False)

    print("Saving test labels to {}".format(test_labels_output_path))
    y_test.to_csv(test_labels_output_path, header=False, index=False)

    print("Saving validation labels to {}".format(valid_labels_output_path))
    y_valid.to_csv(valid_labels_output_path, header=False, index=False)