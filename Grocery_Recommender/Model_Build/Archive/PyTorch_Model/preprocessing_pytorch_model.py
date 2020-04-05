import argparse
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

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
    train_df, test_df = train_test_split(
        train_df, test_size=split_ratio, random_state=0
    )

    negative_examples, positive_examples = np.bincount(train_df.TARGET)
    print(
        "Train data contains: {} observations of which {} are positive examples and {} are negative examples".format(
            train_df.shape, positive_examples, negative_examples
        )
    )

    negative_examples, positive_examples = np.bincount(test_df.TARGET)
    print(
        "Test data contains: {} observations of which {} are positive examples and {} are negative examples".format(
            test_df.shape, positive_examples, negative_examples
        )
    )

    negative_examples, positive_examples = np.bincount(valid_df.TARGET)
    print(
        "Validation data contains: {} observations of which {} are positive examples and {} are negative examples".format(
            valid_df.shape, positive_examples, negative_examples
        )
    )

    # Get list of categorical features, these will need to one-hot encoded
    cat_features = train_df.select_dtypes(include=["object"]).columns.to_list()
    cat_features = [
        elem for elem in cat_features
    ]

    print('Categorical features {}'.format(cat_features))
    
    # Get list of numerical features, these will be standardized
    int_float_features = train_df.select_dtypes(
        include=["int", "float64"]
    ).columns.to_list()
    int_float_features = [elem for elem in int_float_features if not elem in ("TARGET")]

    print('Numeric features {}'.format(int_float_features))
    
    # Standardize numeric features 

    print("Standardizing numeric features")
    std_scaler = StandardScaler()
  
    train_numeric = pd.DataFrame(std_scaler.fit_transform(train_df[int_float_features]))
    train_numeric.columns = int_float_features
    
    test_numeric = pd.DataFrame(std_scaler.transform(test_df[int_float_features]))
    test_numeric.columns = int_float_features
    
    valid_numeric = pd.DataFrame(std_scaler.transform(valid_df[int_float_features]))
    valid_numeric.columns = int_float_features
    
    
    # One-hot encode categorical features
    print("One hot encoding categorical features")
    one_hot_enc = OneHotEncoder(sparse=False)
    
    train_cat = pd.DataFrame(one_hot_enc.fit_transform(train_df[cat_features]))
    train_cat.columns = one_hot_enc.get_feature_names()

    test_cat = pd.DataFrame(one_hot_enc.transform(test_df[cat_features]))
    test_cat.columns = one_hot_enc.get_feature_names()
    
    valid_cat = pd.DataFrame(one_hot_enc.transform(valid_df[cat_features]))
    valid_cat.columns = one_hot_enc.get_feature_names()

    # Merge numeric and categorical features together
    train_std = train_numeric.merge(train_cat, left_index=True, right_index=True)
    test_std = test_numeric.merge(test_cat, left_index=True, right_index=True)
    valid_std = train_numeric.merge(valid_cat, left_index=True, right_index=True)
    
    # Append back the target
    train_df = train_df[["TARGET"]].merge(
        train_std, left_index=True, right_index=True
    )
    test_df = test_df[["TARGET"]].merge(
        test_std, left_index=True, right_index=True
    )
    valid_df = valid_df[["TARGET"]].merge(
        valid_std, left_index=True, right_index=True
    )

    print("Train data shape after standardizing: {}".format(train_df.shape))
    print("Test data shape after standardizing: {}".format(test_df.shape))
    print("Validation data shape after standardizing: {}".format(valid_df.shape))

    train_df_output_path = os.path.join("/opt/ml/processing/train", "train_df.csv")
    test_df_output_path = os.path.join("/opt/ml/processing/test", "test_df.csv")
    valid_df_output_path = os.path.join("/opt/ml/processing/valid", "valid_df.csv")
    print("Saving training DataFrame to {}".format(train_df_output_path))
    pd.DataFrame(train_df).to_csv(train_df_output_path, header=True, index=False)

    print("Saving test DataFrame to {}".format(test_df_output_path))
    pd.DataFrame(test_df).to_csv(test_df_output_path, header=True, index=False)

    print("Saving validation DataFrame to {}".format(valid_df_output_path))
    pd.DataFrame(valid_df).to_csv(valid_df_output_path, header=True, index=False)