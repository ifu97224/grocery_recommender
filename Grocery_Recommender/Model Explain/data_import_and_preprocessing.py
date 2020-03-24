import boto3
import sagemaker
import pandas as pd
import numpy as np
from sagemaker import get_execution_role

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

def download_data(bucket, train_key, test_key, feat_list):
    """Function to download training and test DataFrames from S3

        Parameters
        ----------
        bucket : str
            String containing the S3 bucket name where the files are located
        train_key : str
            String that contains the name of the training file
        test_key : str
            String that contains the name of the test file
        feat_list : list
            list of features chosen through the feature selection module

        Returns
        -------
        train_df : pandas.DataFrame
            DataFrame containing the training data
        test_df:  pandas.DataFrame
            DataFrame containing the test data

        """

    role = get_execution_role()
    region = boto3.Session().region_name
    s3c = boto3.client("s3")
    bucket = bucket
    train_key = train_key
    test_key = test_key

    # Train
    train_key = train_key
    obj = s3c.get_object(Bucket=bucket, Key=train_key)
    train_df = pd.read_csv(obj["Body"])
    train_df = train_df[["TARGET"] + feat_list]
    train_df.columns = train_df.columns.str.replace(" ", "_")

    # Test
    test_key = test_key
    obj = s3c.get_object(Bucket=bucket, Key=test_key)
    test_df = pd.read_csv(obj["Body"])
    test_df = test_df[["TARGET"] + feat_list]
    test_df.columns = test_df.columns.str.replace(" ", "_")

    # Replace all inf values with 0
    train_df = train_df.replace([np.inf, -np.inf], 0)
    test_df = test_df.replace([np.inf, -np.inf], 0)

    return train_df, test_df

def preprocess(train_df, test_df, split_ratio):
    """Function to split the training data into training and testing (saving the current
    test DataFrame for validation).  Standardizes the numeric features and one-hot encodes
    the categorical features

        Parameters
        ----------
        train_df : Pandas DataFrame
            Pandas DataFrame containing the training data
        test_df : Pandas DataFrame
            Pandas DataFrame containing the test data
        split_ratio : float
            ratio to use to split the training data into training and testing

        Returns
        -------
        train_df : pandas.DataFrame
            DataFrame containing the training data
        test_df:  pandas.DataFrame
            DataFrame containing the test data

        """

    valid_df = test_df

    split_ratio = split_ratio
    train_df, test_df = train_test_split(
        train_df, test_size=split_ratio, random_state=0
    )

    ### Standardize the numeric features
    int_float_features = train_df.select_dtypes(
        include=["int", "float64"]
    ).columns.to_list()
    int_float_features = [elem for elem in int_float_features if not elem in ("TARGET")]

    # Standardize numeric features and one-hot encode categorical features
    preprocess = make_column_transformer((StandardScaler(), int_float_features))

    print("Standardizing numeric features")
    train_std = pd.DataFrame(preprocess.fit_transform(train_df))
    test_std = pd.DataFrame(preprocess.transform(test_df))
    valid_std = pd.DataFrame(preprocess.transform(valid_df))

    # Get the column names back
    train_std.columns = int_float_features
    test_std.columns = int_float_features
    valid_std.columns = int_float_features

    ### One-hot encode the categorical features
    
    print("One-hot encode the categorical features")
    # Get list of categorical features, these will need to one-hot encoded
    cat_features = train_df.select_dtypes(include=["object"]).columns.to_list()
    cat_features = [
        elem for elem in cat_features if not elem in ("CUST_CODE", "PROD_CODE")
    ]

    cat_df_train = train_df[cat_features]
    cat_df_test = test_df[cat_features]
    cat_df_valid = valid_df[cat_features]

    preprocess = make_column_transformer((OneHotEncoder(sparse=False), cat_features))

    train_cat_features = pd.DataFrame(preprocess.fit_transform(cat_df_train))
    test_cat_features = pd.DataFrame(preprocess.transform(cat_df_test))
    valid_cat_features = pd.DataFrame(preprocess.transform(cat_df_valid))

    train_cat_features.columns = preprocess.get_feature_names()
    test_cat_features.columns = preprocess.get_feature_names()
    valid_cat_features.columns = preprocess.get_feature_names()

    ### Merge the target, standardized numerical features and one-hot encoded categorical features
    target_df_train = train_df[["TARGET"]]
    target_df_test = test_df[["TARGET"]]
    target_df_valid = valid_df[["TARGET"]]

    train_df = train_cat_features.merge(train_std, left_index=True, right_index=True)
    train_df = train_df.merge(target_df_train, left_index=True, right_index=True)

    test_df = test_cat_features.merge(test_std, left_index=True, right_index=True)
    test_df = test_df.merge(target_df_test, left_index=True, right_index=True)
    test_df.head()

    valid_df = valid_cat_features.merge(valid_std, left_index=True, right_index=True)
    valid_df = valid_df.merge(target_df_valid, left_index=True, right_index=True)

    return train_df, test_df, valid_df
