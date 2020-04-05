import boto3
import sagemaker
from sagemaker import get_execution_role

import pandas as pd

def create_model_input(
    bucket, train_key, test_key, out_file_train, out_file_test, feat_list
):
    """Function keeps only the specified features from the training and test sets that will be used for model developement

    Parameters
    ----------
    bucket : str
        s3 bucket name to read and write the data
    train_key : str
        name of the training data
    test_key : str
        name of the test data
    out_file_train : str
        name of the training output file
    out_file_test : str
        name of the test output file
    feat_list : list
        list containing the features to keep from the DataFrame (CUST_CODE, 'PROD_CODE', 'TARGET' and all other features)

    Returns
    -------
    input_data : str
        the file path to the data to be used for pre-processing

    """

    role = get_execution_role()
    region = boto3.Session().region_name
    s3c = boto3.client("s3")
    bucket = bucket

    # Train
    train_key = train_key
    obj = s3c.get_object(Bucket=bucket, Key=train_key)
    train_df = pd.read_csv(obj["Body"])
    train_df = train_df[["TARGET"] + feat_list]
    train_df.to_csv("./{}".format(out_file_train), header=True, index=False)
    boto3.Session().resource("s3").Bucket(bucket).Object(out_file_train).upload_file(
        "./{}".format(out_file_train)
    )

    # Test
    test_key = test_key
    obj = s3c.get_object(Bucket=bucket, Key=test_key)
    test_df = pd.read_csv(obj["Body"])
    test_df = test_df[["TARGET"] + feat_list]
    test_df.to_csv("./{}".format(out_file_test), header=True, index=False)
    boto3.Session().resource("s3").Bucket(bucket).Object(out_file_test).upload_file(
        "./{}".format(out_file_test)
    )

    # Create the input data paths
    input_data_train = "s3://{}/{}".format(bucket, out_file_train)
    input_data_test = "s3://{}/{}".format(bucket, out_file_test)

    return input_data_train, input_data_test