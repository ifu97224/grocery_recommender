import pandas as pd
import numpy as np
import boto3

def iterate_bucket_items(bucket):
    """Generator that iterates over all objects in a given s3 bucket

    See http://boto3.readthedocs.io/en/latest/reference/services/s3.html#S3.Client.list_objects_v2
    for return data format

    Parameters
    ----------
    bucket : str
        name of s3 bucket

    """

    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket)

    for page in page_iterator:
        if page["KeyCount"] > 0:
            for item in page["Contents"]:
                yield item


def import_data(bucket):
    """Function to import all data csv's from the S3 bucket

    Parameters
    ----------
    bucket : str
        name of s3 bucket

    Returns
    -------
    time_data : pandas.DataFrame
       DataFrame containing calendar lookup
    all_trans : pandas.DataFrame
        Pandas DataFrames containing customer transactions for all weeks

    """

    # Set up empty base DataFrame
    dtypes = np.dtype(
        [
            ("SHOP_WEEK", int),
            ("SHOP_DATE", int),
            ("SHOP_WEEKDAY", int),
            ("SHOP_HOUR", int),
            ("QUANTITY", int),
            ("SPEND", float),
            ("PROD_CODE", str),
            ("PROD_CODE_10", str),
            ("PROD_CODE_20", str),
            ("PROD_CODE_30", str),
            ("PROD_CODE_40", str),
            ("CUST_CODE", str),
            ("CUST_PRICE_SENSITIVITY", str),
            ("CUST_LIFESTAGE", str),
            ("BASKET_ID", int),
            ("BASKET_SIZE", str),
            ("BASKET_PRICE_SENSITIVITY", str),
            ("BASKET_TYPE", str),
            ("BASKET_DOMINANT_MISSION", str),
            ("STORE_CODE", str),
            ("STORE_FORMAT", str),
            ("STORE_REGION", str),
        ]
    )

    data = np.empty(0, dtype=dtypes)
    all_trans = pd.DataFrame(data)

    # Fill DataFrame
    for i in iterate_bucket_items(bucket=bucket):
        if 'udacity_capstone_data' in i["Key"]:
        
            print("Importing data for {}".format(i["Key"]))
            data_location = "s3://{}/{}".format(bucket, i["Key"])
            print(data_location)
            print(i["Key"])

            if i["Key"] == "udacity_capstone_data/time.csv":
                time_data = pd.read_csv(data_location)

            else:
                trans_data = pd.read_csv(data_location)
                all_trans = pd.concat([all_trans, trans_data], axis=0)

    return time_data, all_trans