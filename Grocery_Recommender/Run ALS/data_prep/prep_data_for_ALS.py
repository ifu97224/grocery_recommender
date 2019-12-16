import shutil
import pandas as pd
import numpy as np
import pickle
import boto3
from sagemaker import get_execution_role
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import scipy.sparse as sparse

class als_data_prep():
    """ The als_data_prep class contains all functions necessary to prep the data for input into the ALS model"""
    
    def __init__(self, df, start_wk, end_wk):
        """ Method for initializing the als_data_prep object

        Args:
        df:  The input DataFrame
        start_wk: The start week to use for the ALS fit
        end_wk:  The end week to use for the ALS fit

        """
        self.df = df
        self.start_wk = start_wk
        self.end_wk = end_wk
        
    
    def create_cust_quant_summ(self):
        """
        Function to summarize the customer transaction file at the customer and product level summing the quantity
        of units purchased.  The quantity of units purchased will be used as implicit feedback for the ALS model

        :param df: name of the customer transaction file
        :param start_wk: the start week to use to summarize
        :param end_wk:  the end week to use to summarize
        :return: DataFrame containing CUST_CODE, PROD_CODE and TOT_QUANTITY (total number of units of that product
                 purchased)

        """

        # Remove any records without a CUST_CODE - these are non-customer records where Loyalty cards were
        # not scanned
        df = self.df.dropna(subset=["CUST_CODE"], axis=0)

        # Keep only records after the specified start_wk
        df = df[(df["SHOP_WEEK"] >= self.start_wk) & (df["SHOP_WEEK"] <= self.end_wk)]

        # Summarize the data to CUST_CODE and PROD_CODE by QUANTITY
        df = df.groupby(["CUST_CODE", "PROD_CODE"])["QUANTITY"].sum().reset_index()

        return df


    def create_train_test_split(self, df):
        """
        Function to create training, testing and validation DataFrames

        :param df: name of the file to be split
        :return: DataFrames for training, testing and validation

        """

        unique_custs = df[["CUST_CODE"]].drop_duplicates()

        # First get training set
        cust_summ_train, cust_summ_test = train_test_split(
            unique_custs, test_size=0.33, random_state=42
        )

        # Now split test set into test and validation
        cust_summ_test, cust_summ_valid = train_test_split(
            cust_summ_test, test_size=0.50, random_state=42
        )

        print("Number of customers in training set {}".format(cust_summ_train.count()[0]))
        print("Number of customers in test set {}".format(cust_summ_test.count()[0]))
        print("Number of customers in validation set {}".format(cust_summ_valid.count()[0]))

        # Merge back to the main input DataFrame
        cust_summ_train = cust_summ_train.merge(df, on="CUST_CODE", how="inner")
        cust_summ_test = cust_summ_test.merge(df, on="CUST_CODE", how="inner")
        cust_summ_valid = cust_summ_valid.merge(df, on="CUST_CODE", how="inner")

        return cust_summ_train, cust_summ_test, cust_summ_valid


    def winsorize_df(self, train_df, test_df, valid_df, cols, lower, upper, test_set, valid_set):
        """
        Function to winsorize numeric values in a DataFrame to remove potential outliers

        :param train_df: DataFrame containing training set customers
        :param test_df: DataFrame containing test set customers
        :param valid_df: DataFrame containing validation set customers
        :param cols: List of columns to winsorize
        :param lower: Lower value e.g. 0.05 will cap the data at the 5th percentile
        :param upper: Upper value e.g. 0.05 will cap the data at the 95th percentile
        :param test_set: Boolean value indicating if a test set has been provided
        :param valid_set: Boolean value indicating if a training set has been provided
        :return: List containing winsorized DataFrames for train, test and validation sets

        """

        for i in range(0, len(cols)):

            winsor = pd.DataFrame(winsorize(train_df[cols[i]], limits=(lower, upper)))
            winsor.columns = [cols[i]]

            winsor_min = winsor[cols[i]].min()
            winsor_max = winsor[cols[i]].max()

            # Replace the column with the winsorized version for the training data
            train_df.drop(cols[i], axis=1)
            pd.concat([train_df, winsor], axis=1)

            # Now replace in the test and validation sets using the values from the training data
            if test_set:
                test_df.loc[test_df[cols[i]] > winsor_max, cols[i]] = winsor_max
                test_df.loc[test_df[cols[i]] < winsor_min, cols[i]] = winsor_min

            if valid_set:
                valid_df.loc[valid_df[cols[i]] > winsor_max, cols[i]] = winsor_max
                valid_df.loc[valid_df[cols[i]] < winsor_min, cols[i]] = winsor_min

            return train_df, test_df, valid_df


    def normalize_min_max(
        self, train_df, test_df, valid_df, group_var, normal_var, test_set, valid_set
    ):
        """Function to normalize a DataFrame numeric column using a min / max scaler

        :param train_df: DataFrame containing training set customers
        :param test_df: DataFrame containing test set customers
        :param valid_df: DataFrame containing validation set customers
        :param group_var: Grouping variable to normalize within
        :param normal_var: Variable to normalize
        :param test_set: Boolean value indicating if a test set has been provided
        :param valid_set: Boolean value indicating if a training set has been provided
        :return: List containing normalized DataFrames

        """

        # Get the min and max for the column to be normalized
        min_var = train_df.groupby([group_var]).agg({normal_var: "min"}).reset_index()
        min_var.columns = [group_var, "min_var"]

        max_var = train_df.groupby([group_var]).agg({normal_var: "max"}).reset_index()
        max_var.columns = [group_var, "max_var"]

        train_df = train_df.merge(min_var, on=group_var, how="inner")
        train_df = train_df.merge(max_var, on=group_var, how="inner")

        # Normalize
        train_df.loc[:, normal_var] = (train_df[normal_var] - train_df["min_var"]) / (
            train_df["max_var"] - train_df["min_var"]
        )

        # Normalized score can be NaN if the max purchases for a product is 1 and customers purchased 1 (the denominator
        # will be 0. Replace with 0
        train_df[normal_var].fillna(0, inplace=True)

        # To distinguish a purchase from no purchase add 0.01 to all records
        train_df.loc[:, normal_var] = train_df[normal_var] + 0.01

        # drop variables not needed
        train_df.drop(["min_var", "max_var"], axis=1, inplace=True)

        # Normalize the test set if provided
        if test_set:
            test_df = test_df.merge(min_var, on=group_var, how="inner")
            test_df = test_df.merge(max_var, on=group_var, how="inner")
            test_df.loc[:, normal_var] = (test_df[normal_var] - test_df["min_var"]) / (
                test_df["max_var"] - test_df["min_var"]
            )
            test_df[normal_var].fillna(0, inplace=True)

            # To distinguish a purchase from no purchase add 0.01 to all records
            test_df.loc[:, normal_var] = test_df[normal_var] + 0.01

            test_df.drop(["min_var", "max_var"], axis=1, inplace=True)

        # Normalize the validation set if provided
        if valid_set:
            valid_df = valid_df.merge(min_var, on=group_var, how="inner")
            valid_df = valid_df.merge(max_var, on=group_var, how="inner")
            valid_df.loc[:, normal_var] = (valid_df[normal_var] - valid_df["min_var"]) / (
                valid_df["max_var"] - valid_df["min_var"]
            )
            valid_df[normal_var].fillna(0, inplace=True)

            # To distinguish a purchase from no purchase add 0.01 to all records
            valid_df.loc[:, normal_var] = valid_df[normal_var] + 0.01

            valid_df.drop(["min_var", "max_var"], axis=1, inplace=True)

        return train_df, test_df, valid_df


    def create_als_train_test(self, df, frac):
        """
        Function to create training and test sets for input to ALS.  As ALS needs all customers to build the model
        the function chooses random customer/product rows to be held out as a test set

        :param df: DataFrame containing the customer, product and quantity
        :param frac: Percentage of total records to hold out for testing
        :return test_df: DataFrame for testing
        :return train_df: DataFrame for training
        """

        # Create a group key - the concatenation of the CUST_CODE and the PROD_CODE
        df.loc[:, "GROUP_KEY"] = df["CUST_CODE"] + df["PROD_CODE"]

        # Sample records to create the test DataFrame
        test_group_key = df["GROUP_KEY"].sample(
            frac=frac, replace=False, random_state=1234, axis=0
        )
        test_group_key = pd.DataFrame(test_group_key)
        test_group_key.columns = ["GROUP_KEY"]
        test_group_key.loc[:, "IN_TEST_GROUP"] = 1
        test_df = df.merge(test_group_key, on=["GROUP_KEY"], how="inner")

        # Create the training DataFrame by removing the customer / product combinations selected for the test set
        train_df = df.merge(test_group_key, on=["GROUP_KEY"], how="left")
        train_df = train_df.loc[train_df["IN_TEST_GROUP"].isnull()]

        # Drop columns no longer needed
        test_df.drop(["GROUP_KEY", "IN_TEST_GROUP"], axis=1, inplace=True)
        train_df.drop(["GROUP_KEY", "IN_TEST_GROUP"], axis=1, inplace=True)

        # Reset the index
        test_df.reset_index(inplace=True, drop=True)
        train_df.reset_index(inplace=True, drop=True)

        return test_df, train_df


    # Create the customer and item mapping
    def create_mapping(self, df, custs_items):
        """
        Function to create a customer / item index mapping file

        :param df: DataFrame containing the customer and item file to be mapped
        :return: Mapping DataFrame
        """

        df = df[[custs_items]].drop_duplicates()
        df.sort_values(custs_items, inplace=True)
        df = df.reset_index()
        df.loc[:, "index_" + custs_items] = df.index
        df = df[[custs_items, "index_" + custs_items]]

        return df


    def merge_mapping(self, df, cust_map, item_map):
        """
        Function to merge the customer and item mapping files

        :param df: DataFrame containing the customer and item file to be mapped
        :param cust_map:  DataFrame containing the customer ID and the integer index to map
        :param item_map:  DataFrame containing the product code and the integer index to map
        :return: Mapping DataFrame

        """

        df = df.merge(cust_map, on="CUST_CODE")
        df = df.merge(item_map, on="PROD_CODE")

        return df


    def create_sparse_matrix(self, df):
        """
        Function to create the sparse matrix required by the ALS algorithm

        :param df: DataFrame containing the customer and item file on which to create the matrix
        :return: Sparse matrix for input to ALS

        """

        customers = list(np.sort(df["index_CUST_CODE"].unique()))
        products = list(df["index_PROD_CODE"].unique())
        quantity = list(df["QUANTITY"])

        cols = df["index_CUST_CODE"].astype(
            pd.api.types.CategoricalDtype(categories=customers)
        )
        rows = df["index_PROD_CODE"].astype(
            pd.api.types.CategoricalDtype(categories=products)
        )

        purchases_sparse = sparse.csr_matrix(
            (quantity, (rows, cols)), shape=(len(products), len(customers))
        )

        return purchases_sparse