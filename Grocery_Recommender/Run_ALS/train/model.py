import implicit
import os
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from hyperopt import STATUS_OK, hp, Trials, fmin, tpe
from timeit import default_timer as timer
import csv


class fit_and_tune_als:
    """ The fit_and_tune_als class contains all functions necessary to fit and tune ALS models using the Implicit and
        Hyperopt libraries"""

    def __init__(self, sparse_matrix):
        """ Method for initializing the fit_and_tune_als object

        Args:
        sparse_matrix (a sparse matrix)

        Attributes:
            sparse_matrix (sparse matrix):  The customer / item sparse matrix

        """
        self.sparse_matrix = sparse_matrix

    def fit_als(self, alpha, num_factors, reg, iters):
        """Function to fit the ALS algorithm based on the provided hyperparameters

        Parameters
        ----------
        alpha : int
            ALS alpha parameter
        num_factors : int
            Number of user and item factors to use
        reg : int
            Regularization parameter for the ALS
        iters : int
            Number of iterations to run the ALS

        Returns
        -------
        als_model : implicit.als.AlternatingLeastSquares object
           Fitted ALS model
        user_vecs : pandas.DataFrame
            DataFrame containing the user vectors for each customer
        item_vecs : pandas.DataFrame
            DataFrame containing the item vectors for each item
        """

        os.environ["MKL_NUM_THREADS"] = "1"

        als_model = implicit.als.AlternatingLeastSquares(
            factors=num_factors,
            regularization=reg,
            iterations=iters,
            use_gpu=False
        )

        als_model.fit((self.sparse_matrix * alpha).astype("double"))

        # Get the user and item vectors from the fitted model
        user_vecs = als_model.user_factors
        item_vecs = als_model.item_factors

        return als_model, user_vecs, item_vecs

    def get_mse(self, user_vecs, item_vecs, train_df, test_df):
        """Function to get the MSE from the fitted model on the train and test sets

        Parameters
        ----------
        user_vecs : pandas.DataFrame
            DataFrame containing the user vectors from the fitted model
        item_vecs : pandas.DataFrame
            DataFrame containing the item vectors from the fitted model
        train_df : pandas.DataFrame
             DataFrame containing the training data
        test_df : pandas.DataFrame
            DataFrame containing the test data

        Returns
        -------
        training_mse : float
            The training MSE
        test_mse : float
            The test MSE

        """

        # Get the predicted DataFrame from the dot product of the user and item vectors
        pred_df = pd.DataFrame(user_vecs.dot(item_vecs.T))

        # Get the column headers(product indices)
        products = list(train_df["index_PROD_CODE"].unique())

        # Replace the column indices with the product index
        pred_df.columns = products

        # Create a column for the index_CUST_CODE - this is the index of the DataFrame
        pred_df.loc[:, "index_CUST_CODE"] = np.sort(train_df["index_CUST_CODE"].unique())

        # Melt the DataFrame
        pred_df_melt = pd.melt(pred_df, id_vars="index_CUST_CODE", value_vars=products)
        pred_df_melt = pred_df_melt.rename(
            columns={"variable": "index_PROD_CODE", "value": "PRED_QUANTITY"}
        )

        # Merge the DataFrame back to the original DataFrame
        train_pred_actual = train_df.merge(
            pred_df_melt, on=["index_CUST_CODE", "index_PROD_CODE"], how="inner"
        )

        # Get the training set MSE
        y_true = train_pred_actual["QUANTITY"]
        y_pred = train_pred_actual["PRED_QUANTITY"]
        training_mse = mean_squared_error(y_true, y_pred)

        # Get the test set MSE
        test_pred_actual = test_df.merge(
            pred_df_melt, on=["index_CUST_CODE", "index_PROD_CODE"], how="inner"
        )
        y_true = test_pred_actual["QUANTITY"]
        y_pred = test_pred_actual["PRED_QUANTITY"]
        test_mse = mean_squared_error(y_true, y_pred)

        print("training set mse = {}, test set mse = {}".format(training_mse, test_mse))

        return training_mse, test_mse

    def tune_params(self, search_space, out_file, max_evals, train_df, test_df):
        """Function to tune the ALS model parameters

        Parameters
        ----------
        search_space : dict
            Dictionary containing the hyperparameters and values to search over
        out_file : str
            File path to hold output data
        max_evals : int
             The maximum number of trials for the hyperparamater search
        train_df : pandas.DataFrame
            DataFrame containing the training data
        test_df : pandas.DataFrame
            DataFrame containing the test data

        Returns
        ----------
        best : hyperopt object
            hyperopt object containing the best hyperparamters

        """

        train_df = train_df
        test_df = test_df

        def objective(
                params, train_df=train_df, test_df=test_df
        ):
            """
            Returns
            ----------
            Returns validation score from hyperparameters"""

            print(params)

            # Keep track of evals
            global ITERATION

            ITERATION += 1

            start = timer()
            als_model, user_vecs, item_vecs = self.fit_als(
                alpha=params["alpha"],
                num_factors=params["num_factors"],
                reg=params["reg"],
                iters=params["iters"],
            )

            training_set_mse, test_set_mse = self.get_mse(
                user_vecs, item_vecs, train_df, test_df
            )
            loss = test_set_mse

            run_time = timer() - start

            # Write to the csv file ('a' means append)
            of_connection = open(out_file, "a")
            writer = csv.writer(of_connection)
            writer.writerow([loss, params, ITERATION, run_time, STATUS_OK])

            return {
                "loss": loss,
                "params": params,
                "iteration": ITERATION,
                "train_time": run_time,
                "status": STATUS_OK,
            }

        # Trials object to track progress
        bayes_trials = Trials()

        ##### CLEAR THE RESULTS FILE ######

        # File to save first results
        out_file = out_file
        of_connection = open(out_file, "w")
        writer = csv.writer(of_connection)

        # Write the headers to the file
        writer.writerow(["loss", "params", "iteration", "train_time", "status"])
        of_connection.close()

        MAX_EVALS = max_evals

        # Global variable
        global ITERATION

        ITERATION = 0

        # Optimize
        best = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=MAX_EVALS,
            trials=bayes_trials,
        )

        return best

    def get_user_item_factors(
            self, user_vecs, item_vecs, train_df, train_items_mapping, train_cust_mapping
    ):
        """Function to get user and item factors for the fitted ALS model

        Parameters
        ----------
        user_vecs : pandas.DataFrame
            DataFrame containing the user vectors from the fitted ALS model
        item_vecs : pandas.DataFrame
            DataFrame containing the item vectors from the fitted ALS model
        train_df : pandas.DataFrame
            DataFrame containing the training data
        train_items_mapping : pandas.DataFrame
            DataFrame containing the mapping from index_PROD_CODE to PROD_CODE
        train_cust_mapping : pandas.DataFrame
            DataFrame containing the mapping from index_CUST_CODE to CUST_CODE

        Returns
        ----------
        user_vecs : pandas.DataFrame
            DataFrame containing the mapped user vectors from the fitted model
        item_vecs : pandas.DataFrame
            DataFrame containing the mapped item vectors from the fitted model

        """
        
        # Get the user factors
        user_factors = pd.DataFrame(user_vecs)
        user_factors = user_factors.add_prefix("factor_")

        user_factors.loc[:, "index_CUST_CODE"] = np.sort(
            train_df["index_CUST_CODE"].unique()
        )

        user_factors = user_factors.merge(
            train_cust_mapping, on=["index_CUST_CODE"], how="inner"
        )
        user_factors.drop("index_CUST_CODE", axis=1, inplace=True)

        # Get the item factors
        item_factors = pd.DataFrame(item_vecs)
        item_factors = item_factors.add_prefix("factor_")

        item_factors.loc[:, "index_PROD_CODE"] = train_df[
            "index_PROD_CODE"
        ].unique()

        item_factors = item_factors.merge(
            train_items_mapping, on=["index_PROD_CODE"], how="inner"
        )
        item_factors.drop("index_PROD_CODE", axis=1, inplace=True)

        return user_factors, item_factors

    def predict(self, user_vecs, item_vecs, mapping_df, top_x):
        """Function to get predictions from the fitted ALS model

        Parameters
        ----------
        user_vecs : pandas.DataFrame
            DataFrame containing the user vectors from the fitted ALS model
        item_vecs : pandas.DataFrame
            DataFrame containing the item vectors from the fitted ALS model
        mapping_df : pandas.DataFrame
            DataFrame containing the mapping between the customer and product indices and the actual CUST_CODE and
            PROD_CODE
        top_x : int
            Number of predictions to return per customer
            
        Returns
        ----------
        pred_df : pandas.DataFrame
            DataFrame containing the CUsT_CODE and prediction for each PROD_CODE
            
        """
                
        # Get the predicted DataFrame from the dot product of the user and item vectors
        pred_df = pd.DataFrame(user_vecs.dot(item_vecs.T))

        # Get the column headers(product indices)
        products = list(mapping_df["index_PROD_CODE"].unique())

        # Replace the column indices with the product index
        pred_df.columns = products

        # Create a column for the index_CUST_CODE - this is the index of the DataFrame
        pred_df.loc[:, "index_CUST_CODE"] = np.sort(mapping_df["index_CUST_CODE"].unique())

        # Melt the DataFrame
        pred_df = pd.melt(pred_df, id_vars="index_CUST_CODE", value_vars=products)
        pred_df = pred_df.rename(
            columns={"variable": "index_PROD_CODE", "value": "PRED_QUANTITY"}
        )

        # Create rank of the predicted quantity by customer
        pred_df.loc[:, "pred_rank"] = pred_df.groupby("index_CUST_CODE")[
            "PRED_QUANTITY"
        ].rank("dense", ascending=False)

        # Keep only the top x products
        pred_df = pred_df.loc[pred_df["pred_rank"] <= top_x]

        # Map the CUST_CODE and PROD_CODE
        cust_map = mapping_df[["CUST_CODE", "index_CUST_CODE"]].drop_duplicates()
        pred_df = pred_df.merge(cust_map, on="index_CUST_CODE")
        pred_df.drop("index_CUST_CODE", axis=1, inplace=True)

        prod_map = mapping_df[["PROD_CODE", "index_PROD_CODE"]].drop_duplicates()
        pred_df = pred_df.merge(prod_map, on="index_PROD_CODE")
        pred_df.drop("index_PROD_CODE", axis=1, inplace=True)

        return pred_df