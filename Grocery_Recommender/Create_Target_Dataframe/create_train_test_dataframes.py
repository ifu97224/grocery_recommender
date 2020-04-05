import pandas as pd

def get_targets(df, target_wk):
    """Function to create a DataFrame of customers and products purchased during the target week

    Parameters
    ----------
    df : pandas.DataFrame
        name of the customer transaction file
    target_wk : int
        the target week on which the models will be built

    Returns
    -------
    target_custs : pandas.DataFrame
        DataFrame containing CUST_CODE, PROD_CODE and Target flag

    """
    
    target_custs = df[df["SHOP_WEEK"] == target_wk]
    target_custs = target_custs[["CUST_CODE", "PROD_CODE"]].drop_duplicates()
    target_custs.dropna(inplace=True)
    target_custs.loc[:, "TARGET"] = 1

    return target_custs


def get_active_custs(df, start_wk, end_wk):
    """Function to return a set of customers that visited between the provided start and end weeks

        Parameters
        ----------
        df : pandas.DataFrame
            name of the customer transaction file
        start_wk : int
            the start week on which to look for customer visits
        end_wk : int
            the end week on which to look for customer visits

        Returns
        -------
        active_custs : pandas.DataFrame
            DataFrame containing CUST_CODE for all customers that visited between the start and end weeks

    """
    
    active_custs = df[(df["SHOP_WEEK"] >= start_wk) & (df["SHOP_WEEK"] <= end_wk)]
    active_custs = active_custs[["CUST_CODE"]].drop_duplicates()
    active_custs.dropna(inplace=True)

    return active_custs


def add_non_target(active_df, target_df):
    """Function to add all non-target customers (active customers that did not purchase the item in the observation 
    period)

    Parameters
    ----------
    active_df : pandas.DataFrame
        name of the DataFrame containing all active customers
    target_df : pandas.DataFrame
        name of the DataFrame containing the target customers and products
    
    Returns
    -------
    active_target : pandas.DataFrame
        DataFrame containing CUST_CODE, PROD_CODE and TARGET for all customer and product combinations

    """
    
    # Get all unique products purchased in the observation period that will be modelled
    unique_prods = target_df[["PROD_CODE"]].drop_duplicates()

    # Now merge all unique products to the active customers
    active_df.loc[:, "tmp"] = 1
    unique_prods.loc[:, "tmp"] = 1
    active_df = active_df.merge(unique_prods, on="tmp")
    active_df.drop("tmp", axis=1, inplace=True)

    # Merge to the target DataFrame and set all missing TARGET variables to 0
    active_target = active_df.merge(
        target_df, on=["CUST_CODE", "PROD_CODE"], how="left"
    )
    active_target.fillna(0, inplace=True)

    return active_target


def sample_non_target(df):
    """Function to downsample non-target customers and create a balanced DataFrame with equal numbers of TARGET 1 and TARGET 0
    for each PROD_CODE

    Parameters
    ----------
    df : pandas.DataFrame
        name of the DataFrame containing all active customers and the target flag for each product

    Returns
    -------
    final_sample : pandas.DataFrame
        DataFrame containing CUST_CODE, PROD_CODE and TARGET for all TARGET 1 and down-sampled TARGET 0

    """
    
    # Get the count of target customers
    target_count = df[df["TARGET"] == 1].count()[0]

    # Get ALL non-target customers into a DataFrame
    all_non_target = df[df["TARGET"] == 0]

    # Downsample the same number of non-target customers as target customers
    non_target_sample = all_non_target.sample(
        n=target_count, replace=False, random_state=0
    )

    # Concatenate the targets and the downsampled non-targets
    final_sample = pd.concat([df[df["TARGET"] == 1], non_target_sample], axis=0)
    final_sample = final_sample.sort_values(["CUST_CODE", "PROD_CODE"])

    return final_sample