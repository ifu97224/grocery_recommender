import pytest
import numpy as np
import pandas as pd
from create_features import *

##### Set up the DataFrames required to run the feature engineering tests #####

# Import the test customer csv - tests are being run from a sample of a single customer where all features
# have independently verified as correctly calculated
test_cust = pd.read_csv("./test_feature_engineering.csv")

# Instantiate the class
create_feats = create_features(200716, 200815)

# Run the functions on the test customer DataFrame to get the output for testing
test_filtered_cust_trans = create_feats.filter_cust_trans(
    test_cust, test_cust, test_cust
)

test_hierarchy_map = create_feats.get_product_hierarchy_map(test_filtered_cust_trans)

test_sp_qu_vi = create_feats.get_sp_vi_qu(
    test_filtered_cust_trans,
    test_hierarchy_map,
    [200716, 200742, 200808, 200815],
    ["52", "26", "8", "1"],
    ["PROD_CODE", "PROD_CODE_10", "PROD_CODE_20", "PROD_CODE_30", "PROD_CODE_40"],
)

test_chng_features = create_feats.get_chng_features(
    test_sp_qu_vi,
    ["1", "8", "26", "52"],
    ["PROD_CODE", "PROD_CODE_10", "PROD_CODE_20", "PROD_CODE_30", "PROD_CODE_40"],
)

test_time_since = create_feats.time_last_purchased(
    test_filtered_cust_trans,
    test_hierarchy_map,
    ["PROD_CODE", "PROD_CODE_10", "PROD_CODE_20", "PROD_CODE_30", "PROD_CODE_40"],
)

test_time_since_ratios = create_feats.get_time_since_ratios(
    test_time_since,
    ["PROD_CODE", "PROD_CODE_10", "PROD_CODE_20", "PROD_CODE_30", "PROD_CODE_40"],
)


def create_day_part(df):
    if df["SHOP_HOUR"] >= 8 and df["SHOP_HOUR"] <= 11:
        return "MORNING"
    elif df["SHOP_HOUR"] >= 12 and df["SHOP_HOUR"] <= 16:
        return "AFTERNOON"
    elif df["SHOP_HOUR"] >= 17:
        return "EVENING"


test_filtered_cust_trans.loc[:, "DAY_PART"] = test_filtered_cust_trans.apply(
    create_day_part, axis=1
)


# Create a weekday or weekend segment
def wkday_wkend(df):
    if df["SHOP_WEEKDAY"] == 1 or df["SHOP_WEEKDAY"] == 7:
        return "WEEKEND"
    else:
        return "WEEKDAY"


test_filtered_cust_trans.loc[:, "WKDAY_WKEND"] = test_filtered_cust_trans.apply(
    wkday_wkend, axis=1
)

test_seg_summary_cust = create_feats.create_seg_summary(
    test_filtered_cust_trans,
    [
        "BASKET_PRICE_SENSITIVITY",
        "BASKET_SIZE",
        "DAY_PART",
        "WKDAY_WKEND",
        "BASKET_TYPE",
        "BASKET_DOMINANT_MISSION",
        "STORE_FORMAT",
    ],
    "CUST_CODE",
)

##### End of code to set up the DataFrames needed for the test


##### Define the tests #####

def test_filter_cust_trans():
    """Tests that the customer DataFrame is correctly filtered to the specified weeks"""
    
    min_shop_week = test_filtered_cust_trans["SHOP_WEEK"].min()
    max_shop_week = test_filtered_cust_trans["SHOP_WEEK"].max()

    assert (
        min_shop_week == 200716
    ), "Function is not filtering correctly, min_shop_week != specified week"
    assert (
        max_shop_week == 200815
    ), "Function is not filtering correctly, max_shop_week != specified week"


def test_get_product_hierarchy_map():
    """Tests that the hierarchy DataFrame contains the correct columns and has no duplicates"""
    
    # Test to determine DataFrame contains the required columns
    assert (
        all(
            elem in test_hierarchy_map.columns.to_list()
            for elem in [
                "PROD_CODE",
                "PROD_CODE_10",
                "PROD_CODE_20",
                "PROD_CODE_30",
                "PROD_CODE_40",
            ]
        )
        == True
    ), "Hierarchy map does not contain all expected columns"

    # Test if the DataFrame contains duplicates
    assert (
        np.sum(test_hierarchy_map.duplicated()) == 0
    ), "The DataFrame contains unexpected duplicates"


def test_get_sp_vi_qu():
    """Tests that spend, visits and quantity by hierarchy level features are correctly calculated"""
    
    ### PROD_CODE tests
    test_prod_code = test_sp_qu_vi[test_sp_qu_vi["PROD_CODE"] == "PRD0900135"]

    # Test 52 week variables
    assert (
        np.round(test_prod_code["SPEND_PROD_CODE_52"].values[0], 2) == 21.49
    ), "Error in SPEND_PROD_CODE_52 calculation"
    assert (
        test_prod_code["QUANTITY_PROD_CODE_52"].values[0] == 7
    ), "Error in QUANTITY_PROD_CODE_52 calculation"
    assert (
        test_prod_code["VISITS_PROD_CODE_52"].values[0] == 7
    ), "Error in VISITS_PROD_CODE_52 calculation"

    # Test 26 week variables
    assert (
        np.round(test_prod_code["SPEND_PROD_CODE_26"].values[0], 2) == 12.28
    ), "Error in SPEND_PROD_CODE_26 calculation"
    assert (
        test_prod_code["QUANTITY_PROD_CODE_26"].values[0] == 4
    ), "Error in QUANTITY_PROD_CODE_26 calculation"
    assert (
        test_prod_code["VISITS_PROD_CODE_26"].values[0] == 4
    ), "Error in VISITS_PROD_CODE_26 calculation"

    # Test 8 week variables
    assert (
        np.round(test_prod_code["SPEND_PROD_CODE_8"].values[0], 2) == 6.14
    ), "Error in SPEND_PROD_CODE_8 calculation"
    assert (
        test_prod_code["QUANTITY_PROD_CODE_8"].values[0] == 2
    ), "Error in QUANTITY_PROD_CODE_8 calculation"
    assert (
        test_prod_code["VISITS_PROD_CODE_8"].values[0] == 2
    ), "Error in VISITS_PROD_CODE_8 calculation"

    # Test 1 week variables
    assert (
        np.round(test_prod_code["SPEND_PROD_CODE_1"].values[0], 2) == 3.07
    ), "Error in SPEND_PROD_CODE_1 calculation"
    assert (
        test_prod_code["QUANTITY_PROD_CODE_1"].values[0] == 1
    ), "Error in QUANTITY_PROD_CODE_1 calculation"
    assert (
        test_prod_code["VISITS_PROD_CODE_1"].values[0] == 1
    ), "Error in VISITS_PROD_CODE_1 calculation"

    ### PROD_CODE_10 tests
    test_prod_code_10 = test_sp_qu_vi[test_sp_qu_vi["PROD_CODE_10"] == "CL00201"]

    # Test 52 week variables
    assert (
        np.round(test_prod_code_10["SPEND_PROD_CODE_10_52"].values[0], 2) == 38.76
    ), "Error in SPEND_PROD_CODE_10_52 calculation"
    assert (
        test_prod_code_10["QUANTITY_PROD_CODE_10_52"].values[0] == 17
    ), "Error in QUANTITY_PROD_CODE_10_52 calculation"
    assert (
        test_prod_code_10["VISITS_PROD_CODE_10_52"].values[0] == 12
    ), "Error in VISITS_PROD_CODE_10_52 calculation"

    # Test 26 week variables
    assert (
        np.round(test_prod_code_10["SPEND_PROD_CODE_10_26"].values[0], 2) == 26.99
    ), "Error in SPEND_PROD_CODE_10_26 calculation"
    assert (
        test_prod_code_10["QUANTITY_PROD_CODE_10_26"].values[0] == 12
    ), "Error in QUANTITY_PROD_CODE_10_26 calculation"
    assert (
        test_prod_code_10["VISITS_PROD_CODE_10_26"].values[0] == 7
    ), "Error in VISITS_PROD_CODE_10_26 calculation"

    # Test 8 week variables
    assert (
        np.round(test_prod_code_10["SPEND_PROD_CODE_10_8"].values[0], 2) == 10.78
    ), "Error in SPEND_PROD_CODE_10_8 calculation"
    assert (
        test_prod_code_10["QUANTITY_PROD_CODE_10_8"].values[0] == 5
    ), "Error in QUANTITY_PROD_CODE_10_8 calculation"
    assert (
        test_prod_code_10["VISITS_PROD_CODE_10_8"].values[0] == 3
    ), "Error in VISITS_PROD_CODE_10_8 calculation"

    # Test 1 week variables
    assert (
        np.round(test_prod_code_10["SPEND_PROD_CODE_10_1"].values[0], 2) == 3.93
    ), "Error in SPEND_PROD_CODE_10_1 calculation"
    assert (
        test_prod_code_10["QUANTITY_PROD_CODE_10_1"].values[0] == 2
    ), "Error in QUANTITY_PROD_CODE_10_1 calculation"
    assert (
        test_prod_code_10["VISITS_PROD_CODE_10_1"].values[0] == 1
    ), "Error in VISITS_PROD_CODE_10_1 calculation"

    ### PROD_CODE_30 tests
    test_prod_code_30 = test_sp_qu_vi[test_sp_qu_vi["PROD_CODE_30"] == "G00021"]

    # Test 52 week variables
    assert (
        np.round(test_prod_code_30["SPEND_PROD_CODE_30_52"].values[0], 2) == 71.11
    ), "Error in SPEND_PROD_CODE_30_52 calculation"
    assert (
        test_prod_code_30["QUANTITY_PROD_CODE_30_52"].values[0] == 40
    ), "Error in QUANTITY_PROD_CODE_30_52 calculation"
    assert (
        test_prod_code_30["VISITS_PROD_CODE_30_52"].values[0] == 19
    ), "Error in VISITS_PROD_CODE_30_52 calculation"

    # Test 26 week variables
    assert (
        np.round(test_prod_code_30["SPEND_PROD_CODE_30_26"].values[0], 2) == 51.12
    ), "Error in SPEND_PROD_CODE_30_26 calculation"
    assert (
        test_prod_code_30["QUANTITY_PROD_CODE_30_26"].values[0] == 28
    ), "Error in QUANTITY_PROD_CODE_30_26 calculation"
    assert (
        test_prod_code_30["VISITS_PROD_CODE_30_26"].values[0] == 12
    ), "Error in VISITS_PROD_CODE_30_26 calculation"

    # Test 8 week variables
    assert (
        np.round(test_prod_code_30["SPEND_PROD_CODE_30_8"].values[0], 2) == 21.04
    ), "Error in SPEND_PROD_CODE_30_8 calculation"
    assert (
        test_prod_code_30["QUANTITY_PROD_CODE_30_8"].values[0] == 11
    ), "Error in QUANTITY_PROD_CODE_30_8 calculation"
    assert (
        test_prod_code_30["VISITS_PROD_CODE_30_8"].values[0] == 4
    ), "Error in VISITS_PROD_CODE_30_8 calculation"

    # Test 1 week variables
    assert (
        np.round(test_prod_code_30["SPEND_PROD_CODE_30_1"].values[0], 2) == 3.93
    ), "Error in SPEND_PROD_CODE_30_1 calculation"
    assert (
        test_prod_code_30["QUANTITY_PROD_CODE_30_1"].values[0] == 2
    ), "Error in QUANTITY_PROD_CODE_30_1 calculation"
    assert (
        test_prod_code_30["VISITS_PROD_CODE_30_1"].values[0] == 1
    ), "Error in VISITS_PROD_CODE_30_1 calculation"


def test_get_chng_features():
    """Tests that features for change in spend, visits and quantity are correctly calculated"""
    
    ### Test a subset of the PROD_CODE columns
    test_prod_code = test_chng_features[test_sp_qu_vi["PROD_CODE"] == "PRD0900135"]

    assert (
        np.round(test_prod_code["CHNG_SPEND_PROD_CODE_1_8"].values[0], 2) == 0.5
    ), "Error in CHNG_SPEND_PROD_CODE_1_8 calculation"
    assert (
        np.round(test_prod_code["CHNG_VISITS_PROD_CODE_1_8"].values[0], 2) == 0.5
    ), "Error in CHNG_VISITS_PROD_CODE_1_8 calculation"
    assert (
        np.round(test_prod_code["CHNG_QUANTITY_PROD_CODE_1_8"].values[0], 2) == 0.5
    ), "Error in CHNG_QUANTITY_PROD_CODE_1_8 calculation"

    assert (
        np.round(test_prod_code["CHNG_SPEND_PROD_CODE_8_26"].values[0], 2) == 0.5
    ), "Error in CHNG_SPEND_PROD_CODE_8_26 calculation"
    assert (
        np.round(test_prod_code["CHNG_VISITS_PROD_CODE_8_26"].values[0], 2) == 0.5
    ), "Error in CHNG_VISITS_PROD_CODE_8_26 calculation"
    assert (
        np.round(test_prod_code["CHNG_QUANTITY_PROD_CODE_8_26"].values[0], 2) == 0.5
    ), "Error in CHNG_QUANTITY_PROD_CODE_8_26 calculation"

    assert (
        np.round(test_prod_code["CHNG_SPEND_PROD_CODE_1_52"].values[0], 2) == 0.14
    ), "Error in CHNG_SPEND_PROD_CODE_1_52 calculation"
    assert (
        np.round(test_prod_code["CHNG_VISITS_PROD_CODE_1_52"].values[0], 2) == 0.14
    ), "Error in CHNG_VISITS_PROD_CODE_1_52 calculation"
    assert (
        np.round(test_prod_code["CHNG_QUANTITY_PROD_CODE_1_52"].values[0], 2) == 0.14
    ), "Error in CHNG_QUANTITY_PROD_CODE_1_52 calculation"

    ### Test a subset of the PROD_CODE_10 columns
    test_prod_code_10 = test_chng_features[test_sp_qu_vi["PROD_CODE_10"] == "CL00201"]

    assert (
        np.round(test_prod_code_10["CHNG_SPEND_PROD_CODE_10_1_8"].values[0], 2) == 0.36
    ), "Error in CHNG_SPEND_PROD_CODE_10_1_8 calculation"
    assert (
        np.round(test_prod_code_10["CHNG_VISITS_PROD_CODE_10_1_8"].values[0], 2) == 0.33
    ), "Error in CHNG_VISITS_PROD_CODE_10_1_8 calculation"
    assert (
        np.round(test_prod_code_10["CHNG_QUANTITY_PROD_CODE_10_1_8"].values[0], 2)
        == 0.4
    ), "Error in CHNG_QUANTITY_PROD_CODE_10_1_8 calculation"

    assert (
        np.round(test_prod_code_10["CHNG_SPEND_PROD_CODE_10_8_26"].values[0], 2) == 0.4
    ), "Error in CHNG_SPEND_PROD_CODE_10_8_26 calculation"
    assert (
        np.round(test_prod_code_10["CHNG_VISITS_PROD_CODE_10_8_26"].values[0], 2)
        == 0.43
    ), "Error in CHNG_VISITS_PROD_CODE_10_8_26 calculation"
    assert (
        np.round(test_prod_code_10["CHNG_QUANTITY_PROD_CODE_10_8_26"].values[0], 2)
        == 0.42
    ), "Error in CHNG_QUANTITY_PROD_CODE_10_8_26 calculation"

    assert (
        np.round(test_prod_code_10["CHNG_SPEND_PROD_CODE_10_1_52"].values[0], 2) == 0.1
    ), "Error in CHNG_SPEND_PROD_CODE_10_1_52 calculation"
    assert (
        np.round(test_prod_code_10["CHNG_VISITS_PROD_CODE_10_1_52"].values[0], 2)
        == 0.08
    ), "Error in CHNG_VISITS_PROD_CODE_10_1_52 calculation"
    assert (
        np.round(test_prod_code_10["CHNG_QUANTITY_PROD_CODE_10_1_52"].values[0], 2)
        == 0.12
    ), "Error in CHNG_QUANTITY_PROD_CODE_10_1_52 calculation"

    ### Test a subset of the PROD_CODE_30 columns
    test_prod_code_30 = test_chng_features[test_sp_qu_vi["PROD_CODE_30"] == "G00021"]

    assert (
        np.round(test_prod_code_30["CHNG_SPEND_PROD_CODE_30_1_8"].values[0], 2) == 0.19
    ), "Error in CHNG_SPEND_PROD_CODE_30_1_8 calculation"
    assert (
        np.round(test_prod_code_30["CHNG_VISITS_PROD_CODE_30_1_8"].values[0], 2) == 0.25
    ), "Error in CHNG_VISITS_PROD_CODE_30_1_8 calculation"
    assert (
        np.round(test_prod_code_30["CHNG_QUANTITY_PROD_CODE_30_1_8"].values[0], 2)
        == 0.18
    ), "Error in CHNG_QUANTITY_PROD_CODE_30_1_8 calculation"

    assert (
        np.round(test_prod_code_30["CHNG_SPEND_PROD_CODE_30_8_26"].values[0], 2) == 0.41
    ), "Error in CHNG_SPEND_PROD_CODE_30_8_26 calculation"
    assert (
        np.round(test_prod_code_30["CHNG_VISITS_PROD_CODE_30_8_26"].values[0], 2)
        == 0.33
    ), "Error in CHNG_VISITS_PROD_CODE_30_8_26 calculation"
    assert (
        np.round(test_prod_code_30["CHNG_QUANTITY_PROD_CODE_30_8_26"].values[0], 2)
        == 0.39
    ), "Error in CHNG_QUANTITY_PROD_CODE_30_8_26 calculation"

    assert (
        np.round(test_prod_code_30["CHNG_SPEND_PROD_CODE_30_1_52"].values[0], 2) == 0.06
    ), "Error in CHNG_SPEND_PROD_CODE_30_1_52 calculation"
    assert (
        np.round(test_prod_code_30["CHNG_VISITS_PROD_CODE_30_1_52"].values[0], 2)
        == 0.05
    ), "Error in CHNG_VISITS_PROD_CODE_30_1_52 calculation"
    assert (
        np.round(test_prod_code_30["CHNG_QUANTITY_PROD_CODE_30_1_52"].values[0], 2)
        == 0.05
    ), "Error in CHNG_QUANTITY_PROD_CODE_30_1_52 calculation"


def test_time_last_purchased():
    """Tests that features for time since last purchased are correctly calculated"""
    
    test_df = test_time_since[test_time_since["PROD_CODE"] == "PRD0900135"]

    # Test PROD_CODE columns
    assert (
        test_df["TIME_BTWN_MEDIAN_CUST_PROD_CODE"].values[0] == 46
    ), "Error in TIME_BETWN_MEDIAN_CUST_PROD_CODE calculation"
    assert (
        test_df["TIME_BTWN_LAST_PROD_CODE"].values[0] == 36
    ), "Error in TIME_BTWN_LAST_PROD_CODE calculation"

    # Test PROD_CODE_10 columns
    assert (
        test_df["TIME_BTWN_MEDIAN_CUST_PROD_CODE_10"].values[0] == 22
    ), "Error in TIME_BETWN_MEDIAN_CUST_PROD_CODE_10 calculation"
    assert (
        test_df["TIME_BTWN_LAST_PROD_CODE_10"].values[0] == 16
    ), "Error in TIME_BTWN_LAST_PROD_CODE_10 calculation"

    # Test PROD_CODE_30 columns
    assert (
        test_df["TIME_BTWN_MEDIAN_CUST_PROD_CODE_30"].values[0] == 15.5
    ), "Error in TIME_BETWN_MEDIAN_CUST_PROD_CODE_30 calculation"
    assert (
        test_df["TIME_BTWN_LAST_PROD_CODE_30"].values[0] == 16
    ), "Error in TIME_BTWN_LAST_PROD_CODE_30 calculation"

def test_get_time_since_ratios():
    """Tests that features for the ratios of time since last purchased to median time between purchases
    are correctly calculated"""
    
    test_df = test_time_since_ratios[test_time_since_ratios['PROD_CODE'] == 'PRD0900135']

    # Test PROD_CODE columns
    assert np.round(test_df['TIME_BTWN_RATIO_CUST_PROD_CODE'].values[0],
                    2) == 0.78, "Error in TIME_BTWN_RATIO_CUST_PROD_CODE calculation"

    # Test PROD_CODE_10 columns
    assert np.round(test_df['TIME_BTWN_RATIO_CUST_PROD_CODE_10'].values[0],
                    2) == 0.73, "Error in TIME_BTWN_RATIO_CUST_PROD_CODE_10 calculation"

    # Test PROD_CODE_30 columns
    assert np.round(test_df['TIME_BTWN_RATIO_CUST_PROD_CODE_30'].values[0],
                    2) == 1.03, "Error in TIME_BTWN_RATIO_CUST_PROD_CODE_30 calculation"
    
def test_create_seg_summary():
    """Tests that features for spend, visits and quantity by each customer and basket segment are correctly calculated"""
    
    test_df = test_seg_summary_cust

    # Test columns from one level of BASKET_PRICE_SENSITIVITY
    assert (
        np.round(test_df["BASKET_PRICE_SENSITIVITY_SPEND_CUST_CODE_LA"].values[0], 2)
        == 2.27
    ), "Error in BASKET_PRICE_SENSITIVITY_SPEND_CUST_CODE_LA calculation"
    assert (
        test_df["BASKET_PRICE_SENSITIVITY_QUANTITY_CUST_CODE_LA"].values[0] == 2
    ), "Error in BASKET_PRICE_SENSITIVITY_QUANTITY_CUST_CODE_LA calculation"
    assert (
        test_df["BASKET_PRICE_SENSITIVITY_VISITS_CUST_CODE_LA"].values[0] == 1
    ), "Error in BASKET_PRICE_SENSITIVITY_VISITS_CUST_CODE_LA calculation"

    assert (
        np.round(
            test_df["BASKET_PRICE_SENSITIVITY_PROP_SPEND_CUST_CODE_LA"].values[0], 3
        )
        == 0.002
    ), "Error in BASKET_PRICE_SENSITIVITY_PROP_SPEND_CUST_CODE_LA calculation"
    assert (
        np.round(
            test_df["BASKET_PRICE_SENSITIVITY_PROP_QUANTITY_CUST_CODE_LA"].values[0], 3
        )
        == 0.003
    ), "Error in BASKET_PRICE_SENSITIVITY_PROP_QUANTITY_CUST_CODE_LA calculation"
    assert (
        np.round(
            test_df["BASKET_PRICE_SENSITIVITY_PROP_VISITS_CUST_CODE_LA"].values[0], 3
        )
        == 0.014
    ), "Error in BASKET_PRICE_SENSITIVITY_PROP_VISITS_CUST_CODE_LA calculation"

    # Test columns from one level of BASKET_SIZE
    assert (
        np.round(test_df["BASKET_SIZE_SPEND_CUST_CODE_L"].values[0], 2) == 704.42
    ), "Error in BASKET_SIZE_SPEND_CUST_CODE_L calculation"
    assert (
        test_df["BASKET_SIZE_QUANTITY_CUST_CODE_L"].values[0] == 520
    ), "Error in BASKET_SIZE_QUANTITY_CUST_CODE_L calculation"
    assert (
        test_df["BASKET_SIZE_VISITS_CUST_CODE_L"].values[0] == 24
    ), "Error in BASKET_SIZE_VISITS_CUST_CODE_L calculation"

    assert (
        np.round(test_df["BASKET_SIZE_PROP_SPEND_CUST_CODE_L"].values[0], 3) == 0.706
    ), "Error in BASKET_SIZE_PROP_SPEND_CUST_CODE_L calculation"
    assert (
        np.round(test_df["BASKET_SIZE_PROP_QUANTITY_CUST_CODE_L"].values[0], 3) == 0.711
    ), "Error in BASKET_SIZE_PROP_QUANTITY_CUST_CODE_L calculation"
    assert (
        np.round(test_df["BASKET_SIZE_PROP_VISITS_CUST_CODE_L"].values[0], 3) == 0.343
    ), "Error in BASKET_SIZE_PROP_VISITS_CUST_CODE_L calculation"

    # Test columns from one level of DAY_PART
    assert (
        np.round(test_df["DAY_PART_SPEND_CUST_CODE_AFTERNOON"].values[0], 2) == 481.84
    ), "Error in DAY_PART_SPEND_CUST_CODE_AFTERNOON calculation"
    assert (
        test_df["DAY_PART_QUANTITY_CUST_CODE_AFTERNOON"].values[0] == 381
    ), "Error in DAY_PART_QUANTITY_CUST_CODE_AFTERNOON calculation"
    assert (
        test_df["DAY_PART_VISITS_CUST_CODE_AFTERNOON"].values[0] == 38
    ), "Error in DAY_PART_VISITS_CUST_CODE_AFTERNOON calculation"

    assert (
        np.round(test_df["DAY_PART_PROP_SPEND_CUST_CODE_AFTERNOON"].values[0], 3)
        == 0.483
    ), "Error in DAY_PART_PROP_SPEND_CUST_CODE_AFTERNOON calculation"
    assert (
        np.round(test_df["DAY_PART_PROP_QUANTITY_CUST_CODE_AFTERNOON"].values[0], 3)
        == 0.521
    ), "Error in DAY_PART_PROP_QUANTITY_CUST_CODE_AFTERNOON calculation"
    assert (
        np.round(test_df["DAY_PART_PROP_VISITS_CUST_CODE_AFTERNOON"].values[0], 3)
        == 0.543
    ), "Error in DAY_PART_PROP_VISITS_CUST_CODE_AFTERNOON calculation"

    # Test columns from one level of WKDAY_WKEND
    assert (
        np.round(test_df["WKDAY_WKEND_SPEND_CUST_CODE_WEEKEND"].values[0], 2) == 266.72
    ), "Error in DAY_PART_SPEND_CUST_CODE_AFTERNOON calculation"
    assert (
        test_df["WKDAY_WKEND_QUANTITY_CUST_CODE_WEEKEND"].values[0] == 178
    ), "Error in DAY_PART_QUANTITY_CUST_CODE_AFTERNOON calculation"
    assert (
        test_df["WKDAY_WKEND_VISITS_CUST_CODE_WEEKEND"].values[0] == 17
    ), "Error in DAY_PART_VISITS_CUST_CODE_AFTERNOON calculation"

    assert (
        np.round(test_df["WKDAY_WKEND_PROP_SPEND_CUST_CODE_WEEKEND"].values[0], 3)
        == 0.267
    ), "Error in WKDAY_WKEND_PROP_SPEND_CUST_CODE_WEEKEND calculation"
    assert (
        np.round(test_df["WKDAY_WKEND_PROP_QUANTITY_CUST_CODE_WEEKEND"].values[0], 3)
        == 0.244
    ), "Error in WKDAY_WKEND_PROP_QUANTITY_CUST_CODE_WEEKEND calculation"
    assert (
        np.round(test_df["WKDAY_WKEND_PROP_VISITS_CUST_CODE_WEEKEND"].values[0], 3)
        == 0.243
    ), "Error in WKDAY_WKEND_PROP_VISITS_CUST_CODE_WEEKEND calculation"

    # Test columns from one level of BASKET_TYPE
    assert (
        np.round(test_df["BASKET_TYPE_SPEND_CUST_CODE_Top Up"].values[0], 2) == 494.61
    ), "Error in BASKET_TYPE_SPEND_CUST_CODE_Top Up calculation"
    assert (
        test_df["BASKET_TYPE_QUANTITY_CUST_CODE_Top Up"].values[0] == 349
    ), "Error in BASKET_TYPE_QUANTITY_CUST_CODE_Top Up calculation"
    assert (
        test_df["BASKET_TYPE_VISITS_CUST_CODE_Top Up"].values[0] == 24
    ), "Error in BASKET_TYPE_VISITS_CUST_CODE_Top Up calculation"

    assert (
        np.round(test_df["BASKET_TYPE_PROP_SPEND_CUST_CODE_Top Up"].values[0], 3)
        == 0.496
    ), "Error in BASKET_TYPE_PROP_SPEND_CUST_CODE_Top Up calculation"
    assert (
        np.round(test_df["BASKET_TYPE_PROP_QUANTITY_CUST_CODE_Top Up"].values[0], 3)
        == 0.477
    ), "Error in BASKET_TYPE_PROP_QUANTITY_CUST_CODE_Top Up calculation"
    assert (
        np.round(test_df["BASKET_TYPE_PROP_VISITS_CUST_CODE_Top Up"].values[0], 3)
        == 0.343
    ), "Error in BASKET_TYPE_PROP_VISITS_CUST_CODE_Top Up calculation"

    # Test columns from one level of BASKET_TYPE
    assert (
        np.round(test_df["BASKET_DOMINANT_MISSION_SPEND_CUST_CODE_Mixed"].values[0], 2)
        == 525.85
    ), "Error in BASKET_DOMINANT_MISSION_SPEND_CUST_CODE_Mixed Up calculation"
    assert (
        test_df["BASKET_DOMINANT_MISSION_QUANTITY_CUST_CODE_Mixed"].values[0] == 373
    ), "Error in BASKET_DOMINANT_MISSION_QUANTITY_CUST_CODE_Mixed calculation"
    assert (
        test_df["BASKET_DOMINANT_MISSION_VISITS_CUST_CODE_Mixed"].values[0] == 22
    ), "Error in BASKET_DOMINANT_MISSION_VISITS_CUST_CODE_Mixed calculation"

    assert (
        np.round(
            test_df["BASKET_DOMINANT_MISSION_PROP_SPEND_CUST_CODE_Mixed"].values[0], 3
        )
        == 0.527
    ), "Error in BASKET_DOMINANT_MISSION_PROP_SPEND_CUST_CODE_Mixed calculation"
    assert (
        np.round(
            test_df["BASKET_DOMINANT_MISSION_PROP_QUANTITY_CUST_CODE_Mixed"].values[0],
            3,
        )
        == 0.510
    ), "Error in BASKET_DOMINANT_MISSION_PROP_QUANTITY_CUST_CODE_Mixed calculation"
    assert (
        np.round(
            test_df["BASKET_DOMINANT_MISSION_PROP_VISITS_CUST_CODE_Mixed"].values[0], 3
        )
        == 0.314
    ), "Error in BASKET_DOMINANT_MISSION_PROP_VISITS_CUST_CODE_Mixed calculation"

    # Test columns from one level of STORE_FORMAT
    assert (
        np.round(test_df["STORE_FORMAT_SPEND_CUST_CODE_LS"].values[0], 2) == 944.92
    ), "Error in BASKET_DOMINANT_MISSION_SPEND_CUST_CODE_Mixed Up calculation"
    assert (
        test_df["STORE_FORMAT_QUANTITY_CUST_CODE_LS"].values[0] == 698
    ), "Error in BASKET_DOMINANT_MISSION_QUANTITY_CUST_CODE_Mixed calculation"
    assert (
        test_df["STORE_FORMAT_VISITS_CUST_CODE_LS"].values[0] == 62
    ), "Error in BASKET_DOMINANT_MISSION_VISITS_CUST_CODE_Mixed calculation"

    assert (
        np.round(test_df["STORE_FORMAT_PROP_SPEND_CUST_CODE_LS"].values[0], 3) == 0.947
    ), "Error in STORE_FORMAT_PROP_SPEND_CUST_CODE_LS calculation"
    assert (
        np.round(test_df["STORE_FORMAT_PROP_QUANTITY_CUST_CODE_LS"].values[0], 3)
        == 0.955
    ), "Error in STORE_FORMAT_PROP_QUANTITY_CUST_CODE_LS calculation"
    assert (
        np.round(test_df["STORE_FORMAT_PROP_VISITS_CUST_CODE_LS"].values[0], 3) == 0.886
    ), "Error in STORE_FORMAT_PROP_VISITS_CUST_CODE_LS calculation"

##### End of code to define the tests


##### Run the tests #####
test_filter_cust_trans()
test_get_product_hierarchy_map()
test_get_sp_vi_qu()
test_get_chng_features()
test_time_last_purchased()
test_get_time_since_ratios()
test_time_last_purchased()
test_create_seg_summary
