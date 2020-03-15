import pytest
import numpy as np
import pandas as pd
from create_features import *

# Import the test customer csv
test_cust = pd.read_csv('./test_feature_engineering.csv')

create_feats = create_features(200716, 200815)

def test_filter_cust_trans():
    
    test_df = create_feats.filter_cust_trans(df = test_cust, train_df = test_cust, test_df = test_cust)
    
    min_shop_week = test_df['SHOP_WEEK'].min()
    max_shop_week = test_df['SHOP_WEEK'].max()
    
    assert min_shop_week == 200716, "Function is not filtering correctly, min_shop_week != specified week"
    assert max_shop_week == 200815, "Function is not filtering correctly, max_shop_week != specified week"
    
def test_get_product_hierarchy_map():
    
    test_df = create_feats.get_product_hierarchy_map(test_cust)
    
    # Test to determine DataFrame contains the required columns
    assert all(elem in test_df.columns.to_list() for elem in \
               ['PROD_CODE','PROD_CODE_10','PROD_CODE_20','PROD_CODE_30','PROD_CODE_40']) == True, \
    "Hierarchy map does not contain all expected columns"
    
    # Test if the DataFrame contains duplicates
    assert np.sum(test_df.duplicated()) == 0, "The DataFrame contains unexpected duplicates"
    

def test_get_sp_vi_qu():
    
    test_hierarchy_map = create_feats.get_product_hierarchy_map(test_cust)
    test_sp_qu_vi = create_feats.get_sp_vi_qu(test_cust, 
                                              test_hierarchy_map,
                                              [200716, 200742, 200808, 200815],
                                              ["52", "26", "8", "1"],
                                              ["PROD_CODE", "PROD_CODE_10", "PROD_CODE_20", 
                                               "PROD_CODE_30", "PROD_CODE_40"])

    test_sp_qu_vi = test_sp_qu_vi[test_sp_qu_vi['PROD_CODE'] == 'PRD0900135']
    
    # Test 52 week variables
    assert test_sp_qu_vi['SPEND_PROD_CODE_52'].values[0] == 21.49, "Error in SPEND_PROD_CODE_52 calculation"
    assert test_sp_qu_vi['QUANTITY_PROD_CODE_52'].values[0] == 7, "Error in QUANTITY_PROD_CODE_52 calculation"
    assert test_sp_qu_vi['VISITS_PROD_CODE_52'].values[0] == 7, "Error in VISITS_PROD_CODE_52 calculation"
    
    # Test 26 week variables
    assert test_sp_qu_vi['SPEND_PROD_CODE_26'].values[0] == 12.28, "Error in SPEND_PROD_CODE_26 calculation"
    assert test_sp_qu_vi['QUANTITY_PROD_CODE_26'].values[0] == 4, "Error in QUANTITY_PROD_CODE_26 calculation"
    assert test_sp_qu_vi['VISITS_PROD_CODE_26'].values[0] == 4, "Error in VISITS_PROD_CODE_26 calculation"
    
    # Test 8 week variables
    assert test_sp_qu_vi['SPEND_PROD_CODE_8'].values[0] == 6.14, "Error in SPEND_PROD_CODE_8 calculation"
    assert test_sp_qu_vi['QUANTITY_PROD_CODE_8'].values[0] == 2, "Error in QUANTITY_PROD_CODE_8 calculation"
    assert test_sp_qu_vi['VISITS_PROD_CODE_8'].values[0] == 2, "Error in VISITS_PROD_CODE_8 calculation"
    
    # Test 1 week variables
    assert test_sp_qu_vi['SPEND_PROD_CODE_1'].values[0] == 3.07, "Error in SPEND_PROD_CODE_1 calculation"
    assert test_sp_qu_vi['QUANTITY_PROD_CODE_1'].values[0] == 1, "Error in QUANTITY_PROD_CODE_1 calculation"
    assert test_sp_qu_vi['VISITS_PROD_CODE_1'].values[0] == 1, "Error in VISITS_PROD_CODE_1 calculation"
    
    

    
test_filter_cust_trans()
test_get_product_hierarchy_map()
test_get_sp_vi_qu()
