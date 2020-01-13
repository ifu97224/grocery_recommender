import pandas as pd
import numpy as np

class create_features:
    """ Create features for classification model"""

    def __init__(self, obs_st, obs_end):
        """ Method for initializing the create_features object

        Args:
        obs_st
        obs_end

        Attributes:
            obs_st:  The start week for the observation period on which features are created
            obs_end: The end week for the observation period on which features are created

        """
        self.obs_st = obs_st
        self.obs_end = obs_end

    def filter_cust_trans(self, df, train_df, test_df):
        """
        Function to filter the customer transactions file to only those customer and product combinations that are in the
        training or test sets and to the observation period

        :param df: name of the unfiltered customer transaction file
        :param train_df: name of the DataFrame containing the customers and products for training
        :param test_df: name of the DataFrame containing the customers and products for testing
            :param obs_st:  The start week of the observation period
        :param obs_end:  The end week of the observations period
        :return: Filtered customer transactions DataFrame

        """

        train_df = train_df[["CUST_CODE", "PROD_CODE"]]
        test_df = test_df[["CUST_CODE", "PROD_CODE"]]

        train_test_cust_prods = pd.concat([train_df, test_df], axis=0)
        train_test_cust_prods.drop_duplicates(inplace=True)

        df = df.merge(train_test_cust_prods, on=["CUST_CODE", "PROD_CODE"])

        # Filter to the observation period
        df = df.loc[(df["SHOP_WEEK"] >= self.obs_st) & (df["SHOP_WEEK"] <= self.obs_end)]

        return df


    def get_product_hierarchy_map(self, df):
        """
        Function to create a mapping file between PROD_CODE and all levels of the product hierarchy

        :param df: name of the customer transaction file
        :return: Hierarchy mapping DataFrame

        """

        hierarchy_map = df[
            ["PROD_CODE", "PROD_CODE_10", "PROD_CODE_20", "PROD_CODE_30", "PROD_CODE_40"]
        ].drop_duplicates()

        return hierarchy_map


    def get_sp_vi_qu(self, df, hierarchy_map, periods, wks, lvls):
        """
        Function to create a DataFrame of total customer spend, quantity, visits over specified time blocks and
        levels of the hierarchy

        :param df: name of the customer transaction file
        :param hierarchy_map: DataFrame containing a lookup between PROD_CODE and all other levels of the hierarchy
        :param periods: List containing the start weeks for a period
        :param wks:  List containing the weeks associasted with the periods
        :param lvls:  Levels of the hierarchy on which to create the metrics - MUST CONTAIN PROD_CODE
        :return: DataFrame containing the summary metrics for each customer

        """

        # Keep only the columns required
        cust_trans = df[
            [
                "CUST_CODE",
                "SHOP_WEEK",
                "PROD_CODE",
                "PROD_CODE_10",
                "PROD_CODE_20",
                "PROD_CODE_30",
                "PROD_CODE_40",
                "SPEND",
                "QUANTITY",
                "BASKET_ID",
            ]
        ]

        # Check that the hierarchy levels list contains PROD_CODE
        assert (
            "PROD_CODE" in lvls
        ), "PROD_CODE must be included in the hierarchy levels list"

        for i in range(0, len(lvls)):
            for j in range(0, len(periods)):

                # Restrict to only the time period required
                cust_summ = cust_trans.loc[cust_trans["SHOP_WEEK"] >= periods[j]]

                # Summarize
                cust_summ = (
                    cust_summ.groupby(["CUST_CODE", lvls[i]])
                    .agg({"SPEND": "sum", "QUANTITY": "sum", "BASKET_ID": "nunique"})
                    .reset_index()
                )
                # Rename columns
                cust_summ.columns = [
                    "CUST_CODE",
                    lvls[i],
                    "SPEND_{}_{}".format(lvls[i], wks[j]),
                    "QUANTITY_{}_{}".format(lvls[i], wks[j]),
                    "VISITS_{}_{}".format(lvls[i], wks[j]),
                ]

                if j == 0:
                    cust_summ_all_wks = cust_summ
                else:
                    cust_summ_all_wks = cust_summ_all_wks.merge(
                        cust_summ, on=["CUST_CODE", lvls[i]], how="left"
                    )

            if lvls[i] == "PROD_CODE":
                cust_summ_final = cust_summ_all_wks
            else:
                hier = hierarchy_map[["PROD_CODE", lvls[i]]]
                cust_summ_final_hier = cust_summ_all_wks.merge(hier, on=lvls[i])
                cust_summ_final = cust_summ_final.merge(
                    cust_summ_final_hier, on=["CUST_CODE", "PROD_CODE"]
                )

        return cust_summ_final.fillna(0)


    def get_chng_features(self, df, wks, lvls):
        """
        Function to create change in spend, visits, quantity over the most recent period

        :param df: DataFrame containing spend, quantity and visits over various time periods and levels of the hierarchy
                   for each customer
        :param wks:  list of weeks to run change calculations
        :param lvls:  list of levels to run change calculations
        :return: DataFrame containing the percentage change over time bands from the most recent time period
        """
        chng_df = df.copy()
        for lvl in lvls:
            for i in range(0, len(wks)):
                for j in range(0, len(wks)):
                    if i < j:
                        # Spend
                        chng_df.loc[
                            :, "CHNG_SPEND_{}_{}_{}".format(lvl, wks[i], wks[j])
                        ] = (
                            chng_df["SPEND_{}_{}".format(lvl, wks[i])]
                            / chng_df["SPEND_{}_{}".format(lvl, wks[j])]
                        )

                        # Quantity
                        chng_df.loc[
                            :, "CHNG_QUANTITY_{}_{}_{}".format(lvl, wks[i], wks[j])
                        ] = (
                            chng_df["QUANTITY_{}_{}".format(lvl, wks[i])]
                            / chng_df["QUANTITY_{}_{}".format(lvl, wks[j])]
                        )

                        # Visits
                        chng_df.loc[
                            :, "CHNG_VISITS_{}_{}_{}".format(lvl, wks[i], wks[j])
                        ] = (
                            chng_df["VISITS_{}_{}".format(lvl, wks[i])]
                            / chng_df["VISITS_{}_{}".format(lvl, wks[j])]
                        )

        # Keep only the change columns
        keep_cols = [col for col in chng_df if "CHNG" in col]
        keep_cols = ["CUST_CODE", "PROD_CODE"] + keep_cols
        chng_df = chng_df[keep_cols]

        return chng_df.fillna(0)


    def time_last_purchased(self, df, hierarchy_map, lvls):
        """
        Function to get the time since last purchased for specified levels of the hierarchy

        :param df: Customer transaction DataFrame
        :param hierarchy_map:  mapping file between PROD_CODE and other levels of the hierarchy
        :param lvls:  list of levels to run change calculations
        :return: DataFrame containing the median time a customer purchased an item or shopped in each level of the hierarchy.
                 DataFrame also contains the median time ALL customers purchased an item or shopped in each level of the
                 hierarchy and the number of weeks since the customer last purchased an item or shopped in each level of the
                 hierarchy
        """

        last_df = df[["CUST_CODE", "SHOP_DATE"] + lvls].copy()
        last_df.drop_duplicates(inplace=True)

        # Check that the hierarchy levels list contains PROD_CODE
        assert (
            "PROD_CODE" in lvls
        ), "PROD_CODE must be included in the hierarchy levels list"

        # Convert the SHOP_DATE variable to datetime
        last_df.loc[:, "SHOP_DATE"] = pd.to_datetime(
            last_df["SHOP_DATE"].astype(str), format="%Y%m%d"
        )

        for i in range(0, len(lvls)):

            # Sort by customer, product and date purchased and dedupe
            lags = last_df.sort_values(
                ["CUST_CODE", lvls[i], "SHOP_DATE"]
            ).drop_duplicates()
            lags = lags[["CUST_CODE", lvls[i], "SHOP_DATE"]]

            # Get the lag of the SHOP_DATE by CUST_CODE and hierarchy level
            lags.loc[:, "LAG_SHOP_DATE"] = lags.groupby(["CUST_CODE", lvls[i]])[
                "SHOP_DATE"
            ].shift(1)
            lags.dropna(inplace=True)

            # Calculate the time between visits
            lags.loc[:, "TIME_BTWN"] = (lags["SHOP_DATE"] - lags["LAG_SHOP_DATE"]).dt.days

            # Now get the median time between visits BY CUST_CODE AND hierarchy level
            median_cust = (
                lags.groupby(["CUST_CODE", lvls[i]])
                .agg({"TIME_BTWN": "median"})
                .reset_index()
            )
            median_cust.rename(
                columns={"TIME_BTWN": "TIME_BTWN_MEDIAN_CUST_{}".format(lvls[i])},
                inplace=True,
            )

            # Now get the median time between visits at the overall hierarchy level for ALL customers
            median_overall = (
                lags.groupby([lvls[i]]).agg({"TIME_BTWN": "median"}).reset_index()
            )
            median_overall.rename(
                columns={"TIME_BTWN": "TIME_BTWN_MEDIAN_OVERALL_{}".format(lvls[i])},
                inplace=True,
            )

            median_final = median_cust.merge(median_overall, on=[lvls[i]])

            # Get the last time between visits for each customer
            last_time_btween = lags.groupby(["CUST_CODE", lvls[i]]).last().reset_index()
            last_time_btween = last_time_btween[["CUST_CODE", lvls[i], "TIME_BTWN"]]
            last_time_btween.rename(
                columns={"TIME_BTWN": "TIME_BTWN_LAST_{}".format(lvls[i])}, inplace=True
            )

            median_final = median_final.merge(last_time_btween, on=["CUST_CODE", lvls[i]])

            if lvls[i] == "PROD_CODE":
                time_since_final = median_final
            else:
                hier = hierarchy_map[["PROD_CODE", lvls[i]]]
                time_since_final_hier = median_final.merge(hier, on=lvls[i])
                time_since_final = time_since_final.merge(
                    time_since_final_hier, on=["CUST_CODE", "PROD_CODE"]
                )

        return time_since_final


    def get_time_since_ratios(self, df, lvls):
        """
        Function to get the ratio of time since last purchased to customer and overall median by levels of the hierarchy

        :param df: DataFrame containing median and overall time between purchases as well as time since last purchase
                   by different levels of the hierarchy
        :param lvls:  list of levels to run change calculations
        :return: DataFrame containing for each customer the ratio between the time since last purchased and median time
                 between purchases by customer and overall for specified levels of the hierarchy
        """
        ratio_df = df.copy()
        for lvl in lvls:
            ratio_df.loc[:, "TIME_BTWN_RATIO_CUST_{}".format(lvl)] = (
                ratio_df["TIME_BTWN_LAST_{}".format(lvl)]
                / ratio_df["TIME_BTWN_MEDIAN_CUST_{}".format(lvl)]
            )
            ratio_df.loc[:, "TIME_BTWN_RATIO_OVERALL_{}".format(lvl)] = (
                ratio_df["TIME_BTWN_LAST_{}".format(lvl)]
                / ratio_df["TIME_BTWN_MEDIAN_OVERALL_{}".format(lvl)]
            )

        # Keep only the ratio columns
        keep_cols = [col for col in ratio_df if "RATIO" in col]
        keep_cols = ["CUST_CODE", "PROD_CODE"] + keep_cols
        ratio_df = ratio_df[keep_cols]

        return ratio_df


    def create_seg_summary(self, df, segs, item_or_cust):
        """
        Function to calculate the spend, visits and quantity (total and proportion) for each basket segment.  Summaries
        can be created by customer or item.  NOTE:  If item is selected then the summary includes only the proportion variables
        as the totals overall will be driven by sample size

        :param df: Customer transaction DataFrame
        :param segs:  List of basket segments on which to create the summaries
        :item_or_cust: Takes value 'CUST_CODE' if summarizing by customer otherwise 'PROD_CODE'
        :return: DataFrame containing the segment level summaries
        """

        # Get total spend, quantity and visits by customer
        tot_cust_summ = (
            df.groupby([item_or_cust])
            .agg({"SPEND": "sum", "QUANTITY": "sum", "BASKET_ID": "nunique"})
            .reset_index()
        )

        tot_cust_summ.columns = [item_or_cust, "TOT_SPEND", "TOT_QUANTITY", "TOT_VISITS"]

        for i in range(0, len(segs)):

            seg_summ = (
                df.groupby([item_or_cust, segs[i]])
                .agg({"SPEND": "sum", "QUANTITY": "sum", "BASKET_ID": "nunique"})
                .reset_index()
            )

            seg_summ.columns = [item_or_cust, segs[i], "SPEND", "QUANTITY", "VISITS"]

            seg_summ = seg_summ.merge(tot_cust_summ, on=[item_or_cust])

            # Calculate proportion of spend, quantity and visits
            seg_summ.loc[:, "PROP_SPEND"] = seg_summ["SPEND"] / seg_summ["TOT_SPEND"]
            seg_summ.loc[:, "PROP_QUANTITY"] = (
                seg_summ["QUANTITY"] / seg_summ["TOT_QUANTITY"]
            )
            seg_summ.loc[:, "PROP_VISITS"] = seg_summ["VISITS"] / seg_summ["TOT_VISITS"]

            seg_summ.drop(["TOT_SPEND", "TOT_QUANTITY", "TOT_VISITS"], axis=1, inplace=True)

            var_list = [
                "SPEND",
                "QUANTITY",
                "VISITS",
                "PROP_SPEND",
                "PROP_QUANTITY",
                "PROP_VISITS",
            ]

            for j in range(0, len(var_list)):
                summ = seg_summ.pivot(
                    index=item_or_cust, columns=segs[i], values=var_list[j]
                ).reset_index()

                # Rename the columns
                cols = [col for col in summ if item_or_cust not in col]
                cols = ["{}_{}_".format(segs[i], var_list[j]) + col for col in cols]
                cols = [item_or_cust] + cols
                summ.columns = cols

                # Fill na with 0
                summ.fillna(0, inplace=True)

                if i == 0 and j == 0:
                    seg_summary = summ
                else:
                    seg_summary = seg_summary.merge(summ, on=[item_or_cust])

        # If the item_or_cust param == 'PROD_CODE' then keep only the proportion features - the totals by item are
        # dependent on sample size
        if item_or_cust == "PROD_CODE":
            keep_cols = [col for col in seg_summary if "PROP" in col]
            keep_cols = ["PROD_CODE"] + keep_cols
            seg_summary = seg_summary[keep_cols]

        return seg_summary


    def merge_train_test(
        self,
        df,
        cust_summ_df,
        chng_df,
        time_since_df,
        ratios_df,
        seg_summ_df,
        user_factors,
        item_factors,
    ):
        """
        Function to merge all feature DataFrames together with the train or test set

        :param df: name of the train or test set
        :param cust_summ_df: name of the DataFrame containing the summarized spend, quantity and visits by
                             levels of the hiearchy
        :param chng_df: name of the DataFrame containing the change in spend, quantity and visits by levels of the hierarchy
        :param time_since_df:  name of the DataFrame containing the time since last purchased data
        :param ratios_df:  name of the DataFrame containing the ratios of spend, quantity and visits by various time bands
                           and levels of the hierarchy
        :param seg_summ_df:  name of the DataFrame containing summaries of the basket segments
        :param user_factors:  name of the DataFrame containing the user factors
        :param item_factors:  name of the DataFrame containing the item factors
        :return: DataFrame for the train or test set containing all generated features

        """

        df = df.copy()

        # Merge with the customer and item summary
        df = df.merge(cust_summ_df, on=["CUST_CODE", "PROD_CODE"], how="left").fillna(0)

        # Merge the change features
        df = df.merge(chng_df, on=["CUST_CODE", "PROD_CODE"], how="left").fillna(0)

        # Merge the time since purchased data
        df = df.merge(time_since_df, on=["CUST_CODE", "PROD_CODE"], how="left")

        # Replace missing with the max of the median time between product purachases overall
        df = df.fillna(df["TIME_BTWN_MEDIAN_OVERALL_PROD_CODE"].max())

        # Merge the time ratio features
        df = df.merge(ratios_df, on=["CUST_CODE", "PROD_CODE"], how="left")

        # Replace missing with the max of the median time between product purachases overall
        df = df.fillna(df["TIME_BTWN_RATIO_OVERALL_PROD_CODE"].max())

        # Merge the basket segment summary
        df = df.merge(seg_summ_df, on="CUST_CODE", how="left").fillna(0)

        # Merge the user factors
        df = df.merge(user_factors, on="CUST_CODE")

        # Merge the item factors
        df = df.merge(item_factors, on="PROD_CODE")

        return df