import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
import warnings
import numpy as np
import pandas as pd


def feature_profiling(df, feat, target, hist_bins):
    """Generate data profiling for features.  For numeric features generate:

        1.) histograms
        2.) boxplots
        3.) basic stats (mean, min, max, standard deviation, quartiles and counts)
        4.) counts of missing values

    For categorical features generate:

        1.) count plot
        2.) bar plot
        3.) counts of missing values

    Parameters
    ----------
    df : pandas.DataFrame
        Pandas DataFrame containing the feature and the target
    feat : str
        name of the feature for profiling
    target : str
        name of the target variable
    hist_bins : int
        number of bins for the histogram

    """

    sns.set(
        style="whitegrid",
        palette="deep",
        font_scale=1.1,
        rc={"figure.figsize": [15, len(feat) * 6.5]},
    )

    # Run missing value checks
    for i in range(0, len(feat)):
        # Count of missing values
        print(
            "The feature {} has {} missing value(s) out of {}".format(
                feat[i], np.sum(df[[feat[i]]].isna())[0], df.shape[0]
            )
        )

    iter = 1

    for i in range(0, len(feat)):

        # If feature is categorical run the categorical profiling
        if df[feat[i]].dtypes == object:

            # Count plot
            plt.subplot(len(feat), 2, iter)
            sns.countplot(df[feat[i]])

            # Bar Plot
            plt.subplot(len(feat), 2, iter + 1)
            sns.barplot(
                x=feat[i],
                y=target,
                data=df,
                capsize=0.05,
                errcolor="gray",
                errwidth=2,
                ci="sd",
            )

            iter = iter + 2

        # If feature is numeric run numeric profiling
        else:

            print("Key stats for {}".format(feat[i]))
            display(df[[feat[i]]].describe())

            # Histogram
            plt.subplot(len(feat), 2, iter)
            sns.distplot(
                df[feat[i]],
                norm_hist=False,
                kde=True,
                bins=hist_bins,
                hist_kws={"alpha": 1},
            ).set(
                xlabel=feat[i], ylabel="Count", title="Histogram of {}".format(feat[i])
            )

            # Boxplot
            plt.subplot(len(feat), 2, iter + 1)
            sns.boxplot(x=df[target], y=df[feat[i]]).set(
                title="Box Plot of {} against target".format(feat[i])
            )

            iter = iter + 2


def get_paired_correlations(df, plot_heatmap, num_pairs):
    """Calculate the pearson correlation between pairs of items in a DataFrame:

    Parameters
    ----------
    df : pandas.DataFrame
        Pandas DataFrame containing the features on which to run the correlations
    plot_heatmap : bool
        Flag identifying if a heatmap should be generated
    num_pairs : int
        The number of pairs to return the correlation for
    """

    num_vars = df.select_dtypes(exclude=["object", "datetime"])

    corr_data = num_vars.copy()
    corr = corr_data.corr("pearson")

    if plot_heatmap:
        # Plot a heatmap since there aren't that many variables in this dataset
        plt.figure(1, figsize=(16, 12))
        sns.heatmap(
            corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values
        )

        # Look at the top correlated pairs by absolute correlation value
        corr = corr.abs()

        s = corr.unstack()
        so = s.sort_values(kind="quicksort", ascending=False)

    def get_redundant_pairs(df):
        # Get diagnoal and lower triangular pairs of the correlation matrix
        pairs_to_drop = set()
        cols = df.columns
        for i in range(0, df.shape[1]):
            for j in range(0, i + 1):
                pairs_to_drop.add((cols[i], cols[j]))
        return pairs_to_drop

    au_corr = corr_data.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(corr_data)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

    print("Top Absolute Correlations")
    display(au_corr[0:num_pairs])
