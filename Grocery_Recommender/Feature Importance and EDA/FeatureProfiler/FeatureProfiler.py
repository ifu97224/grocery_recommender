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
        Pandas dataframe containing the feature and the target
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
