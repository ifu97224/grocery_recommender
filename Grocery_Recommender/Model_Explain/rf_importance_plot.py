from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_rf_importance(train_df, n_estimators, max_features, max_depth, top_n):
    """Function to train a Random Forest model and plot the RF Importance

        Parameters
        ----------
        train_df : Pandas DataFrame
            Pandas DataFrame containing the training data
        n_estimators : int
            number of estimators to use for fitting the model
        max_features : int
            maximum number of features to consider at every tree split
        max_depth : int
            maximum depth of each tree

        Returns
        -------
        plt : Matplotlib object
            Matplotlib object containing the RF Importance plot
        top_features_list : list
            list containing the top features from RF Importance (and the target variable)

        """

    X_train = train_df.drop("TARGET", axis=1)
    y_train = train_df[["TARGET"]]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
        random_state=0,
        class_weight="balanced",
        oob_score=True,
        n_jobs=-1,
    )

    print("Training Random Forest model")
    model.fit(X_train, y_train.values.ravel())

    ### Get the RF Importance
    feat_labels = pd.DataFrame(X_train.columns)

    importances = pd.DataFrame(model.feature_importances_)

    rf_importances = feat_labels.merge(importances, left_index=True, right_index=True)
    rf_importances.columns = ["features", "rf_importance"]
    rf_importances.sort_values("rf_importance", ascending=False, inplace=True)
    rf_importances["rf_rank"] = range(1, len(rf_importances) + 1)

    top_rf_importance = rf_importances.loc[rf_importances["rf_rank"] <= top_n]

    ### Plot the RF Importance
    plt.figure(figsize=(13, 8))
    ax = plt.gca()

    N = len(top_rf_importance)
    ind = np.arange(N)

    sns.barplot(
        x="features",
        y="rf_importance",
        data=top_rf_importance,
        capsize=0.05,
        errcolor="gray",
        errwidth=2,
        ci="sd",
        color="blue",
    )

    ax.set_ylabel("RF Importance")
    ax.set_xlabel("Features")
    ax.set_xticks(ind)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(labelrotation=90, axis="x")

    # Get a list of the top features and the target variable
    top_features_list = top_rf_importance["features"].to_list()
    top_features_list = top_features_list + ["TARGET"]

    return plt, top_features_list