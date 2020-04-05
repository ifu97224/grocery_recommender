from sklearn.ensemble import RandomForestClassifier
import shap

def get_shap_values(df, top_features_list, n_estimators, max_features, max_depth):
    """Function to get the Shap values from a Random Forest model

    Parameters
    ----------
    df : Pandas DataFrame
        Pandas DataFrame containing the training data
    top_features_list : list
        list containing the column names of the top features (+ target)
    n_estimators : int
        the number of trees to train
    max_features: int
        the maximum number of features to include in each tree
    max_depth : int
        the maximum depth of each tree

    Returns
    -------
    shap_values : Shap object
        Shap values for the model fitted on the top features

    """

    top_features = df.loc[:, top_features_list]
    y_train = top_features[["TARGET"]]
    top_features.drop("TARGET", axis=1, inplace=True)

    model_top_feat = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
        random_state=0,
        class_weight="balanced",
        oob_score=True,
        n_jobs=-1,
    )

    print("Training Random Forest model")
    model_top_feat.fit(top_features, y_train.values.ravel())

    print("Generating Shap values")
    explainer = shap.TreeExplainer(
        model_top_feat, feature_perturbation="tree_path_dependent"
    )

    shap_values = explainer.shap_values(top_features)

    return shap_values