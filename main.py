import numpy as np
import pandas as pd
from colorama import Fore, Style

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize
from sklearn.ensemble import VotingRegressor


from tqdm import tqdm


df = pd.read_csv("train.csv")
df = df.dropna(subset=['sii']) # keeping labeled values only
test = pd.read_csv("test.csv")
season_mapping = {
    'Winter': -1,
    'Spring': -0.5,
    'Summer': 0.5,
    'Fall': 1
}
# mapping non-string values
df = df.replace(season_mapping)
test = test.replace(season_mapping)

# dropping questions not in test dataset
test_missing_columns = set(df.columns) - set(test.columns)
for col in test_missing_columns:
    if col != 'sii':  # Retain the target column for training
        df.drop(columns=col, inplace=True)
# for later use
train_ids = df['id']
test_ids = test['id']
train_labels = df['sii']
df = df.drop(columns=['id'])


# weak correlation dropping
# correlations = df.corr()['sii']  # Get correlations of all features with 'sii'
#
# # Identify features to drop based on correlation threshold
# weak_corr_features = correlations[(correlations > -0.2) & (correlations < 0.2)].index
#
# # Drop weakly correlated features
# df = df.drop(columns=weak_corr_features)
# test = test.drop(columns=weak_corr_features)

df = df.drop(columns=['sii'])
imputer = KNNImputer(n_neighbors=4)  # k=4
imputed_data = imputer.fit_transform(df)

train = pd.DataFrame(imputed_data, columns=df.columns)
train[:5]


# qwk score
def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


# threshold rounder
def threshold_Rounder(oof_non_rounded, thresholds):
    return np.where(oof_non_rounded < thresholds[0], 0,
                    np.where(oof_non_rounded < thresholds[1], 1,
                             np.where(oof_non_rounded < thresholds[2], 2, 3)))


# prediction evaluation using qwk function
def evaluate_predictions(thresholds, y_true, oof_non_rounded):
    rounded_p = threshold_Rounder(oof_non_rounded, thresholds)
    return -quadratic_weighted_kappa(y_true, rounded_p)


# funciton that trains any regressor model using kfold cross validation, k hard coded = 5
def TrainML(model_class, test_data) -> list[int]:
    X = train  # .drop(['sii'], axis=1)
    y = train_labels
    n_splits = 5
    random_state = 42

    ################
    scaler = StandardScaler()

    scaler.fit(X)

    X = pd.DataFrame(scaler.transform(X), columns=X.columns)

    # ids are stored in test_ids variable
    test_data = test_data.drop(columns='id')
    test_data = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)
    #############
    SKF = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    train_S = []
    test_S = []

    oof_non_rounded = np.zeros(len(y), dtype=float)
    oof_rounded = np.zeros(len(y), dtype=int)
    test_preds = np.zeros((len(test_data), n_splits))

    for fold, (train_idx, test_idx) in enumerate(tqdm(SKF.split(X, y), desc="Training Folds", total=n_splits)):
        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]
        model = clone(model_class)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        oof_non_rounded[test_idx] = y_val_pred
        y_val_pred_rounded = y_val_pred.round(0).astype(int)
        oof_rounded[test_idx] = y_val_pred_rounded

        train_kappa = quadratic_weighted_kappa(y_train, y_train_pred.round(0).astype(int))
        val_kappa = quadratic_weighted_kappa(y_val, y_val_pred_rounded)

        train_S.append(train_kappa)
        test_S.append(val_kappa)

        test_preds[:, fold] = model.predict(test_data)

        print(f"Fold {fold + 1} - Train QWK: {train_kappa:.4f}, Validation QWK: {val_kappa:.4f}")

    print(f"Mean Train QWK --> {np.mean(train_S):.4f}")
    print(f"Mean Validation QWK ---> {np.mean(test_S):.4f}")

    KappaOPtimizer = minimize(evaluate_predictions,
                              x0=[0.5, 1.5, 2.5], args=(y, oof_non_rounded),
                              method='Nelder-Mead')
    assert KappaOPtimizer.success, "Optimization did not converge."

    oof_tuned = threshold_Rounder(oof_non_rounded, KappaOPtimizer.x)
    tKappa = quadratic_weighted_kappa(y, oof_tuned)

    print(f"----> || Optimized QWK SCORE :: {Fore.CYAN}{Style.BRIGHT} {tKappa:.3f}{Style.RESET_ALL}")

    tpm = test_preds.mean(axis=1)
    tp_rounded = threshold_Rounder(tpm, KappaOPtimizer.x)
    return tp_rounded.tolist()



LGBM_params = {
    'learning_rate': 0.046,
    'max_depth': 12,
    'num_leaves': 478,
    'min_data_in_leaf': 13,
    'feature_fraction': 0.893,
    'bagging_fraction': 0.784,
    'bagging_freq': 4,
    'lambda_l1': 10,
    'lambda_l2': 0.01
}


XGB_Params = {
    'learning_rate': 0.05,
    'max_depth': 6,
    'n_estimators': 400,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1,
    'reg_lambda': 5,
    'random_state': 42,
    'tree_method': 'exact'
}


CatBoost_Params = {
    'learning_rate': 0.05,
    'depth': 6,
    'iterations': 400,
    'random_seed': 42,
    'verbose': 0,
    'l2_leaf_reg': 10
}
Light = LGBMRegressor(**LGBM_params, random_state=42, verbose=-1, n_estimators=300)
XGB_Model = XGBRegressor(**XGB_Params)
CatBoost_Model = CatBoostRegressor(**CatBoost_Params)

voting_model = VotingRegressor(estimators=[
    ('lightgbm', Light),
    ('xgboost', XGB_Model),
    ('catboost', CatBoost_Model)],
     weights=[0.3, 0.5, 0.2]
)

vote_preds = TrainML(model_class=voting_model, test_data=test)
final_sub = pd.DataFrame({

    'id': test_ids,
    'sii': vote_preds
})

#final_sub.to_csv('submission.csv', index=False)
print(final_sub)