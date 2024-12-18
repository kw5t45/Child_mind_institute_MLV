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
from concurrent.futures import ThreadPoolExecutor
import os
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

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


def load_time_series(dirname) -> pd.DataFrame:
    # opening parquet files and returning dataframe.
    ids = os.listdir(dirname)

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda fname: process_file(fname, dirname), ids), total=len(ids)))

    stats, indexes = zip(*results)

    df = pd.DataFrame(stats, columns=[f"stat_{i}" for i in range(len(stats[0]))])
    df['id'] = indexes
    return df


def process_file(filename, dirname):
    df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
    df.drop(columns=['step', 'battery_voltage', 'non-wear_flag'], axis=1, inplace=True)
    return df.describe().values.reshape(-1), filename.split('=')[1]


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 3),
            nn.ReLU(),
            nn.Linear(encoding_dim * 3, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim * 3),
            nn.ReLU(),
            nn.Linear(input_dim * 3, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def perform_autoencoder(df, encoding_dim=50, epochs=50, batch_size=32):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    data_tensor = torch.FloatTensor(df_scaled)

    input_dim = data_tensor.shape[1]
    autoencoder = AutoEncoder(input_dim, encoding_dim)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters())

    for epoch in range(epochs):
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i: i + batch_size]
            optimizer.zero_grad()
            reconstructed = autoencoder(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}]')

    with torch.no_grad():
        encoded_data = autoencoder.encoder(data_tensor).numpy()

    df_encoded = pd.DataFrame(encoded_data, columns=[f'Enc_{i + 1}' for i in range(encoded_data.shape[1])])

    return df_encoded


# funciton that trains any regressor model using kfold cross validation, k hard coded = 5
def TrainML(model_class, test_data, acc=False) -> list[int]:
    if not acc:
        global train
        X = train.drop(['sii'], axis=1)
        y = train_labels
        test_data = test_data[X.columns]  # Reorder test_data columns to match X

        ################
        scaler = StandardScaler()
        scaler.fit(X)
        X = pd.DataFrame(scaler.transform(X), columns=X.columns)

        # ids are stored in test_ids variable
        #test_data = test_data.drop(columns='id')
        test_data = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)
        #############
    else:
        global train_ts
        y = train_ts['sii']
        train_ts = train_ts.drop(columns=['id', 'sii'], axis=1)
        X = train_ts
    n_splits = 5
    random_state = 42

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
    return tp_rounded.tolist(), tKappa



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

#
LGBM_params_acc = {

    'learning_rate': 0.01,
    'max_depth': 3,
    'num_leaves': 31,
    'min_child_samples': 10,
    'min_data_in_leaf': 13,
    'feature_fraction': 0.893,
    'bagging_fraction': 0.784,
    'bagging_freq': 4,
    'lambda_l1': 10,
    'lambda_l2': 0.01
}


XGB_Params_acc = {
    'learning_rate': 0.01,
    'max_depth': 3,
    'n_estimators': 300,
    'num_leaves': 31,
    'min_child_samples': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1,
    'reg_lambda': 5,
    'random_state': 42,
    'tree_method': 'exact'
}


CatBoost_Params_acc = {
    'learning_rate': 0.01,
    'depth': 3,
    'min_child_samples': 10,
    'num_leaves': 31,
    'iterations': 300,
    'random_seed': 42,
    'verbose': 0,
    'l2_leaf_reg': 10
}



#####################################
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

##################
print('Loading train timeseries data...')
train_ts = load_time_series("series_train.parquet")
print('Loading test timeseries data...')
test_ts = load_time_series("series_test.parquet")
print(f'Shape of Train accelerometer data: {train_ts.shape}')
print(f'Shape of Test accelerometer data: {test_ts.shape}')

train_ts = pd.merge(train_ts, df[['id', 'sii']], how='left', on='id')

######################
Light = LGBMRegressor(**LGBM_params_acc, random_state=42, verbose=-1, n_estimators=300)
XGB_Model = XGBRegressor(**XGB_Params_acc)
CatBoost_Model = CatBoostRegressor(**CatBoost_Params_acc)

voting_model = VotingRegressor(estimators=[
    ('lightgbm', Light),
    ('xgboost', XGB_Model),
    ('catboost', CatBoost_Model)],
     weights=[1, 1, 1]
)
ttsid = train_ts['id']
testtsid = test_ts['id']


train_ts = train_ts.drop(columns=['id'])
test_ts = test_ts.drop(columns=['id'])

train_ts_encoded = perform_autoencoder(train_ts, encoding_dim=40, epochs=100, batch_size=32)
test_ts_encoded = perform_autoencoder(test_ts, encoding_dim=40, epochs=100, batch_size=32)

train_ts_encoded['id'] = ttsid
test_ts_encoded['id'] = testtsid

train = pd.merge(df, train_ts_encoded, how="left", on='id')
test = pd.merge(test, test_ts_encoded, how="left", on='id')

# test_ts_ds = test_ts['id']
# test_ts = test_ts.drop(columns=['id'])

# vote_preds = TrainML(model_class=voting_model, test_data=test_ts, acc=True)
# acc_sub = pd.DataFrame({

#     'id': test_ts_ds,
#     'sii': vote_preds
# })

# acc_sub
################
# df = df.drop(columns=['id'])
# # df = df.drop(columns=['sii'])
# imputer = KNNImputer(n_neighbors=4)  # k=4
# imputed_data = imputer.fit_transform(df)

# df = pd.DataFrame(imputed_data, columns=df.columns)
# df['id'] = train_ids
# train = df
####
Light = LGBMRegressor(**LGBM_params, random_state=42, verbose=-1, n_estimators=300)
XGB_Model = XGBRegressor(**XGB_Params)
CatBoost_Model = CatBoostRegressor(**CatBoost_Params)
##########
best_weights = None
best_score = -float('inf')  # Assuming higher QWK score is better
train = train.drop(columns=['id'])

# Iterate over weights with step of 0.1
for i in np.arange(0, 1.1, 0.1):  # Loop for i
    for j in np.arange(0, 1.1, 0.1):  # Loop for j
        k = 1 - (i + j)  # Ensure the sum of weights equals 1
        if k < 0 or k > 1:  # Skip invalid weight combinations
            continue

        # Define the VotingRegressor with current weights
        voting_model = VotingRegressor(estimators=[
            ('lightgbm', Light),
            ('xgboost', XGB_Model),
            ('catboost', CatBoost_Model)],
            weights=[i, j, k]
        )

        # Drop the 'id' column from training data

        # Train the model and get predictions and QWK score
        vote_preds, score = TrainML(model_class=voting_model, test_data=test, acc=False)

        # Track the best weights and score
        if score > best_score:
            best_score = score
            best_weights = [i, j, k]
            print(f"New Best Weights: {best_weights}, Score: {best_score}")

# Print the final best weights and score
print(f"Optimal Weights: {best_weights}, Best Score: {best_score}")
# vote_sub = pd.DataFrame({

#     'id': test['id'],
#     'sii': vote_preds
# })
# vote_sub.to_csv('submission.csv', index=False)
# print(score)
