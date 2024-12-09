import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score
from sklearn.impute import KNNImputer
from scipy.optimize import minimize
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten, MaxPooling1D, Embedding, GlobalMaxPooling1D, Input
from tensorflow.keras.models import Model
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor

# Load and preprocess data
df = pd.read_csv("/kaggle/input/child-mind-institute-problematic-internet-use/train.csv")
df = df.dropna(subset=['sii'])  # Keep labeled values only
test = pd.read_csv("/kaggle/input/child-mind-institute-problematic-internet-use/test.csv")
season_mapping = {
    'Winter': -1,
    'Spring': -0.5,
    'Summer': 0.5,
    'Fall': 1
}
df = df.replace(season_mapping)
test = test.replace(season_mapping)

test_missing_columns = set(df.columns) - set(test.columns)
for col in test_missing_columns:
    if col != 'sii':  # Retain the target column for training
        df.drop(columns=col, inplace=True)

train_ids = df['id']
test_ids = test['id']
train_labels = df['sii']
df = df.drop(columns=['id'])
df = df.drop(columns=['sii'])

imputer = KNNImputer(n_neighbors=4)
df_imputed = imputer.fit_transform(df)
test_imputed = imputer.transform(test.drop(columns=['id']))

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_imputed)
test_scaled = scaler.transform(test_imputed)

X = pd.DataFrame(df_scaled, columns=df.columns)
y = train_labels

# CNN model for numeric features
def build_cnn(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        Dropout(0.3),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='linear')  # Regression output
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Custom threshold rounding
def threshold_rounder(preds, thresholds):
    return np.where(preds < thresholds[0], 0,
                    np.where(preds < thresholds[1], 1,
                             np.where(preds < thresholds[2], 2, 3)))

# QWK evaluation
def evaluate_predictions(thresholds, y_true, preds):
    rounded_preds = threshold_rounder(preds, thresholds)
    return -cohen_kappa_score(y_true, rounded_preds, weights='quadratic')

# Training
n_splits = 5
random_state = 42
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
test_preds = np.zeros((len(test_scaled), n_splits))
oof_preds = np.zeros(len(y))

for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(X, y), total=n_splits, desc="Training Folds")):
    X_train, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
    y_train, y_val = y.iloc[train_idx].values, y.iloc[val_idx].values

    # Reshape for Conv1D
    X_train = X_train.reshape(-1, X_train.shape[1], 1)
    X_val = X_val.reshape(-1, X_val.shape[1], 1)
    test_reshaped = test_scaled.reshape(-1, test_scaled.shape[1], 1)

    # Train CNN
    cnn = build_cnn(input_shape=(X_train.shape[1], 1))
    cnn.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, verbose=1)

    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
    rf.fit(X_train.reshape(X_train.shape[0], -1), y_train)

    # Combine predictions
    cnn_val_preds = cnn.predict(X_val).flatten()
    rf_val_preds = rf.predict(X_val.reshape(X_val.shape[0], -1))

    val_preds = (cnn_val_preds + rf_val_preds) / 2

    # Test predictions
    cnn_test_preds = cnn.predict(test_reshaped).flatten()
    rf_test_preds = rf.predict(test_scaled)

    test_fold_preds = (cnn_test_preds + rf_test_preds) / 2

    oof_preds[val_idx] = val_preds
    test_preds[:, fold] = test_fold_preds

    val_qwk = cohen_kappa_score(y_val, threshold_rounder(val_preds, [0.5, 1.5, 2.5]), weights='quadratic')
    print(f"Fold {fold + 1} - Validation QWK: {val_qwk:.4f}")

# Optimize thresholds
opt_result = minimize(evaluate_predictions, [0.5, 1.5, 2.5], args=(y, oof_preds), method='Nelder-Mead')
optimal_thresholds = opt_result.x
final_qwk = cohen_kappa_score(y, threshold_rounder(oof_preds, optimal_thresholds), weights='quadratic')
print(f"Optimized QWK: {final_qwk:.4f}")

# Test predictions
final_test_preds = test_preds.mean(axis=1)
final_test_preds = threshold_rounder(final_test_preds, optimal_thresholds)

submission = pd.DataFrame({'id': test_ids, 'sii': final_test_preds})
submission.to_csv('/kaggle/working/submission.csv', index=False)
print(submission)
