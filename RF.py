import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
# Load the data
df = pd.read_csv('rf_data.csv')
X = df.iloc[:, 1:75]
y = df.iloc[:, 75]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Initializing the RandomForestClassifier
rf = RandomForestClassifier(
    random_state=42,
    class_weight='balanced'
)

# Setting up RandomizedSearchCV
random_search = RandomizedSearchCV(
    rf, 
    param_distributions=param_grid, 
    n_iter=100, 
    cv=5, 
    random_state=42, 
    n_jobs=-1, 
    verbose=2
)

# Fitting the model with RandomizedSearchCV to find the best parameters
random_search.fit(X_train, y_train)
best_rf = random_search.best_estimator_

# Refitting the best model on the training data
best_rf.fit(X_train, y_train)
y_pred = best_rf.predict(X_test)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Impurity-Based Feature Importance
impurity_feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf.feature_importances_
}).sort_values(by='Importance', ascending=False)
impurity_feature_importances.to_excel('impurity_feature_importance.xlsx', index=False)

# Permutation-Based Feature Importance with Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
permutation_importances_cv = []
for train_idx, test_idx in cv.split(X, y):
    X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
    y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
    
    # Fit model on the current fold
    best_rf.fit(X_train_cv, y_train_cv)
    
    # Compute permutation importance
    perm_importance = permutation_importance(best_rf, X_test_cv, y_test_cv, n_repeats=10, random_state=42)
    permutation_importances_cv.append(perm_importance.importances_mean)

# Average permutation importances across folds
avg_permutation_importances = np.mean(permutation_importances_cv, axis=0)

#DataFrame for permutation importance
perm_feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': avg_permutation_importances
}).sort_values(by='Importance', ascending=False)
perm_feature_importances.to_excel('permutation_feature_importance.xlsx', index=False)

# Normalization of both impurity and permutation importances for comparability
scaler = MinMaxScaler()
impurity_feature_importances['Normalized_Importance'] = scaler.fit_transform(impurity_feature_importances[['Importance']])
perm_feature_importances['Normalized_Importance'] = scaler.fit_transform(perm_feature_importances[['Importance']])

# Merging both dataframes by feature
merged_importances = pd.merge(
    impurity_feature_importances[['Feature', 'Normalized_Importance']],
    perm_feature_importances[['Feature', 'Normalized_Importance']],
    on='Feature',
    suffixes=('_Impurity', '_Permutation')
)

# Calculation of the combined score (average of normalized importances)
merged_importances['Combined_Score'] = merged_importances[['Normalized_Importance_Impurity', 'Normalized_Importance_Permutation']].mean(axis=1)
final_feature_importances = merged_importances[['Feature', 'Combined_Score']].sort_values(by='Combined_Score', ascending=False)
final_feature_importances.to_excel('final_feature_importance.xlsx', index=False)