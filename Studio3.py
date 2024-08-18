import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load multiple datasets from CSV files and combine them into one DataFrame
df_w1 = pd.read_csv('w1.csv')
df_w2 = pd.read_csv('w2.csv')
df_w3 = pd.read_csv('w3.csv')
df_w4 = pd.read_csv('w4.csv')

# Concatenate all datasets vertically and reset the index
combined_data = pd.concat([df_w1, df_w2, df_w3, df_w4], ignore_index=True)
# Save the combined data into a new CSV file for future use
combined_data.to_csv('combined_data.csv', index=False)

# Shuffle the combined dataset to ensure randomness in data distribution
shuffled_data = combined_data.sample(frac=1).reset_index(drop=True)
# Save the shuffled dataset to a new CSV file
shuffled_data.to_csv('all_data.csv', index=False)

# Step 2: Prepare the data for training an SVM model
# Split the data into features (X) and target labels (y)
X = shuffled_data.iloc[:, :-1]
y = shuffled_data.iloc[:, -1]

# Further split the data into training and testing sets using a 70-30 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Initialize and train an SVM model on the training data
clf = svm.SVC()
clf.fit(X_train, y_train)

# Predict the target labels for the test data and calculate accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Train-Test Accuracy:", accuracy)

# Perform 10-fold cross-validation on the entire dataset and print the scores
scores = cross_val_score(clf, X, y, cv=10)
print("Cross Validation Scores:", scores)

# Step 3: Hyperparameter tuning using GridSearchCV
# Define a parameter grid to search over different values of C, gamma, and kernel
param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf']}
# Use GridSearchCV to find the best hyperparameters
grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train, y_train)

# Print the best parameters found and the corresponding accuracy
print("Best Parameters:", grid.best_params_)
best_svm = grid.best_estimator_
y_pred_best = best_svm.predict(X_test)
best_accuracy = accuracy_score(y_test, y_pred_best)
print("Best Train-Test Accuracy after Hyperparameter Tuning:", best_accuracy)

# Step 4: Feature selection using SelectKBest
# Select the top 100 features based on the ANOVA F-value between the feature and target
selector = SelectKBest(f_classif, k=100)
X_new = selector.fit_transform(X, y)

# Split the new feature set into training and testing sets
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y, test_size=0.3, random_state=1)
# Train an SVM model on the selected features and calculate accuracy
clf_new = svm.SVC()
clf_new.fit(X_train_new, y_train_new)
y_pred_new = clf_new.predict(X_test_new)
accuracy_new = accuracy_score(y_test_new, y_pred_new)
print("Train-Test Accuracy with Feature Selection:", accuracy_new)

# Step 5: Dimensionality reduction using PCA
# Reduce the feature set to 10 principal components
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

# Split the reduced feature set into training and testing sets
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=1)
# Train an SVM model on the principal components and calculate accuracy
clf_pca = svm.SVC()
clf_pca.fit(X_train_pca, y_train_pca)
y_pred_pca = clf_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test_pca, y_pred_pca)
print("Train-Test Accuracy with PCA:", accuracy_pca)

# Step 6: Summarize the results of different SVM models in a table
svm_data = {
    'Model': ['Original Features', 'Hyperparameter Tuning', 'Feature Selection', 'PCA'],
    'Train-Test Accuracy': [accuracy, best_accuracy, accuracy_new, accuracy_pca],
    'Cross Validation': [scores.mean(), grid.best_score_, None, None]
}
svm_summary_table = pd.DataFrame(svm_data)
print(svm_summary_table)

# Step 7: Train other classifiers (SGD, RandomForest, MLP) and summarize their performance
# SGD Classifier
sgd_clf = SGDClassifier()
sgd_clf.fit(X_train, y_train)
sgd_y_pred = sgd_clf.predict(X_test)
sgd_accuracy = accuracy_score(y_test, sgd_y_pred)
sgd_cv_scores = cross_val_score(sgd_clf, X, y, cv=10)
sgd_cv_accuracy = sgd_cv_scores.mean()
print("SGD Classifier Accuracy:", sgd_accuracy)
print("SGD Cross-Validation Accuracy:", sgd_cv_accuracy)

# RandomForest Classifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
rf_y_pred = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_cv_scores = cross_val_score(rf_clf, X, y, cv=10)
rf_cv_accuracy = rf_cv_scores.mean()
print("RandomForest Classifier Accuracy:", rf_accuracy)
print("RandomForest Cross-Validation Accuracy:", rf_cv_accuracy)

# MLP Classifier
mlp_clf = MLPClassifier()
mlp_clf.fit(X_train, y_train)
mlp_y_pred = mlp_clf.predict(X_test)
mlp_accuracy = accuracy_score(y_test, mlp_y_pred)
mlp_cv_scores = cross_val_score(mlp_clf, X, y, cv=10)
mlp_cv_accuracy = mlp_cv_scores.mean()
print("MLP Classifier Accuracy:", mlp_accuracy)
print("MLP Cross-Validation Accuracy:", mlp_cv_accuracy)

# Create a summary table for the performance of all classifiers
model_data = {
    'Model': ['SVM', 'SGD', 'RandomForest', 'MLP'],
    'Train-Test Accuracy': [accuracy, sgd_accuracy, rf_accuracy, mlp_accuracy],
    'Cross Validation': [scores.mean(), sgd_cv_accuracy, rf_cv_accuracy, mlp_cv_accuracy]
}
all_models_summary_table = pd.DataFrame(model_data)
print(all_models_summary_table)
