import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# Step 1: Data Collection
# Function to load and preprocess data from Boning and Slicing CSV files
def data_collection():
    # Load data from CSV files into pandas DataFrames
    df_boning = pd.read_csv('Boning.csv')
    df_slicing = pd.read_csv('Slicing.csv')
    
    # Define the columns of interest, representing x, y, z coordinates for both feet
    columns = ['Frame', 'Right Foot x', 'Right Foot y', 'Right Foot z', 'Left Foot x', 'Left Foot y', 'Left Foot z']
    
    # Select relevant columns and assign a class label (0 for Boning, 1 for Slicing)
    df_boning_selected = df_boning[columns]
    df_boning_selected['class'] = 0  # Labeling Boning as class 0
    df_slicing_selected = df_slicing[columns]
    df_slicing_selected['class'] = 1  # Labeling Slicing as class 1
    
    # Combine both datasets into a single DataFrame
    df_combined = pd.concat([df_boning_selected, df_slicing_selected], ignore_index=True)
    
    # Save the combined dataset into a new CSV file
    df_combined.to_csv('combined_data.csv', index=False)
    return df_combined

# Step 2: Create Composite Columns
# Function to compute composite features based on existing columns
def compute_composite_columns(df):
    # Calculate the Root Mean Square (RMS) for right and left foot coordinates
    df['rms_xy_right'] = np.sqrt(df['Right Foot x']**2 + df['Right Foot y']**2)
    df['rms_yz_right'] = np.sqrt(df['Right Foot y']**2 + df['Right Foot z']**2)
    df['rms_zx_right'] = np.sqrt(df['Right Foot z']**2 + df['Right Foot x']**2)
    df['rms_xyz_right'] = np.sqrt(df['Right Foot x']**2 + df['Right Foot y']**2 + df['Right Foot z']**2)
    df['rms_xy_left'] = np.sqrt(df['Left Foot x']**2 + df['Left Foot y']**2)
    df['rms_yz_left'] = np.sqrt(df['Left Foot y']**2 + df['Left Foot z']**2)
    df['rms_zx_left'] = np.sqrt(df['Left Foot z']**2 + df['Left Foot x']**2)
    df['rms_xyz_left'] = np.sqrt(df['Left Foot x']**2 + df['Left Foot y']**2 + df['Left Foot z']**2)
    
    # Calculate the roll and pitch angles for both feet using arctan2
    df['roll_right'] = np.degrees(np.arctan2(df['Right Foot y'], np.sqrt(df['Right Foot x']**2 + df['Right Foot z']**2)))
    df['pitch_right'] = np.degrees(np.arctan2(df['Right Foot x'], np.sqrt(df['Right Foot y']**2 + df['Right Foot z']**2)))
    df['roll_left'] = np.degrees(np.arctan2(df['Left Foot y'], np.sqrt(df['Left Foot x']**2 + df['Left Foot z']**2)))
    df['pitch_left'] = np.degrees(np.arctan2(df['Left Foot x'], np.sqrt(df['Left Foot y']**2 + df['Left Foot z']**2)))
    
    # Save the DataFrame with the new composite columns into a CSV file
    df.to_csv('combined_data_with_composites.csv', index=False)
    return df

# Step 3: Data Pre-processing and Feature Computation
# Function to compute statistical features from the dataset
def compute_statistical_features(df):
    # Initialize an empty DataFrame to store the computed features
    features = pd.DataFrame()
    
    # Loop through each column to calculate statistical features
    for column in df.columns:
        if column != 'Frame' and column != 'class':  # Exclude Frame and class columns
            features[column + '_mean'] = [df[column].mean()]
            features[column + '_std'] = [df[column].std()]
            features[column + '_min'] = [df[column].min()]
            features[column + '_max'] = [df[column].max()]
            features[column + '_auc'] = [np.trapz(df[column])]
            features[column + '_peaks'] = [len(df[column][df[column] > df[column].mean()])]
    return features

# Function to extract features over fixed-length windows and assign class labels
def feature_extraction(df):
    # Compute statistical features in windows of 60 frames and concatenate the results
    df_features = pd.concat([compute_statistical_features(df.iloc[i:i+60]) for i in range(0, len(df), 60)], ignore_index=True)
    
    # Assign the class label for each window based on the first frame in that window
    df_features['class'] = df['class'][::60].values
    
    # Save the processed features into a CSV file
    df_features.to_csv('processed_features.csv', index=False)
    return df_features

# Step 4: Model Training and Evaluation
# Function to train and evaluate different machine learning models
def model_training_and_evaluation(df_features):
    # Separate features (X) from labels (y)
    X = df_features.drop('class', axis=1)
    y = df_features['class']
    
    # Split the data into training and testing sets (70-30 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Train an SVM model
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    
    # Predict the test set and compute accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("SVM Train-Test Accuracy:", accuracy)

    # Perform 10-fold cross-validation and print the mean accuracy
    scores = cross_val_score(clf, X, y, cv=10)
    print("SVM Cross Validation Accuracy:", scores.mean())

    # Hyperparameter tuning for SVM using GridSearchCV
    param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf']}
    grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2)
    grid.fit(X_train, y_train)
    
    # Evaluate the best model from GridSearchCV
    best_svm = grid.best_estimator_
    y_pred_best = best_svm.predict(X_test)
    best_accuracy = accuracy_score(y_test, y_pred_best)
    print("SVM Best Train-Test Accuracy after Hyperparameter Tuning:", best_accuracy)

    # Feature selection using SelectKBest
    selector = SelectKBest(f_classif, k=10)
    X_new = selector.fit_transform(X, y)
    
    # Train and evaluate SVM on the selected features
    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y, test_size=0.3, random_state=1)
    clf_new = svm.SVC()
    clf_new.fit(X_train_new, y_train_new)
    y_pred_new = clf_new.predict(X_test_new)
    accuracy_new = accuracy_score(y_test_new, y_pred_new)
    print("SVM Train-Test Accuracy with Feature Selection:", accuracy_new)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X)
    
    # Train and evaluate SVM on the principal components
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=1)
    clf_pca = svm.SVC()
    clf_pca.fit(X_train_pca, y_train_pca)
    y_pred_pca = clf_pca.predict(X_test_pca)
    accuracy_pca = accuracy_score(y_test_pca, y_pred_pca)
    print("SVM Train-Test Accuracy with PCA:", accuracy_pca)

    # Train and evaluate additional models: SGD, RandomForest, MLP
    sgd_clf = SGDClassifier()
    sgd_clf.fit(X_train, y_train)
    sgd_y_pred = sgd_clf.predict(X_test)
    sgd_accuracy = accuracy_score(y_test, sgd_y_pred)
    sgd_cv_accuracy = cross_val_score(sgd_clf, X, y, cv=10).mean()
    print("SGD Classifier Accuracy:", sgd_accuracy)
    print("SGD Cross-Validation Accuracy:", sgd_cv_accuracy)

    rf_clf = RandomForestClassifier()
    rf_clf.fit(X_train, y_train)
    rf_y_pred = rf_clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_y_pred)
    rf_cv_accuracy = cross_val_score(rf_clf, X, y, cv=10).mean()
    print("RandomForest Classifier Accuracy:", rf_accuracy)
    print("RandomForest Cross-Validation Accuracy:", rf_cv_accuracy)

    mlp_clf = MLPClassifier()
    mlp_clf.fit(X_train, y_train)
    mlp_y_pred = mlp_clf.predict(X_test)
    mlp_accuracy = accuracy_score(y_test, mlp_y_pred)
    mlp_cv_accuracy = cross_val_score(mlp_clf, X, y, cv=10).mean()
    print("MLP Classifier Accuracy:", mlp_accuracy)
    print("MLP Cross-Validation Accuracy:", mlp_cv_accuracy)

    # Visualization: Feature Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = X_train.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.show()

    # Visualization: Feature Importances for RandomForest
    plt.figure(figsize=(10, 6))
    importances = rf_clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.title('Feature Importances - RandomForest')
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.tight_layout()
    plt.show()

    # Visualization: Model Accuracy Comparison
    models = ['SVM', 'SGD', 'RandomForest', 'MLP']
    accuracies = [accuracy, sgd_accuracy, rf_accuracy, mlp_accuracy]
    plt.figure(figsize=(8, 6))
    plt.bar(models, accuracies, color='skyblue')
    plt.ylim(0, 1)
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.show()

    # Boxplots: Analysis of Peaks and AUC Distributions
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=X_train.filter(regex='_peaks'))
    plt.title('Boxplot of Peaks Distributions')
    plt.xticks(rotation=90)

    plt.subplot(1, 2, 2)
    sns.boxplot(data=X_train.filter(regex='_auc'))
    plt.title('Boxplot of AUC Distributions')
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()

# Main Execution
# Execute the workflow by calling the functions in sequence
if __name__ == "__main__":
    df_combined = data_collection()  # Step 1: Data Collection
    df_combined = compute_composite_columns(df_combined)  # Step 2: Compute Composite Columns
    df_features = feature_extraction(df_combined)  # Step 3: Feature Extraction
    model_training_and_evaluation(df_features)  # Step 4: Model Training and Evaluation
