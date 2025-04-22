import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                            silhouette_score, calinski_harabasz_score, davies_bouldin_score,
                            rand_score, adjusted_rand_score, mutual_info_score, 
                            normalized_mutual_info_score)
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization parameters (modern compatible)
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.facecolor'] = 'white'

# Load and prepare data
data = pd.read_csv('dataset.csv')
data = data.set_index('Unnamed: 0')

# Visualization 1: Heatmap of original data 
plt.figure(figsize=(12, 10))
sns.heatmap(data.iloc[:, :100].corr(), cmap='coolwarm') # only did 100 here but will change soon
plt.title('Original Data Correlation Heatmap (First 100 Features)')
plt.savefig('original_heatmap.png', bbox_inches='tight')
plt.close()

# Extract cancer types from index
cancer_types = []
for idx in data.index:
    cancer_type = ''.join([c for c in idx if c.isalpha()]).lower()
    cancer_types.append(cancer_type)

labels = pd.Series(cancer_types, index=data.index)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Get indices for each cancer type
brca_indices = labels[labels == 'brca'].index
luad_indices = labels[labels == 'luad'].index
prad_indices = labels[labels == 'prad'].index

# Create balanced train-test split
train_indices = np.concatenate([brca_indices[:300], luad_indices[:300], prad_indices[:300]])
test_indices = np.concatenate([brca_indices[300:450], luad_indices[300:450], prad_indices[300:450]])

# Prepare full datasets
X_train_full = data.loc[train_indices]
X_test_full = data.loc[test_indices]
y_train = encoded_labels[np.isin(data.index, train_indices)]
y_test = encoded_labels[np.isin(data.index, test_indices)]

# Feature selection parameters
variance_features = 4800  # Top features by variance
mi_features = 900        # Top features by mutual information, added this for clarity

# Feature selection pipeline
print("Performing feature selection...")
variances = data.var()
top_variance_features = variances.nlargest(variance_features).index
X_train_var = X_train_full[top_variance_features]
X_test_var = X_test_full[top_variance_features]

mi_selector = SelectKBest(mutual_info_classif, k=mi_features)
X_train_selected = mi_selector.fit_transform(X_train_var, y_train)
X_test_selected = mi_selector.transform(X_test_var)

print(f"\nNumber of features after variance selection: {variance_features}")
print(f"Number of features after mutual information selection: {mi_features}")

# Visualization 2: Heatmap of selected features
plt.figure(figsize=(12, 10))
selected_df = pd.DataFrame(X_train_selected, 
                         columns=[f"MI_{i}" for i in range(X_train_selected.shape[1])])
sns.heatmap(selected_df.iloc[:, :100].corr(), cmap='coolwarm')
plt.title('Selected Features Correlation Heatmap (First 100 Features)')
plt.savefig('selected_heatmap.png', bbox_inches='tight')
plt.close()

# Create complete pipeline with scaling and KNN
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# Define parameter grid for GridSearchCV
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11, 15],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan', 'minkowski'],
    'knn__p': [1, 2]  # For minkowski metric
}

# Set up stratified k-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform grid search with cross-validation
print("\nPerforming grid search with cross-validation...")
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_selected, y_train)

# Best model evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_selected)

# Evaluation metrics
print("\n=== Best Model Evaluation ===")
print("Best Parameters:", grid_search.best_params_)
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png', bbox_inches='tight')
plt.close()

# Class-wise accuracy
print("\nClass-wise Accuracy:")
cm = confusion_matrix(y_test, y_pred)
for i, cancer in enumerate(label_encoder.classes_):
    class_total = sum(y_test == i)
    class_correct = cm[i, i]
    class_accuracy = (class_correct / class_total) * 100
    print(f"{cancer.upper()}: {class_accuracy:.2f}%")

# Cross-validation scores for best model
print("\nCross-validation scores for best model:")
cv_scores = cross_val_score(best_model, X_train_selected, y_train, cv=cv)
print(f"Scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")

# Visualization 3: Accuracy vs k values
results = pd.DataFrame(grid_search.cv_results_)
k_values = results['param_knn__n_neighbors'].unique()
k_values.sort()

plt.figure(figsize=(10, 6))
for metric in ['euclidean', 'manhattan']:
    subset = results[results['param_knn__metric'] == metric]
    plt.plot(subset['param_knn__n_neighbors'], 
             subset['mean_test_score'],
             marker='o',
             label=f"{metric} distance")

plt.title('KNN Performance by Number of Neighbors')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean CV Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('knn_performance.png', bbox_inches='tight')
plt.close()

# Clustering evaluation
print("\n=== Clustering Evaluation ===")
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X_test_selected)

metrics = {
    'Silhouette': silhouette_score(X_test_selected, cluster_labels),
    'CH': calinski_harabasz_score(X_test_selected, cluster_labels),
    'DBI': davies_bouldin_score(X_test_selected, cluster_labels),
    'RI': rand_score(y_test, cluster_labels),
    'ARI': adjusted_rand_score(y_test, cluster_labels),
    'MI': mutual_info_score(y_test, cluster_labels),
    'NMI': normalized_mutual_info_score(y_test, cluster_labels)
}

print("\nClustering Metrics:")
for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")

# Visualization 4: PCA comparison before/after feature selection
pca = PCA(n_components=2)

# Original data PCA
X_pca_original = pca.fit_transform(StandardScaler().fit_transform(X_train_full))
# Selected data PCA
X_pca_selected = pca.fit_transform(StandardScaler().fit_transform(X_train_selected))

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca_original[:, 0], y=X_pca_original[:, 1], hue=y_train)
plt.title(f'Original Data PCA\n({X_train_full.shape[1]} features)')

plt.subplot(1, 2, 2)
sns.scatterplot(x=X_pca_selected[:, 0], y=X_pca_selected[:, 1], hue=y_train)
plt.title(f'Selected Features PCA\n({X_train_selected.shape[1]} features)')

plt.tight_layout()
plt.savefig('pca_comparison.png', bbox_inches='tight')
plt.close()

# Visualization 5: Clustering results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca_selected[:, 0], y=X_pca_selected[:, 1], hue=y_test)
plt.title('True Cancer Type Distribution')

plt.subplot(1, 2, 2)
sns.scatterplot(x=X_pca_selected[:, 0], y=X_pca_selected[:, 1], hue=cluster_labels)
plt.title('KMeans Cluster Assignment')

plt.tight_layout()
plt.savefig('clustering_results.png', bbox_inches='tight')
plt.close()

print("\nAll visualizations saved as PNG files:")
print("- original_heatmap.png")
print("- selected_heatmap.png")
print("- confusion_matrix.png")
print("- knn_performance.png")
print("- pca_comparison.png")
print("- clustering_results.png")