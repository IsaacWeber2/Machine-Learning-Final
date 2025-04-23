import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             silhouette_score, calinski_harabasz_score, davies_bouldin_score,
                             rand_score, adjusted_rand_score, mutual_info_score, 
                             normalized_mutual_info_score)
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

# Load data
data = pd.read_csv('dataset.csv')
data = data.set_index('Unnamed: 0')

# Heatmap of original data
plt.figure(figsize=(12, 10))
sns.heatmap(data.iloc[:, :100].corr(), cmap='coolwarm')
plt.title('Original Data Correlation Heatmap')
plt.savefig('original_heatmap.png', bbox_inches='tight')
plt.close()

# Extract cancer types
cancer_types = [''.join([c for c in idx if c.isalpha()]).lower() for idx in data.index]
labels = pd.Series(cancer_types, index=data.index)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split train/test
brca_indices = labels[labels == 'brca'].index
luad_indices = labels[labels == 'luad'].index
prad_indices = labels[labels == 'prad'].index
train_indices = np.concatenate([brca_indices[:300], luad_indices[:300], prad_indices[:300]])
test_indices = np.concatenate([brca_indices[300:450], luad_indices[300:450], prad_indices[300:450]])

X_train_full = data.loc[train_indices]
X_test_full = data.loc[test_indices]
y_train = encoded_labels[np.isin(data.index, train_indices)]
y_test = encoded_labels[np.isin(data.index, test_indices)]

# Variance selection
features = 5000
top_variance_features = data.var().nlargest(features).index
X_train_var = X_train_full[top_variance_features]
X_test_var = X_test_full[top_variance_features]

# Mutual information selection
features2 = 900
mi_selector = SelectKBest(mutual_info_classif, k=features2)
X_train_selected = mi_selector.fit_transform(X_train_var, y_train)
X_test_selected = mi_selector.transform(X_test_var)

# Heatmap of selected features
plt.figure(figsize=(12, 10))
selected_df = pd.DataFrame(X_train_selected, columns=[f"MI_{i}" for i in range(X_train_selected.shape[1])])
sns.heatmap(selected_df.iloc[:, :100].corr(), cmap='coolwarm')
plt.title('Selected Features Correlation Heatmap')
plt.savefig('selected_heatmap.png', bbox_inches='tight')
plt.close()

# KNN pipeline and grid search
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11, 15],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan', 'minkowski', 'cosine'],
    'knn__p': [1, 2]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_sea = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
grid_sea.fit(X_train_selected, y_train)
best_model = grid_sea.best_estimator_
y_pred = best_model.predict(X_test_selected)

# Evaluation
print("Best Parameters:", grid_sea.best_params_)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png', bbox_inches='tight')
plt.close()

# Clustering evaluation
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
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

# Static 2D PCA comparison
pca_2d = PCA(n_components=2)
X_pca_original_2d = pca_2d.fit_transform(StandardScaler().fit_transform(X_train_full))
X_pca_selected_2d = pca_2d.fit_transform(StandardScaler().fit_transform(X_train_selected))
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca_original_2d[:, 0], y=X_pca_original_2d[:, 1], hue=y_train, palette='viridis')
plt.title(f'Original Data PCA (2D)')
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_pca_selected_2d[:, 0], y=X_pca_selected_2d[:, 1], hue=y_train, palette='viridis')
plt.title(f'Selected Features PCA (2D)')
plt.tight_layout()
plt.savefig('pca_comparison_2d.png', bbox_inches='tight')
plt.close()

# 2D PCA on test set for interactive plots
X_pca_test_2d = pca_2d.fit_transform(StandardScaler().fit_transform(X_test_selected))

# Interactive Plotly 2D PCA: True Labels
fig2d_true = px.scatter(
    x=X_pca_test_2d[:, 0], y=X_pca_test_2d[:, 1],
    color=[label_encoder.classes_[i] for i in y_test],
    title='Interactive 2D PCA: True Cancer Types',
    labels={'x': 'PC1', 'y': 'PC2'}
)
fig2d_true.write_html('interactive_2d_pca_true.html', include_plotlyjs='cdn')
fig2d_true.show()

# Interactive Plotly 2D PCA: Predicted Labels
fig2d_pred = px.scatter(
    x=X_pca_test_2d[:, 0], y=X_pca_test_2d[:, 1],
    color=[label_encoder.classes_[i] for i in y_pred],
    title='Interactive 2D PCA: Predicted Cancer Types',
    labels={'x': 'PC1', 'y': 'PC2'}
)
fig2d_pred.write_html('interactive_2d_pca_pred.html', include_plotlyjs='cdn')
fig2d_pred.show()

# Interactive Plotly 3D PCA: True Labels
X_pca_test_3d = PCA(n_components=3).fit_transform(StandardScaler().fit_transform(X_test_selected))
fig3d_true = px.scatter_3d(
    x=X_pca_test_3d[:, 0], y=X_pca_test_3d[:, 1], z=X_pca_test_3d[:, 2],
    color=[label_encoder.classes_[i] for i in y_test],
    title='Interactive 3D PCA: True Cancer Types',
    labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'}
)
fig3d_true.write_html('interactive_3d_pca_true.html', include_plotlyjs='cdn')
fig3d_true.show()

# Interactive Plotly 3D PCA: Predicted Labels
fig3d_pred = px.scatter_3d(
    x=X_pca_test_3d[:, 0], y=X_pca_test_3d[:, 1], z=X_pca_test_3d[:, 2],
    color=[label_encoder.classes_[i] for i in y_pred],
    title='Interactive 3D PCA: Predicted Cancer Types',
    labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'}
)
fig3d_pred.write_html('interactive_3d_pca_pred.html', include_plotlyjs='cdn')
fig3d_pred.show()

print("HTML files saved:")
print("- interactive_2d_pca_true.html")
print("- interactive_2d_pca_pred.html")
print("- interactive_3d_pca_true.html")
print("- interactive_3d_pca_pred.html")
