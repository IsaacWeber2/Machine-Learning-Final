import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import rand_score, adjusted_rand_score, mutual_info_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, SelectKBest


data = pd.read_csv('dataset.csv')
data = data.set_index('Unnamed: 0')


cancer_types = []
for idx in data.index:
    cancer_type = ''.join([c for c in idx if c.isalpha()]).lower()
    cancer_types.append(cancer_type)


labels = pd.Series(cancer_types, index=data.index)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
brca_indices = labels[labels == 'brca'].index
luad_indices = labels[labels == 'luad'].index
prad_indices = labels[labels == 'prad'].index

# Select training and testing samples
train_indices = np.concatenate([brca_indices[:300], luad_indices[:300], prad_indices[:300]])
test_indices = np.concatenate([brca_indices[300:450], luad_indices[300:450], prad_indices[300:450]])

#using all features for selection 
X_train_full = data.loc[train_indices]
X_test_full = data.loc[test_indices]
y_train = encoded_labels[np.isin(data.index, train_indices)]
y_test = encoded_labels[np.isin(data.index, test_indices)]

#top features based on variance
features = 4800  # Number of features to select based on variance

variances = data.var()
top_variance_features = variances.nlargest(features).index
X_train_var = X_train_full[top_variance_features]
X_test_var = X_test_full[top_variance_features]

#using mutual information to select features
features2 = 900  # Number of features to select based on mutual information
mi_selector = SelectKBest(mutual_info_classif, k=features2)
X_train_selected = mi_selector.fit_transform(X_train_var, y_train)
X_test_selected = mi_selector.transform(X_test_var)

#printing both features used 
print(f"Number of features after variance selection: {features}")
print(f"Number of features after mutual information selection: {features2}")


# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Accuracy: {accuracy * 100:.2f}%")


cm = confusion_matrix(y_test, y_pred)
class_names = label_encoder.classes_

for i, cancer in enumerate(class_names):
    class_total = sum(y_test == i)
    class_correct = cm[i, i]
    class_accuracy = (class_correct / class_total) * 100
    print(f"{cancer.upper()}: {class_accuracy:.2f}%")

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X_test_scaled)

# Clustering metrics
metrics = {
    'Silhouette': silhouette_score(X_test_scaled, cluster_labels),
    'CH': calinski_harabasz_score(X_test_scaled, cluster_labels),
    'DBI': davies_bouldin_score(X_test_scaled, cluster_labels),
    'RI': rand_score(y_test, cluster_labels),
    'ARI': adjusted_rand_score(y_test, cluster_labels),
    'MI': mutual_info_score(y_test, cluster_labels),
    'NMI': normalized_mutual_info_score(y_test, cluster_labels)
}

print("\nClustering Metrics:")
for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")