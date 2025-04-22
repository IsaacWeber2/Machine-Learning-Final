import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import rand_score, adjusted_rand_score, mutual_info_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Function for plotting dimensionality reduction results
def plot_dr(X, y, title):
    plt.figure(figsize=(10, 8))

    for label in np.unique(y):
        plt.scatter(X[y == label, 0], X[y == label, 1], label=f"Class {label}", edgecolor='k', alpha=0.7)
    
    plt.title(title)
    plt.xlabel("Component #1")
    plt.ylabel("Component #2")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Loading data and extracting each type
data = pd.read_csv('dataset.csv')
data = data.set_index('Unnamed: 0')
cancer_types = []
for idx in data.index:
    cancer_type = ''.join([c for c in idx if c.isalpha()]).lower()
    cancer_types.append(cancer_type)

# 0 = BRCA
# 1 = LUAD
# 2 = PRAD
labels = pd.Series(cancer_types, index=data.index)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
brca_indices = labels[labels == 'brca'].index
luad_indices = labels[labels == 'luad'].index
prad_indices = labels[labels == 'prad'].index

#Starting with same amount of samples for each
# 300 for training and 150 for testing
brca_train = brca_indices[:300]
luad_train = luad_indices[:300]
prad_train = prad_indices[:300]

brca_test = brca_indices[300:450]
luad_test = luad_indices[300:450]
prad_test = prad_indices[300:450]

# Combine indices
train_indices = np.concatenate([brca_train, luad_train, prad_train])
test_indices = np.concatenate([brca_test, luad_test, prad_test])

# Starting with first 3000 features
reduced_data = data.iloc[:, :3000]

#Splitting data
X_train = reduced_data.loc[train_indices]
X_test = reduced_data.loc[test_indices]
y_train = encoded_labels[np.isin(data.index, train_indices)]
y_test = encoded_labels[np.isin(data.index, test_indices)]

# Feature scaling with standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# K-Neighbors Classification without dimensionality reduction
print("Performing 3 cases of KNN classification: no dimensionality reduction, PCA, and LDA...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

# K-Neighbors Classification with PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

knn.fit(X_train_pca, y_train)
y_pred_pca = knn.predict(X_test_pca)

# K-Neighbors Classification with LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_test_lda = lda.transform(X_test_scaled)

knn.fit(X_train_lda, y_train)
y_pred_lda = knn.predict(X_test_lda)

# Data visualization for dimensionality reduction
plot_dr(X_train_pca, y_train, "Dimensionality Reduction using PCA")
plot_dr(X_train_lda, y_train, "Dimensionality Reduction using LDA")

# Print accuracy
print("Printing accuracy...")
accuracy = accuracy_score(y_test, y_pred)
accuracy_pca = accuracy_score(y_test, y_pred_pca)
accuracy_lda = accuracy_score(y_test, y_pred_lda)
print(f"KNN Accuracy without dimensionality reduction: {accuracy * 100:.2f}%")
print(f"KNN Accuracy with PCA: {accuracy_pca * 100:.2f}%")
print(f"KNN Accuracy with LDA: {accuracy_lda * 100:.2f}%")
