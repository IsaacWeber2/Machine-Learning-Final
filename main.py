import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import rand_score, adjusted_rand_score, mutual_info_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif

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


X_train_full = data.loc[train_indices]
X_test_full = data.loc[test_indices]
y_train = encoded_labels[np.isin(data.index, train_indices)]
y_test = encoded_labels[np.isin(data.index, test_indices)]


#using variance to get best list
features = 5000  # Number of features to select based on variance
variances = data.var()
top_variance_features = variances.nlargest(features).index
X_train_var = X_train_full[top_variance_features]
X_test_var = X_test_full[top_variance_features]

#using mutail information to select features
features2 = 900  # Number of features to select based on mutual information
mi_selector = SelectKBest(mutual_info_classif, k=features2)
X_train_selected = mi_selector.fit_transform(X_train_var, y_train)
X_test_selected = mi_selector.transform(X_test_var)

# Printing both features used
print(f"Number of features after variance selection: {features}")
print(f"Number of features after mutual information selection: {features2}")

# Scale the selected features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

# percent correct
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall KNN Accuracy: {accuracy * 100:.2f}%")