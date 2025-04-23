import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Load and preprocess data
data = pd.read_csv('dataset.csv')
data = data.set_index('Unnamed: 0')

# Extract and encode labels
cancer_types = [''.join([c for c in idx if c.isalpha()]).lower() for idx in data.index]
labels = pd.Series(cancer_types, index=data.index)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Get indices by cancer type
brca_indices = labels[labels == 'brca'].index
luad_indices = labels[labels == 'luad'].index
prad_indices = labels[labels == 'prad'].index

# Train/test split
brca_train = brca_indices[:300]
luad_train = luad_indices[:300]
prad_train = prad_indices[:300]
brca_test = brca_indices[300:450]
luad_test = luad_indices[300:450]
prad_test = prad_indices[300:450]

train_indices = np.concatenate([brca_train, luad_train, prad_train])
test_indices = np.concatenate([brca_test, luad_test, prad_test])

# Use first 3000 features
reduced_data = data.iloc[:, :3000]
X_train = reduced_data.loc[train_indices]
y_train = encoded_labels[np.isin(data.index, train_indices)]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Grid search over k
param_grid = {'n_neighbors': list(range(1, 21))}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Extract results
k_values = param_grid['n_neighbors']
mean_test_scores = grid_search.cv_results_['mean_test_score']

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(k_values, mean_test_scores, marker='o', linestyle='-', color='blue')
plt.title('KNN Accuracy vs. Number of Neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validated Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.tight_layout()
plt.show()
