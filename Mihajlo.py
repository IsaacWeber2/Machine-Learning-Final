import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)

data = pd.read_csv("dataset.csv")
data = data.rename(columns={"Unnamed: 0": "sample_id"})
data.set_index("sample_id", inplace=True)

cancer_labels = data.index.to_series().str.extract(r'([a-zA-Z]+)')[0].str.lower()
encoder = LabelEncoder()
y = encoder.fit_transform(cancer_labels)

X = data.copy()
X = X.loc[:, X.nunique() > 1] 

sample_counts = cancer_labels.value_counts()
min_class_size = sample_counts.min()

train_idx, test_idx = [], []
for label in sample_counts.index:
    indices = cancer_labels[cancer_labels == label].index
    train_idx += list(indices[:min_class_size//2])
    test_idx += list(indices[min_class_size//2:min_class_size])

X_train, X_test = X.loc[train_idx], X.loc[test_idx]
y_train, y_test = y[np.isin(data.index, train_idx)], y[np.isin(data.index, test_idx)]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

top_k_features = 10
mi_selector = SelectKBest(score_func=mutual_info_classif, k=top_k_features)
X_train_fs = mi_selector.fit_transform(X_train_scaled, y_train)
X_test_fs = mi_selector.transform(X_test_scaled)

pca = PCA(n_components=2)
lda = LDA(n_components=2)

X_train_pca = pca.fit_transform(X_train_fs)
X_test_pca = pca.transform(X_test_fs)

X_train_lda = lda.fit_transform(X_train_fs, y_train)
X_test_lda = lda.transform(X_test_fs)

def plot_projection(X_proj, y_labels, title, filename):
    plt.figure(figsize=(8, 6))
    for label in np.unique(y_labels):
        plt.scatter(X_proj[y_labels == label, 0],
                    X_proj[y_labels == label, 1],
                    label=encoder.classes_[label],
                    alpha=0.6, edgecolor='k')
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_projection(X_train_pca, y_train, "PCA Projection (Train)", "pca_scatter.png")
plot_projection(X_train_lda, y_train, "LDA Projection (Train)", "lda_scatter.png")

def evaluate_knn(X_tr, X_te, y_tr, y_te, desc):
    knn = KNeighborsClassifier()
    params = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    grid = GridSearchCV(knn, params, cv=StratifiedKFold(n_splits=3), scoring='accuracy')
    grid.fit(X_tr, y_tr)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_te)

    print(f"\nEvaluation: {desc}")
    print(f"Best Parameters: {grid.best_params_}")
    print(f"Test Accuracy: {accuracy_score(y_te, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_te, y_pred, target_names=encoder.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_te, y_pred))

# Evaluate all three versions
evaluate_knn(X_train_fs, X_test_fs, y_train, y_test, "Original KNN (Post-Feature Selection)")
evaluate_knn(X_train_pca, X_test_pca, y_train, y_test, "KNN + PCA")
evaluate_knn(X_train_lda, X_test_lda, y_train, y_test, "KNN + LDA")
