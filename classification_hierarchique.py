import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import pairwise_distances

class HierarchicalClustering:
    def __init__(self, k):
        self.k = k

    def fit(self, data, distances):
        n = len(data)
        clusters = [[i] for i in range(n)]

        while len(clusters) > self.k:
            min_dist = np.inf
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    cluster1 = clusters[i]
                    cluster2 = clusters[j]
                    avg_dist = 0
                    count = 0
                    for idx1 in cluster1:
                        for idx2 in cluster2:
                            if not np.isnan(distances[idx1, idx2]):
                                avg_dist += distances[idx1, idx2]
                                count += 1
                    if count > 0:
                        avg_dist /= count
                        if avg_dist < min_dist:
                            min_dist = avg_dist
                            merge_index = (i, j)
            i, j = merge_index
            new_cluster = clusters[i] + clusters[j]
            clusters[i] = new_cluster
            clusters.pop(j)

        self.labels_ = np.zeros(n)
        for i, cluster in enumerate(clusters):
            for idx in cluster:
                self.labels_[idx] = i

def import_data_from_excel(file_path):
    data = pd.read_excel(file_path)
    # Check if the first row or first column contains non-numeric values
    first_row_is_str = any(isinstance(val, str) for val in data.iloc[0])
    first_column_is_str = isinstance(data.iloc[:, 0].values[0], str)
    if first_row_is_str or first_column_is_str:
        data = data.iloc[1:, 1:]  # Exclude first row and first column if they contain non-numeric values
    return data

def dendrogram_visualization(data, labels):
    X = data.values
    Z = linkage(X, method='ward')
    fig, ax = plt.subplots(figsize=(10, 5))
    dn = dendrogram(Z, ax=ax)
    ax.set_title("Dendrogram")
    ax.set_xlabel("Data Points")
    ax.set_ylabel("Distance")
    st.pyplot(fig)  # Display the plot in Streamlit

# Example usage:
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])
if uploaded_file is not None:
    data = import_data_from_excel(uploaded_file)
    k = 3  # Number of clusters
    distances = pairwise_distances(data.values, metric='euclidean')
    clustering_algorithm = HierarchicalClustering(k=k)
    clustering_algorithm.fit(data, distances)
    labels = clustering_algorithm.labels_
    dendrogram_visualization(data, labels)








