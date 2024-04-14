import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from scipy.cluster import hierarchy

class HierarchicalClustering:
    def __init__(self, k):
        self.k = k

    def fit(self, data):
        self.data = data
        n = len(data)
        distances = np.zeros((n, n))

        # Calculate pairwise distances
        for i in range(n):
            for j in range(n):
                distances[i, j] = self.euclidean_distance(data[i], data[j])

        # Initialize clusters
        clusters = [[i] for i in range(n)]

        # Hierarchical clustering algorithm
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

    def euclidean_distance(self, x1, x2):
        x1 = np.array(x1, dtype=np.float64)  # Convert x1 to numpy array of numbers
        x2 = np.array(x2, dtype=np.float64)  # Convert x2 to numpy array of numbers
        if np.any(np.isnan(x1)) or np.any(np.isnan(x2)):
            return np.nan
        return np.sqrt(np.sum((x1 - x2) ** 2))

def import_data_from_excel(file_path):
    data = pd.read_excel(file_path)
    # Check if the first row or first column contains non-numeric values
    first_row_is_str = any(isinstance(val, str) for val in data.iloc[0])
    first_column_is_str = isinstance(data.iloc[:, 0].values[0], str)
    if first_row_is_str or first_column_is_str:
        data = data.iloc[1:, 1:]  # Exclude first row and first column if they contain non-numeric values
    return data

# Improved dendrogram visualization function
def plot_dendrogram(data):
    Z = hierarchy.linkage(data, method='ward')
    plt.figure(figsize=(10, 5))
    dn = hierarchy.dendrogram(Z)
    plt.title("Dendrogram")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    st.pyplot()

# Example usage
uploaded_file = st.file_uploader("Import Excel File", type=["xlsx", "xls"])
if uploaded_file is not None:
    data = import_data_from_excel(uploaded_file)
    clustering_algorithm = HierarchicalClustering(k=3)  # Initialize the clustering algorithm
    clustering_algorithm.fit(data.values)  # Fit the data
    labels = clustering_algorithm.labels_  # Get the cluster labels
    st.write(labels)  # Display the cluster labels
    plot_dendrogram(data.values)  # Visualize the dendrogram







