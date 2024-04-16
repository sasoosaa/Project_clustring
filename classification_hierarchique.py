import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram  # Modified import statement
import base64
import requests

# Set background image
background_image_url = "https://img.freepik.com/free-vector/ai-technology-brain-background-vector-digital-transformation-concept_53876-117820.jpg?w=900&t=st=1713012632~exp=1713013232~hmac=05da65e2d9e6da77202ded9006cfecb86c0c11fbc70346fb0532307b18d8ac3f"

def get_base64_of_bin_file(url):
    data = requests.get(url).content
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    st.markdown(
        f'<style>'
        f'.stApp {{'
        f'background-image: url("data:image/png;base64,{bin_str}");'
        f'background-size: cover;'
        f'}}'
        f'</style>',
        unsafe_allow_html=True
    )

set_png_as_page_bg(background_image_url)

class KMeansClustering:
    def __init__(self, k=3, distance_metric='euclidean', random_state=None):
        self.k = k
        self.distance_metric = distance_metric
        self.centroids = None
        self.random_state = random_state

    def calculate_distance(self, data_point, centroids):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((centroids - data_point) ** 2, axis=1))
        elif self.distance_metric == 'city_block':
            return np.sum(np.abs(centroids - data_point), axis=1)
        else:
            raise ValueError("Invalid distance metric. Supported metrics are 'euclidean' and 'city_block'.")

    def fit(self, X, max_iterations=200):
        np.random.seed(self.random_state)  # Set the random seed

        # Initialize centroids randomly
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]

        inertia_values = []

        for _ in range(max_iterations):
            # Assign each data point to the nearest centroid
            distances = np.vstack([self.calculate_distance(data_point, self.centroids) for data_point in X])
            cluster_assignment = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array([X[cluster_assignment == i].mean(axis=0) for i in range(self.k)])

            # Calculate inertia
            inertia = np.sum((X - new_centroids[cluster_assignment]) ** 2)
            inertia_values.append(inertia)

            # Check convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        # Calculate silhouette score
        labels = np.argmin(distances, axis=1)
        silhouette_avg = silhouette_score(X, labels)

        return cluster_assignment, self.centroids, inertia_values, silhouette_avg

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

def plot_dendrogram(data):
    Z = linkage(data, method='ward')  # Use linkage from scipy.cluster.hierarchy directly
    plt.figure(figsize=(10, 5))
    dn = dendrogram(Z)
    plt.title("Dendrogram")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    st.pyplot(plt.gcf())  # Pass the current figure as an argument to st.pyplot()

def plot_result(X_reduced, labels, centroids):
    if X_reduced.shape[1] == 2:
        df = pd.DataFrame(X_reduced, columns=['Component 1', 'Component 2'])
    else:
        df = pd.DataFrame(X_reduced[:, :2], columns=['Component 1', 'Component 2'])
    df['Cluster'] = labels.astype(str)  # Convert labels to string for coloring purposes

    fig = px.scatter(df, x='Component 1', y='Component 2', color='Cluster', 
                     title='Clustering Result', hover_name=df.index)
    fig.add_trace(go.Scatter(
        x=centroids[:, 0],
        y=centroids[:, 1],
        mode='markers',
        marker=dict(size=10, color='black', symbol='star'),
        name='Centroids'
    ))
    return fig

def plot_convergence(inertia_values):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(inertia_values) + 1)),
        y=inertia_values,
        mode='lines+markers',
        marker=dict(color='blue'),
        name='Inertia'
    ))
    fig.update_layout(
        title='Convergence Plot',
        xaxis=dict(title='Iteration'),
        yaxis=dict(title='Inertia'),
        hovermode='closest'
    )
    return fig

def plot_silhouette_score(silhouette_values):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(2, 12)),
        y=silhouette_values,
        mode='lines+markers',
        marker=dict(color='green'),
        name='Silhouette Score'
    ))
    fig.update_layout(
        title='Silhouette Score',
        xaxis=dict(title='Number of Clusters (K)'),
        yaxis=dict(title='Silhouette Score'),
        hovermode='closest'
    )
    return fig

def main():
    st.title("Clustering")

    selected = st.radio("Select Option", ["Home", "KMeans Clustering", "Hierarchical Clustering"])

    if selected == "KMeans Clustering":
        st.title("KMeans Clustering")
        uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
        if uploaded_file is not None:
            data = pd.read_excel(uploaded_file, index_col=0)
            st.write("Data imported successfully!")

            X = data.values
            X = X[:, 2:]  # Adjust this according to your data

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            k = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=10, value=3)
            distance_metric = st.sidebar.radio("Distance Metric", ["Euclidean", "City Block"], index=0)
            distance_metric = 'euclidean' if distance_metric == 'Euclidean' else 'city_block'
            
            use_pca = st.sidebar.checkbox("Use PCA")

            labels, centroids, inertia_values, silhouette_avg = None, None, None, None  # Initialize variables
            
            if st.sidebar.button("Cluster"):
                kmeans = KMeansClustering(k=k, distance_metric=distance_metric, random_state=42)
                labels, centroids, inertia_values, silhouette_avg = kmeans.fit(X_scaled)
                
                st.success(f"Convergence achieved in {len(inertia_values)} iterations.")
                st.write(f"Number of clusters: {k}")
                st.write(f"Number of data points: {len(X)}")
                st.write(f"Number of features: {X.shape[1]}")
                st.write(f"Silhouette Score: {silhouette_avg:.4f}")


            plot_choice = st.sidebar.selectbox("Select Plot", ["Clustering Result", "Convergence Plot", "Silhouette Score"])

            if plot_choice == "Clustering Result" and labels is not None:
                fig = plot_result(X_scaled, labels, centroids)
                st.plotly_chart(fig, use_container_width=True)
            elif plot_choice == "Convergence Plot" and inertia_values is not None:
                fig = plot_convergence(inertia_values)
                st.plotly_chart(fig, use_container_width=True)
            elif plot_choice == "Silhouette Score" and silhouette_avg is not None:
                silhouette_values = []
                for k in range(2, 12):
                    kmeans = KMeansClustering(k=k, random_state=42)
                    _, _, _, silhouette_avg = kmeans.fit(X_scaled)
                    silhouette_values.append(silhouette_avg)
                fig = plot_silhouette_score(silhouette_values)
                st.plotly_chart(fig, use_container_width=True)

    elif selected == "Hierarchical Clustering":
        st.title("Hierarchical Clustering")
        uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])
        if uploaded_file is not None:
            data = pd.read_excel(uploaded_file, index_col=0)
            clustering_algorithm = HierarchicalClustering(k=3)  # Initialize the clustering algorithm
            clustering_algorithm.fit(data.values)  # Fit the data
            labels = clustering_algorithm.labels_  # Get the cluster labels
            st.write(labels)  # Display the cluster labels
            plot_dendrogram(data.values)  # Visualize the dendrogram

if __name__ == "__main__":
    st.set_option('deprecation.showPyplotGlobalUse', False)  # Disable the warning about st.pyplot() usage
    main()

    clustering_algorithm.fit(data.values)  # Fit the data
    labels = clustering_algorithm.labels_  # Get the cluster labels
    st.write(labels)  # Display the cluster labels
    plot_dendrogram(data.values)  # Visualize the dendrogram







