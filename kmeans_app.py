import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.graph_objs as go
import plotly.express as px

# Define custom theme colors
PRIMARY_COLOR = "#1E90FF"
BACKGROUND_COLOR = "#F0F2F6"
FONT_COLOR = "#212529"

# Apply custom theme
st.set_page_config(page_title="KMeans Clustering", layout="wide", page_icon=":chart_with_upwards_trend:")
st.markdown(f"""
    <style>
        .reportview-container .main .block-container{{
            max-width: 1600px;
            padding-top: 2rem;
            padding-right: 2rem;
            padding-left: 2rem;
            padding-bottom: 2rem;
        }}
        .reportview-container .main{{
            color: {FONT_COLOR};
            background-color: {BACKGROUND_COLOR};
        }}
        .reportview-container .main .block-container .stButton>button{{
            background-color: {PRIMARY_COLOR};
            color: white;
        }}
        .reportview-container .main .block-container .stButton>button:hover{{
            background-color: #4169E1;
            color: white;
        }}
        .reportview-container .main .block-container .stSlider>div>div{{
            background-color: {PRIMARY_COLOR};
        }}
        .reportview-container .main .block-container .stSlider>div>div>div{{
            background-color: white;
        }}
        .reportview-container .main .block-container .stSlider>div>div>div>div{{
            background-color: white;
        }}
        .css-1v3fvcr{{
            color: {FONT_COLOR};
        }}
    </style>
""", unsafe_allow_html=True)

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

def run_clustering(X, k, distance_metric, use_pca):
    if use_pca:
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)

    kmeans = KMeansClustering(k=k, distance_metric=distance_metric, random_state=42)
    labels, centroids, inertia_values, silhouette_avg = kmeans.fit(X)
    return labels, centroids, inertia_values, silhouette_avg

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

def plot_elbow_method(X_scaled):
    inertia_values = []
    for k in range(1, 11):
        kmeans = KMeansClustering(k=k, random_state=42)
        _, _, inertia, _ = kmeans.fit(X_scaled)
        inertia_values.append(inertia[-1])  # Take the inertia of the last iteration

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, 11)),
        y=inertia_values,
        mode='lines+markers',
        marker=dict(color='red'),
        name='Inertia'
    ))
    fig.update_layout(
        title='Elbow Method',
        xaxis=dict(title='Number of Clusters (K)'),
        yaxis=dict(title='Inertia'),
        hovermode='closest'
    )
    return fig

def main():
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
            labels, centroids, inertia_values, silhouette_avg = run_clustering(X_scaled, k, distance_metric, use_pca)

            st.success(f"Convergence achieved in {len(inertia_values)} iterations.")
            st.write(f"Number of clusters: {k}")
            st.write(f"Number of data points: {len(X)}")
            st.write(f"Number of features: {X.shape[1]}")
            st.write(f"Silhouette Score: {silhouette_avg:.4f}")

        plot_choice = st.sidebar.selectbox("Select Plot", ["Clustering Result", "Convergence Plot", "Silhouette Score", "Elbow Method"])

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
        elif plot_choice == "Elbow Method":
            silhouette_values = []
            for k in range(2, 12):
                kmeans = KMeansClustering(k=k, random_state=42)
                _, _, _, silhouette_avg = kmeans.fit(X_scaled)
                silhouette_values.append(silhouette_avg)
            fig = plot_silhouette_score(silhouette_values)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
