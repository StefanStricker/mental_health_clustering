import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer 
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
from sklearn import mixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_pca_correlation_graph


df = pd.read_csv("kmean_ata.csv")

#Create Elbow-Visualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,8), timings=False)
visualizer.fit(df)
visualizer.show(outpath="elbow_method_plot.png")


#Clustering

#normalizing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

#KMean
#n_clusters = 3
#kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
#df["cluster"] = kmeans.fit_predict(X_scaled)
#centers = kmeans.cluster_centers_
#lab = kmeans.labels_
#S = silhouette_score(X_scaled, lab)
#print(S)

#GMM
#gmm = mixture.GaussianMixture(n_components=3, random_state=42)
#gmm.fit(X_scaled)
#lab_gmm = gmm.predict(X_scaled)
#S_gmm = silhouette_score(X_scaled, lab_gmm)
#print(S_gmm)

#Agglomerative Clustering
#agg = AgglomerativeClustering(n_clusters=3)
#lab_agg = agg.fit_predict(X_scaled)
#S_agg = silhouette_score(X_scaled, lab_agg)
#print(S_agg)

#PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Run K-Means again
kmeans_pca = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans_pca.fit_predict(X_pca)
labels_pca = df["cluster"]

# Check silhouette score
S_pca = (silhouette_score(X_pca, labels_pca))
#print(S_pca)

#pca dataframe
df_pca = pd.DataFrame(X_pca, columns=["PCA_1", "PCA_2"])
df_pca["Cluster"] = df["cluster"]

visualizer = SilhouetteVisualizer(kmeans_pca, colors='yellowbrick')
visualizer.fit(X_pca)
visualizer.show(outpath = "silhouette_plot.png")

#MDS
#mds = MDS(n_components=2, random_state=42, dissimilarity="euclidean")
#X_mds = mds.fit_transform(X_scaled)

# Run K-Means again
#kmeans_mds = KMeans(n_clusters=3, random_state=42, n_init=10)
#labels_mds = kmeans_mds.fit_predict(X_mds)

#Check silhouette score again
#S_mds = (silhouette_score(X_pca, labels_mds))
#print(S_mds)


#Correlation Graph
feature_names = df.drop(columns=["cluster"]).columns.tolist()

#Plot PCA Correlation Graph
fig, cor_mat = plot_pca_correlation_graph(X_scaled, 
                                          feature_names, 
                                          dimensions=(1, 2), 
                                          figure_axis_size=10)

#print(cor_mat)

#plot Kmeans clustering
plt.figure(figsize=(8,6))
sns.scatterplot(x=df_pca["PCA_1"], y=df_pca["PCA_2"], hue=df_pca["Cluster"], palette="Set1")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-Means Clustering")
plt.legend(title="Cluster")
plt.savefig("Kmeans_clustering")


#plt cluster %
cluster_count = df["cluster"].value_counts()

plt.figure(figsize=(5,5))
plt.pie(cluster_count, labels = cluster_count.index, autopct="%1.1f%%")
plt.title("Cluster sizes")
plt.savefig("cluster_size.png")

# Get PCA loadings (how much each original feature contributes to the PCA components)
pca_loadings = pd.DataFrame(pca.components_.T, columns=["PCA_1", "PCA_2"], index=df.drop(columns=["cluster"]).columns)

# Sort features by importance for each PCA component
top_features_pca1 = pca_loadings["PCA_1"].abs().sort_values(ascending=False).head(10)
top_features_pca2 = pca_loadings["PCA_2"].abs().sort_values(ascending=False).head(10)

print("Top Features Contributing to PCA Component 1:")
print(top_features_pca1)

print("Top Features Contributing to PCA Component 2:")
print(top_features_pca2)

# Select top contributing features from PCA loadings
top_features = top_features_pca1.index.tolist() + top_features_pca2.index.tolist()

# Visualize feature distributions across clusters
plt.figure(figsize=(12, 6))
for feature in top_features[:5]:  # Only plot first 5 features for readability
    sns.boxplot(x=df["cluster"], y=df[feature])
    plt.title(f"Distribution of {feature} Across Clusters")
    plt.savefig("feature_dist.png")

