import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer 
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_pca_correlation_graph

df = pd.read_csv("kmean_ata.csv")

#normalizing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

#PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(pca.explained_variance_ratio_)

#elbow method
kmeans_pca_el = KMeans()
visualizer_elbow = KElbowVisualizer(kmeans_pca_el, k=(1,8), timings=False)
visualizer_elbow.fit(X_pca)
visualizer_elbow.show(outpath= "elbow.png")

#Run K-Means 
kmeans_pca = KMeans(n_clusters=3, random_state=42, n_init=10)

#create pca dataframe
df_pca = pd.DataFrame(X_pca, columns=["PCA_1", "PCA_2"])
df_pca["cluster"] = kmeans_pca.fit_predict(X_pca)

labels_pca = df_pca["cluster"]

#Check silhouette score
S_pca = (silhouette_score(X_pca, labels_pca))
print(S_pca)

#silhouette plot
visualizer = SilhouetteVisualizer(kmeans_pca, colors="yellowbrick")
visualizer.fit(X_pca)
visualizer.show(outpath = "silhouette_plot.png")

#calculate mean values for each cluster
cluster_means = df_pca.groupby("cluster").mean()
print(cluster_means)

#plot Kmeans clustering
plt.figure(figsize=(8,6))
sns.scatterplot(x=df_pca["PCA_1"], y=df_pca["PCA_2"], hue=df_pca["cluster"], palette="Set1")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-Means Clustering")
plt.legend(title="Cluster")
plt.savefig("Kmeans_clustering")

#plt cluster %
cluster_count = df_pca["cluster"].value_counts()

plt.figure(figsize=(5,5))
plt.pie(cluster_count, labels = cluster_count.index, autopct="%1.1f%%")
plt.title("Cluster sizes")
plt.savefig("cluster_size.png")

#compute pca loadings
pca_loadings = pd.DataFrame(pca.components_.T, columns=["PCA_1", "PCA_2"], index=df.columns)

#extracting top contributing features from pca
top_features_pca1 = pca_loadings["PCA_1"].abs().sort_values(ascending=False).head(10)
top_features_pca2 = pca_loadings["PCA_2"].abs().sort_values(ascending=False).head(10)

# Combine for visualization
pca_features = pd.concat([top_features_pca1, top_features_pca2], axis=1)

# Plot the top contributing features
plt.figure(figsize=(10, 6))
pca_features.plot(kind="barh", figsize=(12, 8), width=0.4, color=["blue", "orange"])
plt.xlabel("Feature Contribution to PCA")
plt.ylabel("Features")
plt.title("Top 10 Features Contributing to PCA Components")
plt.legend(["PCA_1", "PCA_2"])
plt.savefig("pca_feature_importance.png")

