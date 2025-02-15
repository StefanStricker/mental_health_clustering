import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer 
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import silhouette_score
import matplotlib as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("kmean_ata.csv")

# create Elbow-Visualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,8), \
timings=False)
# fit the visualizer and show the plot
visualizer.fit(df)
visualizer.show(outpath="elbow_method_plot.png")


#Clustering

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

n_clusters = 3

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

centers = kmeans.cluster_centers_

lab = kmeans.labels_

S = silhouette_score(X_scaled, lab)
print(S)


