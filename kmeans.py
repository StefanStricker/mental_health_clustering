import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import matplotlib as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns


df = pd.read_csv("kmean_ata.csv")

# create a k-Means model an Elbow-Visualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,8), \
timings=False)
# fit the visualizer and show the plot
visualizer.fit(df)
visualizer.show(outpath="elbow_method_plot.png")  # Saves the plot as an image


#Clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(df)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)
