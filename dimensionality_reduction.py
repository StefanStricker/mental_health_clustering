import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats


#Load dataset
df = pd.read_csv("mh_data_feature_eng.csv")

#Feature Variance
selector = VarianceThreshold(threshold=0.4)
Xs = selector.fit_transform(df)

print(selector.variances_)
#smallest variance ~1.27

#Correlation Matrix
corr_matrix_pearson = df.corr(method= "pearson")

#plt correlation Matrix pearsoon
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix_pearson, cmap="coolwarm", annot=False)
plt.title("Correlation Matrix Pearson")
plt.savefig("Correlation Matrix Pearson")

#country work and country live are highly correlated so country work gets dropped
df.drop("country_work", axis = 1, inplace = True)

#Correlation Matrix 
corr_matrix_spearman = df.corr(method= "spearman")

#plt correlation matrix spearman
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix_spearman, cmap="coolwarm", annot=False)
plt.title("Correlation Matrix spearman")
plt.savefig("Correlation Matrix spearman")

df.to_csv("kmean_ata.csv", index= False)