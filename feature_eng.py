import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv("mh_data_cleaned.csv")

#transfrom yes/no into bool values
binary_cols = ["mh_neg_consequences", "mh_disorder_diagnosis_bool"]

for col in binary_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0})


ordinal_cols = ["mh_discussed", "mh_resources", "mh_anonymity", "mh_ph_equality"]

encoder = OrdinalEncoder()
df[ordinal_cols] = encoder.fit_transform(df[ordinal_cols])

#print(df.info())
#print(df.describe())
#print(df["mh_discussed"].tolist())

