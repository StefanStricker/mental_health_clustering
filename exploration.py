import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("mh_data_cleaned.csv")

#Set plot style
sns.set_style("whitegrid")

#AGE DISTRIBUTION
plt.figure(figsize=(10, 5))
sns.histplot(df["age"], bins=20, kde=True, color="blue")
plt.title("Age Distribution of Respondents")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig("age.png")

#GENDER DISTRIBUTION
plt.figure(figsize=(8, 5))
sns.countplot(x="gender", data=df, order=df["gender"].value_counts().index, palette="coolwarm")
plt.title("Gender Distribution of Respondents")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.savefig("gender.png")

#REMOTE VS. ON-SITE WORK
plt.figure(figsize=(8, 5))
sns.countplot(x="remote", data=df, order=df["remote"].value_counts().index, palette="viridis")
plt.title("Remote vs. On-Site Work")
plt.xlabel("Work Setting")
plt.ylabel("Count")
plt.savefig("remote.png")

#MENTAL HEALTH BENEFITS AT WORK
plt.figure(figsize=(8, 5))
sns.countplot(x="mh_benefits", data=df, order=df["mh_benefits"].value_counts().index, palette="pastel")
plt.title("Does Your Employer Provide Mental Health Benefits?")
plt.xlabel("Response")
plt.ylabel("Count")
plt.savefig("health_benefits.png")

#COMFORT DISCUSSING MENTAL HEALTH WITH EMPLOYERS
plt.figure(figsize=(8, 5))
sns.countplot(x="mh_discussion_employer_consequences", data=df, 
              order=df["mh_discussion_employer_consequences"].value_counts().index, palette="cool")
plt.title("Do Employees Fear Consequences for Discussing Mental Health?")
plt.xlabel("Response")
plt.ylabel("Count")
plt.savefig("mh_discussion_employer.png")

#MENTAL HEALTH HISTORY IN FAMILY
plt.figure(figsize=(8, 5))
sns.countplot(x="mh_history_family", data=df, order=df["mh_history_family"].value_counts().index, palette="Reds")
plt.title("Do Employees Have a Family History of Mental Health Issues?")
plt.xlabel("Response")
plt.ylabel("Count")
plt.savefig("mh_history.png")

#CURRENT MENTAL HEALTH DISORDER
plt.figure(figsize=(8, 5))
sns.countplot(x="mh_disorder_current", data=df, order=df["mh_disorder_current"].value_counts().index, palette="Blues")
plt.title("Do You Currently Have a Mental Health Disorder?")
plt.xlabel("Response")
plt.ylabel("Count")
plt.savefig("mh_disorder.png")

#SEEKING TREATMENT FOR MENTAL HEALTH
plt.figure(figsize=(8, 5))
sns.countplot(x="mh_treatment", data=df, palette="Greens")
plt.title("Have You Ever Sought Treatment for a Mental Health Issue?")
plt.xlabel("Response (1 = Yes, 0 = No)")
plt.ylabel("Count")
plt.savefig("mh_treatment.png")

#IMPACT OF MENTAL HEALTH ON WORK PERFORMANCE
plt.figure(figsize=(8, 5))
sns.countplot(x="mh_treatment_work_interference", data=df, 
              order=df["mh_treatment_work_interference"].value_counts().index, palette="Oranges")
plt.title("Does Mental Health Affect Work Performance (If Treated)?")
plt.xlabel("Response")
plt.ylabel("Count")
plt.savefig("mh_impact.png")

#EMPLOYER AWARENESS OF MENTAL HEALTH ISSUES
plt.figure(figsize=(8, 5))
sns.countplot(x="mh_discussed", data=df, order=df["mh_discussed"].value_counts().index, palette="Purples")
plt.title("Has Your Employer Ever Discussed Mental Health?")
plt.xlabel("Response")
plt.ylabel("Count")
plt.savefig("mh_employer_awareness.png")


