import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_csv("/home/sera/.cache/kagglehub/datasets/osmi/mental-health-in-tech-2016/versions/2/mental-heath-in-tech-2016_20161114.csv")

#rename columns

renamed_columns = ["self_employed", "num_employees", "tech_comp", "tech_role", "mh_benefits", "mh_coverage", "mh_discussed", "mh_resources", 
"mh_anonymity", "mh_medical_leave", "mh_discussion_employer_consequences", "ph_discussion_employer_consequences", "mh_discussion_coworkers", 
"mh_discussion_supervisors", "mh_ph_equality", "mh_neg_consequences", "mh_med_cov", "mh_resources_knowledge", "mh_disorder_revealed_clients", 
"mh_revealed_neg_impact", "mh_disorder_revealed_coworkers", "mh_disorder_revealed_coworkers_impact", "mh_effect_productivity", "mh_effect_productivity_percentage", 
"prev_employers", "prev_emp_mh_benefits", "prev_emp_mh_benefits_awareness", "prev_emp_mh_discussion", "prev_emp_mh_resources", "prev_emp_mh_anonymity", 
"prev_emp_mh_consequences", "prev_emp_ph_consequences", "prev_emp_mh_discussion_coworkers","prev_emp_mh_discussion_supervisor","prev_emp_mh_ph_equality", 
"prev_emp_mh_neg_consequences", "future_emp_ph_interview", "future_emp_ph_interview_why", "future_emp_mh_interview", "future_emp_mh_interview_why", 
"mh_issue_career", "mh_issue_coworker_view", "mh_share_family", "mh_obs_unsupportive_workplace", "mh_obs_less_likely_to_reveal", "mh_history_family", 
"mh_disorder_past", "mh_disorder_current", "mh_disorder_condition", "mh_disorder_condition_maybe", "mh_disorder_diagnosis_bool", "mh_disorder_diagnosis", 
"mh_treatment", "mh_treatment_work_interference", "mh_no_treatment_work_interference", "age", "gender", "country_live", "us_state", "country_work", 
"us_state_work", "position", "remote"]

df.columns = renamed_columns

#cleaning gender columns since it has to many unique values
def clean_gender(gender):
    
    gender = str(gender).strip().lower()
    #Male variations
    male_variants = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "cis male"]
    
    #Female variations
    female_variants = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]
    
    #Non-binary / Other
    nonbinary_variants = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", 
                          "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", 
                          "female (trans)", "queer", "ostensibly male, unsure what that really means", "p", "a little about you"]

    #Map values
    if gender in male_variants:
        return 'Male'
    elif gender in female_variants:
        return 'Female'
    elif any(term in gender for term in nonbinary_variants):
        return 'Non-binary / Other'
    else:
        return 'Prefer not to say'

#Apply function to Gender column
df['gender'] = df['gender'].apply(clean_gender)

#create heatmap of missing values
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title("Missing Values Heatmap")
plt.savefig("missing_values.png")

#Drop all self employed data entries
df = df[df["self_employed"] != 1]  

#Drop columns with more than 50% missing values
threshold = 50  
df = df.dropna(thresh=len(df) * (threshold / 100), axis=1)

missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_summary = missing_percentage[missing_percentage > 0].sort_values(ascending=False)
#print(missing_summary)

#replace outliers in age with the median
median_age = df["age"].median()
df.loc[(df["age"] < 18) | (df["age"] > 100), "age"] = median_age

#replace null values in mh_coverage with mode
df["mh_coverage"].fillna(df["mh_coverage"].mode()[0], inplace=True)

#replace null values in mh_obs_unsupportive_workplace with mode
df["mh_obs_unsupportive_workplace"].fillna(df["mh_obs_unsupportive_workplace"].mode()[0], inplace=True)

#replace null values for previous employer columns to "not applicable"
for col in df.filter(like="prev_emp_").columns:
    df[col].fillna("Not Applicable", inplace=True)

#drop columns with to long open ended question with many null values and self employed
df.drop(columns=["future_emp_ph_interview_why", "future_emp_mh_interview_why", "self_employed"], inplace=True)

#fill null values for people not living in the us
df["us_state"].fillna("Not in the US", inplace=True)
df["us_state_work"].fillna("Not in the US", inplace=True)

df.to_csv("mh_data_cleaned.csv", index = False)
