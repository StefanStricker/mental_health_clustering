import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv("mh_data_cleaned.csv")

#temp transform no_employees
enc = OrdinalEncoder()
df["num_employees"] = enc.fit_transform(df[["num_employees"]])

#transfrom yes/no into bool values
binary_cols = ["mh_neg_consequences", "mh_disorder_diagnosis_bool"]

for col in binary_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0})

#label encoding for: Yes, No, Maybe
ordinal_cols_maybe = ["mh_discussion_employer_consequences", "ph_discussion_employer_consequences", "mh_discussion_coworkers",
                      "mh_discussion_supervisors", "future_emp_ph_interview", "future_emp_mh_interview",
                      "mh_disorder_past", "mh_disorder_current"]

enc = OrdinalEncoder()
df[ordinal_cols_maybe] = enc.fit_transform(df[ordinal_cols_maybe])

#Change Not eligable for coverage to No
coverage_mapping = {"Yes": "Yes",
                    "No": "No",
                    "Not eligible for coverage / N/A": "No",
                    "I don't know": "I don't know"
                    }

df["mh_benefits"] = df["mh_benefits"].map(coverage_mapping)

#label encoding for: Yes, No, I don´t know
ordinal_cols_notknow = ["mh_benefits", "mh_discussed","mh_resources", "mh_anonymity", "mh_ph_equality", "mh_history_family",]
df[ordinal_cols_notknow] = enc.fit_transform(df[ordinal_cols_notknow])

#label encoding for features with unique answers
ordinal_cols_unique = ["mh_coverage", "mh_medical_leave", "prev_emp_mh_benefits", "prev_emp_mh_benefits_awareness", 
"prev_emp_mh_discussion", "prev_emp_mh_resources", "prev_emp_mh_anonymity", "prev_emp_mh_consequences",
"prev_emp_ph_consequences", "prev_emp_mh_discussion_coworkers", "prev_emp_mh_discussion_supervisor",
"prev_emp_mh_ph_equality", "prev_emp_mh_neg_consequences", "mh_issue_career", "mh_issue_coworker_view",
"mh_share_family", "mh_obs_unsupportive_workplace", "mh_treatment_work_interference", "mh_no_treatment_work_interference",
"remote", "gender"]

df[ordinal_cols_unique] = enc.fit_transform(df[ordinal_cols_unique])

# Role mapping dictionary
role_mapping = {
    "Back-end Developer": "Developer",
    "Front-end Developer": "Developer",
    "DevOps/SysAdmin": "Developer",
    "Data Scientist": "Developer",
    "One-person shop": "Developer",
    "Designer": "Developer",
    
    "Supervisor/Team Lead": "Leadership",
    "Executive Leadership": "Leadership",
    "Product Manager": "Leadership",
    "Dev Evangelist/Advocate": "Leadership",
    
    "Sales": "Sales",
    
    "Support": "Other",
    "Other": "Other"
}

# Function to map positions to categories
def map_position(roles):
    role_list = str(roles).split("|") 
    mapped_roles = {role_mapping[role] for role in role_list if role in role_mapping} 
    return "|".join(mapped_roles) if mapped_roles else "Other"

# Apply function to assign new roles
df["position_mapped"] = df["position"].apply(map_position)

# Define the new role categories
role_categories = ["Developer", "Leadership", "Sales", "Other"]

# Create binary columns
for role in role_categories:
    df[role] = df["position_mapped"].apply(lambda x: 1 if role in str(x) else 0)

# Drop original position columns
df.drop(columns=["position", "position_mapped"], inplace=True)


df.to_csv("mh_data_feature_eng.csv", index = False)
