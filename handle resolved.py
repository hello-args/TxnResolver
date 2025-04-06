import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
import warnings
import ast
from pandas import json_normalize

warnings.filterwarnings("ignore")

# Load Excel file
df = pd.read_excel("resolved.xlsx")

# Drop unused columns
drop_cols = ['status', 'summary', 'next_step', 'Comments']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# --- Handle recon_sub_status (parse if dictionary-like) ---
def try_parse_dict(val):
    try:
        return ast.literal_eval(val)
    except:
        return None

if 'recon_sub_status' in df.columns:
    parsed = df['recon_sub_status'].astype(str).apply(try_parse_dict)
    mask_valid = parsed.apply(lambda x: isinstance(x, dict))
    parsed_dicts = parsed[mask_valid]

    if not parsed_dicts.empty:
        parsed_df = json_normalize(parsed_dicts)
        parsed_df.columns = [f"recon_sub_status_{col}" for col in parsed_df.columns]

        df = df.drop(columns=['recon_sub_status'])
        df = pd.concat([df.reset_index(drop=True), parsed_df.reset_index(drop=True)], axis=1)
    else:
        df['recon_sub_status'] = df['recon_sub_status'].astype(str)

# --- Convert date columns to numeric features ---
date_cols = ['sys_a_date', 'sys_b_date']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[f'{col}_day'] = df[col].dt.day
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_year'] = df[col].dt.year
        df = df.drop(columns=col)

# --- Fill missing numeric values with -1 ---
df = df.fillna(-1)

# --- Encode categorical columns ---
cat_cols = df.select_dtypes(include='object').columns
label_encoders = {}
for col in cat_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Add target column (1 for resolved)
df['resolved'] = 1

# Separate features and target
X = df.drop(columns=['transaction_id', 'resolved'], errors='ignore')
y = df['resolved']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Show learned rules
print("\n=== Decision Tree Rules (Pattern for Resolved Cases) ===\n")
print(export_text(clf, feature_names=list(X.columns)))
