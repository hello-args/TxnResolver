import pandas as pd
import os
import json

filename_unresolved = "unresolved_stream.json"
filename_resolved = "resolved_stream.json"

if os.path.exists(filename_unresolved):
    with open(filename_unresolved, 'r') as unresolved_file:
        try:
            unresolved_data = json.load(unresolved_file)
            if not isinstance(unresolved_data, list):
                unresolved_data = []
        except json.JSONDecodeError:
            unresolved_data = []
else:
    unresolved_data = []


if os.path.exists(filename_resolved):
    with open(filename_resolved, 'r') as resolved_file:
        try:
            resolved_data = json.load(resolved_file)
            if not isinstance(resolved_data, list):
                resolved_data = []
        except json.JSONDecodeError:
            resolved_data = []
else:
    resolved_data = []


df_resolved = pd.DataFrame(resolved_data)
df_unresolved = pd.DataFrame(unresolved_data)

print("done")

df_resolved.to_excel("resolved.xlsx")
df_unresolved.to_excel("unresolved.xlsx")
