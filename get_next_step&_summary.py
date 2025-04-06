import pandas as pd
import ollama  # For calling the Mistral model via Ollama
import tqdm
import re

# Load the Excel file
file_path = 'unresolved.xlsx'
unresolved_df = pd.read_excel(file_path)

# Prepare rows that need to be processed

# Function to generate the prompt
def generate_prompt(row):
    prompt = f"""
You are an expert in transaction reconciliation.

You are given a transaction case marked as 'unresolved'. Your task is to:

1. Analyze why the transaction might have been marked unresolved.
2. Suggest clear next steps to resolve the issue.

Focus primarily on the `Comments` and `recon_sub_status` fields, but also consider the other transaction metadata.

Respond in this format:
<Summary> Brief summary of the reason </Summary>
<Next Step> Suggested action </Next Step>

Here is the data:
Transaction ID: {row['transaction_id']}
Recon Sub Status: {row['recon_sub_status']}
Comments: {row['Comments']}
Status: {row['status']}
Sys A Date: {row['sys_a_date']}
Sys A Amount Attr1: {row['sys_a_amount_attribute_1']}
Sys A Amount Attr2: {row['sys_a_amount_attribute_2']}
Sys B Date: {row['sys_b_date']}
Sys B Amount Attr1: {row['sys_b_amount_attribute_1']}
Sys B Amount Attr2: {row['sys_b_amount_attribute_2']}
Txn Type: {row['txn_type']}
Payment Method: {row['payment_method']}
Recon Status: {row['recon_status']}
Currency Type: {row['currency_type']}
"""
    return prompt.strip()

# Loop through unresolved rows and update 'summary' and 'next_step'
for idx, row in tqdm.tqdm(unresolved_df.iterrows(), total=unresolved_df.shape[0]):
    prompt = generate_prompt(row)

    response = ollama.chat(
        model='mistral',
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    result = response['message']['content']

    # Extract summary and next step using tag parsing
    summary_match = re.search(r'<Summary>(.*?)</Summary>', result, re.DOTALL)
    next_step_match = re.search(r'<Next Step>(.*?)</Next Step>', result, re.DOTALL)

    summary = summary_match.group(1).strip() if summary_match else ''
    next_step = next_step_match.group(1).strip() if next_step_match else ''

    unresolved_df.at[idx, 'summary'] = summary
    unresolved_df.at[idx, 'next_step'] = next_step
    
# Update the original dataframe

# Save to a new Excel file
output_file = 'unresolved_transactions_updated.xlsx'
unresolved_df.to_excel(output_file, index=False)

print(f"Updated file saved to {output_file}")
