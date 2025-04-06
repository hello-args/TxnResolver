import pandas as pd
import os
import chardet
import logging
from langchain_ollama import OllamaLLM
from langchain.schema import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from typing_extensions import TypedDict
from typing import Literal
from pydantic import BaseModel
import ast
import json
import csv
import math
import re


nan = math.nan

def write_dicts_to_csv(filename, data):
    """
    Write a list of dictionaries to a CSV file.

    :param filename: Name of the CSV file
    :param data: List of dictionaries
    """
    if not data:
        return

    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

def append_dict_to_json(filename, new_data):
    """
    Append a dictionary to a JSON file as part of a list.

    :param filename: Name of the JSON file
    :param new_data: Dictionary to append
    """
    # Load existing data if file exists
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            try:
                data = json.load(file)
                if not isinstance(data, list):
                    data = []
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Append the new data
    data.append(new_data)

    # Write updated list back to file
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


def transform_dict_in_data(entry):
    try:
        content = entry['content']
        # Replace unquoted `nan` with `None`
        safe_content = re.sub(r'\bnan\b', 'None', content)

        # Now safely evaluate
        content_dict = ast.literal_eval(safe_content)
        t = entry.pop('content')  # Fallback
    except (ValueError, SyntaxError):
        print("Invalid content:", entry['content'])
        content_dict = {}  # Default empty dict         
    # Merge with parent dictionary
    merged_dict = {**entry, **content_dict}
    return merged_dict


def transform_data(data):
    transformed_data = []
    for entry in data:
        # Convert 'content' string to dictionary
        merged_dict = transform_dict_in_data(entry)
        # Append to result list
        transformed_data.append(merged_dict)
    return transformed_data


class TransactionInput(BaseModel):
    transaction_id: str
    summary: str = ""  # Optional with default value


# Configure logging
log_file = "process.log"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_file, mode="w")
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logger.info("Logging initialized.")

# Initialize LLM
llm = OllamaLLM(model="mistral", temperature=0)
logger.info("Ollama LLM initialized.")

# Load datasets with encoding detection
def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read())
    return result["encoding"]

def load_csv(file_path):
    encoding = detect_encoding(file_path)
    return pd.read_csv(file_path, encoding=encoding)

try:
    logger.info("Loading datasets.")
    reply_data = load_csv("recon_data_reply.csv")
    raw_data = load_csv("recon_data_raw.csv")
    logger.info("Datasets loaded successfully.")
except Exception as e:
    logger.error(f"Error loading datasets: {e}")
    raise

# Merge data for processing
merged_df = raw_data.merge(reply_data, left_on='txn_ref_id', right_on='Transaction ID', how='left')
result = merged_df.groupby('txn_ref_id')['Comments'].apply(lambda x: ', '.join(x.dropna())).reset_index()
raw_data = raw_data.merge(result, on='txn_ref_id', how='left')

# Define transaction state
class TransactionState(TypedDict):
    transaction_id: str
    status: str
    summary: str
    next_step: str
    content: str

graph_builder = StateGraph(TransactionState)

def classify_resolution(transaction: TransactionState):  
    comment = transaction["content"] or ""
    if not transaction["transaction_id"]:
        raise ValueError("Missing required field: transaction_id")

    messages = [
        SystemMessage(content=(
            "You are a reconciliation analyst reviewing financial transactions for discrepancies.\n\n"
            "Analyze the provided transaction details and return a Python dictionary with the following keys:\n"
            "- 'status': Either 'resolved' or 'unresolved'.\n"
            "- 'summary': A concise summary of the issue in no more than 3 to 4 lines.\n"
            "- 'next_step': If status is 'unresolved', provide the next step to resolve the issue. If 'resolved', set this to None.\n\n"
            "Make sure to have a proper python dict with only three keys: status, summary and next_step."
            "Key aspects to consider:\n"
            "- Focus on 'recon_status' and 'comments' for determining resolution status.\n"
            "- If the issue is unresolved due to missing key information, explicitly mention the missing keys.\n"
            "- The relevant keys to check are:\n"
            "  - 'sys_a_amount_attribute_1', 'sys_a_amount_attribute_2'\n"
            "  - 'sys_b_date', 'sys_b_amount_attribute_1', 'sys_b_amount_attribute_2'\n"
            "  - 'txn_type', 'payment_method', 'recon_status'\n"
            "- If required information is missing, state that the transaction might not be complete.\n\n"
            
            "Return only a structured Python dictionary as output, without any additional explanations."
        )),
        HumanMessage(content=f"Transaction details: {transaction}, Comment: {comment}")
    ]
    
    response = llm.invoke(messages).strip()
    logger.info(f"Transaction {transaction['transaction_id']} classified with info: {response}")
    
    try:
        result = eval(response)  # Convert string response to dictionary safely (consider using json.loads if API returns JSON)
        if isinstance(result, dict) and "status" in result and "summary" in result and "next_step" in result:
            return result
    except Exception as e:
        result = dict()
        result["status"] = "unknown"
        result["summary"] = ""
        result["next_step"] = ""
        logger.debug(f"Failed to parse response: {e}, trying again")
        try:
            messages2 = [
            SystemMessage(content=(
                "Analyze the provided python dict string which has errors and return a fixed Python dictionary with the following keys:\n"
                "- 'status': Either 'resolved' or 'unresolved'.\n"
                "- 'summary': A concise summary of the issue in no more than 3 to 4 lines.\n"
                "- 'next_step': If status is 'unresolved', provide the next step to resolve the issue.\n\n"
                "Make sure to have a proper python dict with only three keys: status, summary and next_step."
                "Key aspects to consider:\n"
                "- Quotes are handled properly\n"
                "- All the three keys are present\n"
                "- Double Quotes and Single Quoates are handled properly\n"
                
                "Return only a structured Python dictionary as output, without any additional explanations."
            )),
            HumanMessage(content=f"{str(response)}")
            ]
            response2 = llm.invoke(messages2).strip()
            result = eval(response2)  # Convert string response to dictionary safely (consider using json.loads if API returns JSON)
            if isinstance(result, dict) and "status" in result and "summary" in result and "next_step" in result:
                return result
        except Exception as e:
            result = dict()
            result["status"] = "unknown"
            result["summary"] = str(e)
            result["next_step"] = ""
            logger.debug(f"Failed to parse response: {e}")
    
    result["transaction_id"] = transaction['transaction_id']
    result["comment"] = comment
    
    return {"transaction": result}

def classify_next_step(transaction) -> Literal['resolved', 'unresolved']:
    ### We will use state to decide the next node to visit
    try:
        response = "resolved" if transaction["status"] == "resolved" else "unresolved"
    except Exception as e:
        if "transaction" in transaction.keys():
            response = "resolved" if transaction["transaction"]["status"] == "resolved" else "unresolved"
        else:
            response = "unresolved"
    return response

resolved_list = list()
unresolved_list = list()

def handle_resolved_case(transaction: TransactionState):
    append_dict_to_json("resolved_stream.json",  transform_dict_in_data(transaction))
    resolved_list.append(transaction)
    return transaction

def handle_unresolved_case(transaction: TransactionState):
    append_dict_to_json("unresolved_stream.json",  transform_dict_in_data(transaction))
    unresolved_list.append(transaction)
    return transaction

# Define graph structure
graph_builder.add_node("classify", lambda state: classify_resolution(transaction))
graph_builder.add_node("resolved", lambda state: handle_resolved_case(transaction))
graph_builder.add_node("unresolved", lambda state: handle_unresolved_case(transaction))

graph_builder.add_edge(START, "classify")
graph_builder.add_conditional_edges("classify", classify_next_step)
graph_builder.add_edge("resolved", END)
graph_builder.add_edge("unresolved", END)

graph = graph_builder.compile()

raw_data = raw_data.loc[2588:].reset_index(drop=True)

try:
    logger.info("Executing graph workflow.")
    for _, row in raw_data.iterrows():
        transaction = {
            "transaction_id": str(row['txn_ref_id']),
            "status": "",  # Ensure this key exists
            "content": str(row.drop('txn_ref_id').to_dict()),
            "next_step": "",  # Ensure this key exists
            "summary": "",
        }
        graph.invoke(input={"transaction": transaction})
        logger.info("1 event done")
    logger.info("Workflow executed successfully.")
except Exception as e:
    logger.error(f"Error executing workflow: {e}")
    raise


transformed_resolved = transform_data(resolved_list)
transformed_unresolved = transform_data(unresolved_list)

transformed_resolved_df = pd.DataFrame(transformed_resolved)
transformed_resolved_df.to_csv("transformed_resolved.csv", index=False)

transformed_unresolved_df = pd.DataFrame(transformed_unresolved)
transformed_unresolved_df.to_csv("transformed_unresolved.csv", index=False)

logger.info("Script execution completed.")