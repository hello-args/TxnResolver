import pandas as pd
import os
import logging
from langchain_ollama import OllamaLLM
from langchain.schema import SystemMessage, HumanMessage
from langgraph.graph import StateGraph
from langchain.tools import tool
import shutil
import chardet
from typing import TypedDict

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

# Step 1: Resolution Handling using LLM
def classify_resolution(transaction_id, comment):
    messages = [
        SystemMessage(content="Classify the issue resolution status as 'Resolved' or 'Unresolved'."),
        HumanMessage(content=f"Transaction ID: {transaction_id}, Comment: {comment}")
    ]
    response = llm.invoke(messages).strip()
    logger.info(f"Transaction {transaction_id} classified as: {response}")
    
    summary, next_step = "", ""
    if "Unresolved" in response:
        messages = [
            SystemMessage(content="Explain why the issue is unresolved in a concise statement."),
            HumanMessage(content=f"Transaction ID: {transaction_id}, Comment: {comment}")
        ]
        summary = llm.invoke(messages).strip()
        
        messages = [
            SystemMessage(content="Provide the next step to resolve the issue."),
            HumanMessage(content=f"Transaction ID: {transaction_id}, Comment: {comment}")
        ]
        next_step = llm.invoke(messages).strip()
        response = "Unresolved"
    
    return response, summary, next_step

resolved_orders, unresolved_orders = [], []
try:
    for _, row in raw_data.iterrows():
        status, summary, next_step = classify_resolution(row['txn_ref_id'], row['recon_sub_status'] + "; Comments: " + row.get('Comments', ''))
        entry = [row['txn_ref_id'], status, summary, next_step]
        (resolved_orders if "Resolved" in status else unresolved_orders).append(entry)
    
    pd.DataFrame(resolved_orders, columns=["Transaction ID", "Status", "Summary", "Next Step"]).to_csv("resolved_transactions.csv", index=False)
    pd.DataFrame(unresolved_orders, columns=["Transaction ID", "Status", "Summary", "Next Step"]).to_csv("unresolved_transactions.csv", index=False)
    logger.info("Resolution classification completed.")
except Exception as e:
    logger.error(f"Error processing resolution cases: {e}")
    raise

# Step 2: Graph-Based Workflow
class TransactionState(TypedDict):
    transaction_id: str
    status: str

graph = StateGraph(state_schema=TransactionState)

@tool(description="Move resolved case data to a 'resolved' folder.")
def handle_resolved_case(transaction):
    transaction_id = transaction["transaction_id"]
    try:
        logger.info(f"Handling resolved transaction: {transaction_id}")
        return {"transaction_id": transaction_id, "status": "resolved"}
    except Exception as e:
        logger.error(f"Error handling resolved case {transaction_id}: {e}")
        raise

@tool(description="Generate next steps for unresolved transactions.")
def handle_unresolved_case(transaction):
    transaction_id = transaction["transaction_id"]
    try:
        logger.info(f"Generating next steps for unresolved transaction: {transaction_id}")
        messages = [HumanMessage(content=f"Provide next steps to resolve transaction ID: {transaction_id}.")]
        response = llm.invoke(messages).strip()
        logger.info(f"Next steps for unresolved transaction {transaction_id}: {response}")
        return {"transaction_id": transaction_id, "status": "unresolved"}
    except Exception as e:
        logger.error(f"Error handling unresolved case {transaction_id}: {e}")
        raise

graph.add_node("resolved", handle_resolved_case)
graph.add_node("unresolved", handle_unresolved_case)
graph.add_node("done", lambda x: x)
graph.add_edge("resolved", "done")
graph.add_edge("unresolved", "done")

def determine_start_node(transaction_id):
    return {"transaction_id": transaction_id, "status": "resolved" if transaction_id in resolved_orders else "unresolved"}

graph.set_conditional_entry_point(determine_start_node)
graph.compile()

try:
    logger.info("Executing agentic workflow.")
    for transaction in resolved_orders + unresolved_orders:
        executable_graph = graph.compile()
        executable_graph.invoke({"transaction_id": transaction[0]})
    logger.info("Agentic workflow executed successfully.")
except Exception as e:
    logger.error(f"Error executing workflow: {e}")
    raise

logger.info("Script execution completed.")