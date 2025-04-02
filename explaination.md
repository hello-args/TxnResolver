# Code Explanation: AI-Powered Transaction Resolution Workflow

## Overview
This script automates the processing, categorization, and resolution of financial transaction records. It uses AI (`OllamaLLM` with `Mistral`) to determine whether a transaction is resolved or unresolved and follows a structured workflow to manage each case accordingly.

## Detailed Code Breakdown

### 1. Importing Required Libraries
The script begins by importing the necessary libraries:
- `pandas`: Handles CSV data loading and manipulation.
- `os`: Manages file system operations (creating directories, file handling).
- `shutil`: Moves files between directories.
- `logging`: Implements logging to track execution flow and errors.
- `chardet`: Detects character encoding in CSV files to prevent reading errors.
- `langchain_ollama`: Integrates the `OllamaLLM` model for AI-driven classification.
- `langgraph.graph`: Implements a structured workflow using state graphs.
- `langchain.tools`: Defines AI-powered tools for handling resolved/unresolved cases.
- `typing.TypedDict`: Provides type hints for structured data representation.

### 2. Logging Setup
Logging is initialized to record all major execution steps and errors:
- `process.log` stores all log messages.
- A file handler writes logs to the file, and a stream handler prints logs to the console.
- Log messages include timestamps and log levels (INFO, ERROR, etc.).

```python
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
```

### 3. Initializing the LLM Model
An AI model (`Mistral`) is loaded via `OllamaLLM` for deterministic classification of transaction statuses:
```python
llm = OllamaLLM(model="mistral", temperature=0)
```
- The `temperature=0` setting ensures the model produces consistent outputs.

### 4. Loading Datasets
The script loads two CSV files:
- `recon_data_reply.csv`: Contains transaction replies, including `Transaction ID` and `Comments`.
- `recon_data_raw.csv`: Raw transaction data.

Before loading, the script detects the character encoding to prevent errors:
```python
def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read())
    return result["encoding"]

def load_csv(file_path):
    encoding = detect_encoding(file_path)
    return pd.read_csv(file_path, encoding=encoding)
```

The datasets are merged to align transaction data with corresponding comments:
```python
merged_df = raw_data.merge(reply_data, left_on='txn_ref_id', right_on='Transaction ID', how='left')
result = merged_df.groupby('txn_ref_id')['Comments'].apply(lambda x: ', '.join(x.dropna())).reset_index()
raw_data = raw_data.merge(result, on='txn_ref_id', how='left')
```

### 5. AI-Based Resolution Classification
For each transaction, AI determines if it is `Resolved` or `Unresolved` and suggests next steps for unresolved cases:
```python
def classify_resolution(transaction_id, comment):
    messages = [
        SystemMessage(content="Classify the issue resolution status as 'Resolved' or 'Unresolved'."),
        HumanMessage(content=f"Transaction ID: {transaction_id}, Comment: {comment}")
    ]
    response = llm.invoke(messages).strip()
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
```

The results are saved to CSV files:
```python
resolved_orders, unresolved_orders = [], []
for _, row in raw_data.iterrows():
    status, summary, next_step = classify_resolution(row['txn_ref_id'], row['recon_sub_status'] + "; Comments: " + row.get('Comments', ''))
    entry = [row['txn_ref_id'], status, summary, next_step]
    (resolved_orders if "Resolved" in status else unresolved_orders).append(entry)

pd.DataFrame(resolved_orders, columns=["Transaction ID", "Status", "Summary", "Next Step"]).to_csv("resolved_transactions.csv", index=False)
pd.DataFrame(unresolved_orders, columns=["Transaction ID", "Status", "Summary", "Next Step"]).to_csv("unresolved_transactions.csv", index=False)
```

### 6. Graph-Based Workflow Execution
A `StateGraph` is built to handle transactions using AI-driven tools:
```python
class TransactionState(TypedDict):
    transaction_id: str
    status: str

graph = StateGraph(state_schema=TransactionState)
```

#### Resolved Case Handler
```python
@tool(description="Move resolved case data to a 'resolved' folder.")
def handle_resolved_case(transaction):
    logger.info(f"Handling resolved transaction: {transaction['transaction_id']}")
    return {"transaction_id": transaction["transaction_id"], "status": "resolved"}
```

#### Unresolved Case Handler
```python
@tool(description="Generate next steps for unresolved transactions.")
def handle_unresolved_case(transaction):
    messages = [
        HumanMessage(content=f"Provide next steps to resolve transaction ID: {transaction['transaction_id']}.")
    ]
    response = llm.invoke(messages).strip()
    return {"transaction_id": transaction["transaction_id"], "status": "unresolved"}
```

### 7. Execution and Logging
The workflow is executed for each transaction:
```python
for transaction in resolved_orders + unresolved_orders:
    executable_graph = graph.compile()
    executable_graph.invoke({"transaction_id": transaction[0]})
```

## Summary
- **Data Handling**: Loads, merges, and categorizes transaction records.
- **AI-Powered Analysis**: Uses `OllamaLLM` to classify cases and suggest next steps.
- **Automated Workflow**: Processes cases using a structured state graph.
- **Logging & Error Handling**: Ensures transparency and traceability.

This script streamlines transaction reconciliation using AI-driven automation, improving efficiency and accuracy.

