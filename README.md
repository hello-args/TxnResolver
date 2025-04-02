# README: TxnResolver

## Overview
This project automates the categorization and resolution of financial transactions using AI-powered classification with the `OllamaLLM` model. It processes transaction data, categorizes cases, determines resolution status, and executes a workflow using a graph-based approach.

## Prerequisites
Ensure you have the following installed before running the script:

- Python (>=3.8)
- pip (>=21.0)

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure `requirements.txt` includes:
   ```
   pandas
   chardet
   langchain
   langgraph
   langchain_ollama
   ```
3. Create necessary directories:
   ```bash
   mkdir -p uploads/not_found_sys_b uploads/processed uploads/pending uploads/resolved
   ```

## Setting Up Ollama and Mistral
### Install Ollama
Ollama is required to run the LLM. Install it using:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Download and Install Mistral Model
To use the `Mistral` model with Ollama, run:
```bash
ollama pull mistral
```

### Verify Installation
To check if Ollama is running and the Mistral model is installed, execute:
```bash
ollama list
```
This should display `mistral` as an available model.

## Configuration
Modify the script to specify the correct paths for input files:
- `recon_data_reply.csv`
- `recon_data_raw.csv`

These CSV files should contain transaction data with necessary columns such as `Transaction ID`, `Comments`, `recon_status`, etc.

## Running the Script
Execute the script with:
```bash
python script.py
```

### Expected Output
- Categorized transaction files saved in `uploads/`
- Processed logs stored in `process.log`
- Resolved and unresolved transactions moved to corresponding directories
- AI-generated recommendations for unresolved transactions

## Logging
All execution logs are stored in `process.log`, which helps track errors and process flow.

## Troubleshooting
- Ensure input CSV files exist and have the correct column headers.
- Check `process.log` for any error messages.
- Ensure `OllamaLLM` is correctly installed and running.
- Verify that the Mistral model is available by running `ollama list`.

## Contributions
Feel free to submit issues or pull requests to enhance the project!

## License
This project is licensed under the MIT License.