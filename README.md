<!-- # AI Customer Support Agent - Amazon Support RAG System

An AI-powered customer support agent built using real Amazon support conversations from the Customer Support on Twitter dataset.

The system:

* Classifies incoming customer messages into support intents.
* Drafts replies grounded in similar historical Amazon responses using a RAG pipeline.
* Decides whether to auto-handle the request or escalate it to a human, with a reason.

---

## 🚀 15-Minute Reproduction Guide

Follow these steps to set up the project and reproduce the evaluation results locally.

### 1. Clone the Repository

```bash
git clone https://github.com/satyam-2x/amazon-customer-support-ai.git
cd amazon-customer-support-ai
```

### 2. Set Up the Environment

Create and activate a virtual environment:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

Install the required dependencies:

```bash
pip install -r requirements.txt

Create a `.env` file in the project root and add your Groq API key:

```env
GROQ_API_KEY="your_groq_api_key"
GEMINI_API_KEY="your_gemini_api_key"
```

### 3. Prepare the Data

To run the agent and evaluation pipeline, your dataset files must be placed correctly inside a folder named `data` in the project root.

1. Create a folder named `data` in your project directory if it doesn't exist.
2. Place the required CSV files inside the `data/` folder:
   ```text
   amazon-customer-support-ai/
   └── data/
       ├── twcs.csv             # The raw Twitter support dataset (used by ChromaDB for RAG context)
       └── golden_set.csv       # Your 200-row hand-labelled test set (used by evaluate.py)


### 4. Run the Evaluation Pipeline

To strictly respect API rate limits (Groq's 8,000 TPM / 200,000 TPD) and reproduce the headline evaluation results in under 15 minutes, the evaluation harness runs in two fail-safe batches with built-in rate-limiting and append-mode logging[cite: 1].

#### **Batch 1 (Rows 1 to 100):**
1. Open `evaluate.py` and ensure the indices are set to:
   ```python
   START_INDEX = 0
   END_INDEX = 100 -->




