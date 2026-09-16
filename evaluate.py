import time
import re
import pandas as pd
import os
import csv
from google import genai
from agent import SupportAgent
from dotenv import load_dotenv


load_dotenv()

def get_gemini_score(customer_msg, expected_action, agent_reply):
    prompt = f"""
    You are an expert evaluator. 
    Customer Message: {customer_msg}
    Expected Action/Resolution: {expected_action}
    AI Agent Reply: {agent_reply}
    
    Rate the AI Agent Reply from 1 to 5 based on how well it fulfills the Expected Action.
    Output ONLY a single integer number between 1 and 5.
    """

    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=prompt
        )
        # Extract numeric score safely via regex to prevent parsing errors
        match = re.search(r'\d+', response.text)
        return int(match.group()) if match else 3
    except Exception as e:
        print(f"Gemini Score Error: {e}")
        return 3

def run_evaluation():
    try:
        df = pd.read_csv("data/golden_set.csv")
    except FileNotFoundError:
        print("Error: 'golden_set.csv' not found in data folder.")
        return
    
    agent = SupportAgent()
    
    # Configure batch range indices
    START_INDEX = 1
    END_INDEX = 100
    
    batch_df = df.iloc[START_INDEX:END_INDEX]
    csv_filename = "final_evaluation_results.csv"

    print(f"\n🚀 Starting Run from Row {START_INDEX + 1} to {END_INDEX}...\n")

    for index, row in batch_df.iterrows():
        ticket_id = row['ticket_id']
        customer_msg = row['customer_message']
        true_intent = row['expected_intent']
        true_action = row['expected_action'] 

        print(f"Processing Row {index + 1} (Ticket ID: {ticket_id})...")

        # Get prediction from Support Agent (Groq + ChromaDB)
        try:
            response = agent.handle_ticket(customer_msg)
            predicted_intent = response.get('intent', '')
            predicted_escalation = str(response.get('escalate', '')).strip().lower()
            predicted_reply = response.get('reply', '')
        except Exception as e:
            print(f"API Error at row {index + 1}: {e}")
            predicted_intent, predicted_escalation, predicted_reply = "", "", f"Error: {e}"

        
        # Evaluate response quality using Gemini Judge
        judge_score = get_gemini_score(customer_msg, true_action, predicted_reply)

        # Prepare structured output row data
        row_data = {
            "Row": index + 1,
            "Ticket_ID": ticket_id,
            "Customer_Message": customer_msg,
            "Expected_Intent": true_intent,
            "Predicted_Intent": predicted_intent,
            "Expected_Action": true_action,
            "Predicted_Escalation": predicted_escalation,
            "Agent_Reply": predicted_reply,
            "Gemini_Score": judge_score
        }

        # Fail-safe incremental append to CSV file
        file_exists = os.path.isfile(csv_filename)
        with open(csv_filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row_data.keys())
            if not file_exists:
                writer.writeheader() 
            writer.writerow(row_data) 

        # Rate limit protection delay between requests
        time.sleep(15) 
    
    print(f"\n Batch from {START_INDEX + 1} to {END_INDEX} completed and safely saved in {csv_filename}!")


if __name__ == "__main__":
    run_evaluation()