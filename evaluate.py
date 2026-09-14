import time
import json
import pandas as pd
from agent import SupportAgent

def run_evaluation():
    # Load the golden test dataset
    try:
        df = pd.read_csv("data/golden_set.csv")
    except FileNotFoundError:
        print("Error: 'golden_set.csv' not found in data folder.")
        return
    
    agent = SupportAgent()
    correct_intents = 0
    total = len(df)
    failures = []

    print(f"\nStarting evaluation on {total} records...\n")

    for index, row in df.iterrows():
        customer_msg = row['customer_message']
        true_intent = row['expected_intent']

        try:
            response = agent.handle_ticket(customer_msg)
            predicted_intent = response.get('intent', '')
            predicted_reply = response.get('reply', '')
        except Exception as e:
            print(f"API Error at row {index + 1}: {e}")
            continue

        # Compare predicted intent with ground truth
        if predicted_intent.strip().lower() == true_intent.strip().lower():
            correct_intents += 1
        else:
            failures.append({
                "row": index + 1,
                "message": customer_msg,
                "expected_intent": true_intent,
                "predicted_intent": predicted_intent,
                "agent_reply": predicted_reply
            })

        # Rate limit control to prevent hitting API quotas
        time.sleep(2)

    
    accuracy = (correct_intents / total) * 100 if total > 0 else 0

    # Print final evaluation summary
    print("EVALUATION SUMMARY")
    print(f"Total Evaluated: {total}")
    print(f"Intent Accuracy: {accuracy:.2f}%")
    print(f"Total Failures: {len(failures)}")

    # Save failed cases for debugging and analysis
    if failures:
        with open("evaluation_failures.json", "w") as f:
            json.dump(failures, f, indent=4)
        print("Failures cases successfully saved to 'evaluation_failures.json'. Use these for your failure analysis section!")


if __name__ == "__main__":
    run_evaluation()