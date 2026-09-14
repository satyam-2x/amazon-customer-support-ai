import pandas as pd

def generate_golden_set():
    # Load the clean dataset
    try:
        df = pd.read_csv("data/amazon_pairs.csv")
    except FileNotFoundError:
        print("Error: data/amazon_pairs.csv not found.")
        return

    # Sample 250 random rows for the golden evaluation set
    sample_df = df.sample(n=250, random_state=42).reset_index(drop=True)

    # Create the structure for manual or automated ground-truth labeling
    golden_df = pd.DataFrame({
        'ticket_id': [f"TICKET_{i+1}" for i in range(250)],
        'customer_message': sample_df['customer_message'],
        'expected_intent': "",
        'expected_action': ""
    })

    # Save the template
    golden_df.to_csv("golden_set_blank.csv", index=False)
    print("Golden set template generated successfully: golden_set_blank.csv")

if __name__ == "__main__":
    generate_golden_set()