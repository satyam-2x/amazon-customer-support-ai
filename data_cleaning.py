import pandas as pd
import re

def clean_data():
    # Load the raw Twitter customer support dataset
    df = pd.read_csv("data/twcs.csv")
    
    # Filter out customer tweets and AmazonHelp responses
    customer_df = df[df["inbound"] == True]
    amazon_df = df[df["author_id"] == "AmazonHelp"]

    # Match customer tweets with corresponding Amazon replies
    pairs = customer_df.merge(
        amazon_df,
        left_on="tweet_id",
        right_on="in_response_to_tweet_id",
        suffixes=("_customer", "_amazon")
    )

    # Clean up extra spaces in text columns
    pairs["text_customer"] = pairs["text_customer"].str.replace(r"\s+", " ", regex=True).str.strip()
    pairs["text_amazon"] = pairs["text_amazon"].str.replace(r"\s+", " ", regex=True).str.strip()

    # Helper function to remove handles and clean text
    def clean_text(text):
        text = re.sub(r"@AmazonHelp\b", "", text, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", text).strip()

    # Create the final clean dataset structure
    final_pairs = pd.DataFrame({
        "customer_message": pairs["text_customer"].apply(clean_text),
        "amazon_response": pairs["text_amazon"]
    })

    # Drop any empty rows
    final_pairs = final_pairs.dropna().drop_duplicates()

    # Save the cleaned dataset to use for the vector database
    final_pairs.to_csv("data/amazon_pairs.csv", index=False)
    print(f"Dataset cleaned successfully! Saved {len(final_pairs)} rows to data/amazon_pairs.csv")

if __name__ == "__main__":
    clean_data()