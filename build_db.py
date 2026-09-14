import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os

def build_vector_db():
    client = chromadb.PersistentClient(path="./.chroma_db")


    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name="amazon_support_tickets",
        embedding_function=sentence_transformer_ef
    )


    try:
        df = pd.read_csv("data/amazon_pairs.csv")
    except FileNotFoundError:
        print("Error: data/amazon_pairs.csv not found.")
        return
    
    # Clean and sample data for efficient execution
    df = df.dropna(subset=['customer_message', 'amazon_response'])
    df = df.sample(2000, random_state=42).reset_index(drop=True)

    documents, metadatas, ids = [], [], []

    for index, row in df.iterrows():
        cust_msg = str(row['customer_message'])
        amz_reply = str(row['amazon_response'])


        doc = f"Customer Issue: {cust_msg} | Solved Reply: {amz_reply}"
        documents.append(doc)
        metadatas.append({
            "customer_message": cust_msg,
            "amazon_response": amz_reply
        })
        ids.append(f"ticket{index}")


    # Insert records into ChromaDB in batches
    batch_size = 500
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i : i+batch_size],
            metadatas=metadatas[i : i+batch_size],
            ids=ids[i : i+batch_size]
        )
        print(f"Added batch {i} to {min(i + batch_size, len(documents))}")

    print("Vector database built successfully.")

if __name__ == "__main__":
    build_vector_db()