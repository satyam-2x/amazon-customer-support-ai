import os
import json
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class SupportAgent:
    def __init__(self):
        # Initialize Groq client and ChromaDB vector database connection
        self.llm_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = "openai/gpt-oss-120b"

        self.db_client = chromadb.PersistentClient(path="./.chroma_db")
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.db_client.get_collection(
            name="amazon_support_tickets",
            embedding_function=self.embedding_fn
        )

    
    def retrieve_past_tickets(self, user_message, n_results=2):
        # Search for similar past support tickets in the vector database
        results = self.collection.query(
                query_texts=[user_message],
                n_results=n_results
        )

        past_interactions = ""
        for i, meta in enumerate(results['metadatas'][0]):
            past_interactions += f"Tickets {i+1}:\nCustomer: {meta['customer_message']}\nAmazon: {meta['amazon_response']}\n\n"

        return past_interactions
        

    def handle_ticket(self, user_message):
         # Retrieve context from past interactions
         context = self.retrieve_past_tickets(user_message)

         # Define system instructions and constraints for the LLM
         system_prompt = f"""
            You are an expert Amazon Customer Support Agent.
            Analyze the new customer message and draft a professional reply based ONLY on the provided Past Support Tickets.

            PAST SUPPORT TICKETS (context):
            {context}

            CRITICAL RULES:
            1. Grounding: Rely strictly on the provided Past Support Tickets. Do NOT hallucinate or make up policies.
            2. Escalation Trigger: If the customer mentions legal action, fraud, abusive language, or if the past tickets DO NOT provide a clear resolution, set "action" to "Escalate to Human".
            3. Intent Selection: You must select the "intent" strictly from this allowed list:
            ["Refund", "Replacement", "Order Tracking", "Cancellation", "Payment Issue", "Account Login", "Escalation", "Other"]

            OUTPUT FORMAT:
            You MUST return the output ONLY as a valid JSON object with exact keys:
            {{
               "intent": "Choose strictly from the allowed intent list",
               "action": "Specific action (e.g., 'Issue Refund', 'Track Package', 'Escalate to Human')",
               "escalate": "true or false (e.g., 'true' if human needed, else 'false')",
               "reason": "Brief reason explaining why this action was taken based on context",
               "reply": "Professional, empathetic response drafted for the customer"
           }}
            """
         
         # Call Groq API to process the ticket
         response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"New Customer Message: {user_message}"}
               ],
               response_format={"type": "json_object"},
               temperature=0.2
            )


         ai_output_string = response.choices[0].message.content
         ai_output_dict = json.loads(ai_output_string)
 
         return ai_output_dict