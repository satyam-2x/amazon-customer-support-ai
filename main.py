import os
from dotenv import load_dotenv
from agent import SupportAgent

# Load environment variables from the .env file
load_dotenv()


def main():
    # Initialize the support agent
    agent = SupportAgent()
    print("Support Agent is ready. Type 'exit' to quit.\n")
    

    while True:
        try:
            user_message = input("Customer Message: ").strip()
            
            if user_message.strip().lower() == 'exit':
                print("Exiting...")
                break
                
            if not user_message.strip():
                continue

            # Process the ticket using the agent
            response = agent.handle_ticket(user_message)

            # Output the agent's decision and response
            print(f"\nIntent : {response.get('intent')}")
            print(f"Action : {response.get('action')}")
            print(f"Reason : {response.get('reason')}")
            print(f"Reply  :\n{response.get('reply')}\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")


if __name__ == "__main__":
    main()