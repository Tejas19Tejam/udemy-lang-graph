from dotenv import load_dotenv
import os

load_dotenv()


if __name__ == "__main__":
    print("Reflexion Agent is running...")
    print(os.environ.get("LANGCHAIN_PROJECT"))
