from dotenv import load_dotenv
import os

load_dotenv()


if __name__ == "__main__":
    print("Hello from React Agent Function Calling Project !")
    print(os.getenv("LANGCHAIN_PROJECT"))
