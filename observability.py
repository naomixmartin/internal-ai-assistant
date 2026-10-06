import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))

def log_request(request_data):
    # observability should never cause the chatbot itself to fail
    try:
        supabase.table("request_logs").insert(request_data).execute()
    except Exception as e:
        print(f"failed to log request: {e}")


def estimate_cost(model, input_tokens, output_tokens):
    # return no estimate when token usage is unavailable
    if input_tokens is None or output_tokens is None:
        return None

    # prices are USD per 1 million tokens
    pricing = {
        "gemini-3.5-flash-lite": {
            "input": 0.30,
            "output": 2.50
        },
        "gemini-3.8-flash": {
            "input": 0.75,
            "output": 3.75
        }
    }

    if model not in pricing:
        return None

    input_cost = (input_tokens / 1_000_000) * pricing[model]["input"]
    output_cost = (output_tokens / 1_000_000) * pricing[model]["output"]

    return input_cost + output_cost