
import os
import csv
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# ==========================================
# 1. CONFIGURATION
# ==========================================

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = os.getenv("MODEL_NAME", "gpt-6-luna")

# Prices in USD per 1 million tokens.
# Update these values according to the official pricing page.
PRICING = {
    "gpt-6-luna": {
        "input": 0.05,
        "output": 0.25,
        "cached_input": 0.005
    }
}

CSV_FILE = "api_usage.csv"

# ==========================================
# 2. INITIALIZE TOTAL STATISTICS
# ==========================================

total_input_tokens = 0
total_output_tokens = 0
total_tokens = 0
total_cost = 0.0
request_count = 0

# ==========================================
# 3. INITIALIZE CONVERSATION
# ==========================================

messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant."
    }
]

# ==========================================
# 4. CREATE CSV FILE
# ==========================================

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Timestamp",
            "Model",
            "Input Tokens",
            "Output Tokens",
            "Total Tokens",
            "Cached Tokens",
            "Cost USD"
        ])

# ==========================================
# 5. TOKEN AND COST ANALYSIS
# ==========================================

def analyze_usage(response):

    global total_input_tokens
    global total_output_tokens
    global total_tokens
    global total_cost
    global request_count

    usage = response.usage

    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    tokens = usage.total_tokens

    # Cached input tokens, if provided
    cached_tokens = 0

    if usage.prompt_tokens_details:
        cached_tokens = (
            usage.prompt_tokens_details.cached_tokens or 0
        )

    # Get pricing for the actual returned model
    model = response.model
    prices = PRICING.get(model, PRICING.get(MODEL))

    if prices is None:
        raise ValueError(
            f"No pricing configured for model: {model}"
        )

    # Separate cached and non-cached input
    normal_input = input_tokens - cached_tokens

    # Calculate estimated cost
    input_cost = (
        normal_input * prices["input"]
        + cached_tokens * prices["cached_input"]
    ) / 1_000_000

    output_cost = (
        output_tokens * prices["output"]
    ) / 1_000_000

    request_cost = input_cost + output_cost

    # Update cumulative statistics
    total_input_tokens += input_tokens
    total_output_tokens += output_tokens
    total_tokens += tokens
    total_cost += request_cost
    request_count += 1

    # Display request statistics
    print("\n========== REQUEST ANALYSIS ==========")
    print(f"Model:              {model}")
    print(f"Input tokens:       {input_tokens:,}")
    print(f"Output tokens:      {output_tokens:,}")
    print(f"Cached tokens:      {cached_tokens:,}")
    print(f"Total tokens:       {tokens:,}")
    print(f"Input cost:         ${input_cost:.8f}")
    print(f"Output cost:        ${output_cost:.8f}")
    print(f"Request cost:       ${request_cost:.8f}")

    print("\n========== CUMULATIVE ANALYSIS ========")
    print(f"Total requests:     {request_count}")
    print(f"Total input tokens: {total_input_tokens:,}")
    print(f"Total output tokens:{total_output_tokens:,}")
    print(f"Total tokens:       {total_tokens:,}")
    print(f"Total API cost:     ${total_cost:.8f}")
    print("=======================================\n")

    # Save request information to CSV
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            model,
            input_tokens,
            output_tokens,
            tokens,
            cached_tokens,
            round(request_cost, 10)
        ])


# ==========================================
# 6. CHAT COMPLETION
# ==========================================

def chat():

    print("Chat Completion API")
    print(f"Model: {MODEL}")
    print("Type 'exit' to stop.\n")

    while True:

        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        messages.append({
            "role": "user",
            "content": user_input
        })

        try:

            response = client.chat.completions.create(
                model=MODEL,
                messages=messages
            )

            assistant_reply = response.choices[0].message.content

            print("\nAssistant:", assistant_reply)

            messages.append({
                "role": "assistant",
                "content": assistant_reply
            })

            # Analyze tokens and cost automatically
            analyze_usage(response)

        except Exception as error:

            print("API Error:", error)

            # Remove unsuccessful user message
            messages.pop()

    print("\n========== FINAL SUMMARY =============")
    print(f"Requests:       {request_count}")
    print(f"Total tokens:   {total_tokens:,}")
    print(f"Total cost:     ${total_cost:.8f}")
    print(f"Usage saved to: {CSV_FILE}")
    print("=======================================")


# ==========================================
# 7. MAIN
# ==========================================

if __name__ == "__main__":

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing from .env")

    chat()
