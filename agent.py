from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()

MODEL = "gemini-3.5-flash-lite"

# Made-up practice data (later this could come from Zaki Pay or a rate API)
RATES = {
    ("USD", "AFN"): 70.0,
    ("USD", "PKR"): 280.0,
    ("EUR", "AFN"): 76.0,
}

BALANCES = {
    "karim": 12000,
    "ahmad": 5500,
    "bilal": 30000,
}

def get_rate(from_currency: str, to_currency: str) -> dict:
    """Get today's exchange rate between two currencies.
    Use 3-letter codes like USD, AFN, PKR, EUR."""
    print(f"  [tool] get_rate({from_currency}, {to_currency})")
    rate = RATES.get((from_currency.upper(), to_currency.upper()))
    if rate is None:
        return {"error": "This rate is not available."}
    return {"rate": rate}


def convert(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount of money from one currency to another using today's rate."""
    print(f"  [tool] convert({amount}, {from_currency}, {to_currency})")
    rate = RATES.get((from_currency.upper(), to_currency.upper()))
    if rate is None:
        return {"error": "This rate is not available."}
    return {"result": round(amount * rate, 2), "rate": rate}


def get_balance(customer_name: str) -> dict:
    """Get a customer's account balance in AFN."""
    print(f"  [tool] get_balance({customer_name})")
    balance = BALANCES.get(customer_name.lower())
    if balance is None:
        return {"error": "Customer not found."}
    return {"balance_afn": balance}


def calculate_hawala_fee(amount_usd: float) -> dict:
    """Calculate the Hawala fee in USD for sending money.
    The fee is 1% of the amount, with a minimum of 2 USD."""
    print(f"  [tool] calculate_hawala_fee({amount_usd})")
    fee = max(amount_usd * 0.01, 2)
    return {"fee_usd": round(fee, 2)}

system_prompt = """You are an assistant for a money exchange shop in Afghanistan.
Use the tools for any exchange rate, conversion, balance, or fee.
Never calculate money yourself and never invent numbers.
If a tool returns an error, explain it simply to the customer.
Keep answers short and clear."""

chat = client.chats.create(
    model=MODEL,
    config=types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[get_rate, convert, get_balance, calculate_hawala_fee],
    ),
)

while True:
    question = input("\nYou: ")
    if question.lower() == "exit":
        break

    try:
        response = chat.send_message(question)
    except Exception as e:
        print("AI error:", e)
        continue

    print("\nAssistant:", response.text)

    