from typing import Optional   #library for null data
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel # for structuring data 

load_dotenv() #loading env file where API key is stored 
client = genai.Client() # and makes the connection btw app and Gemini main server

class HawalaRequest(BaseModel): # a structure form for data on which AI should focuse
    amount: Optional[float]
    currency: Optional[str]
    sender: Optional[str]
    receiver: Optional[str]
    city: Optional[str]
    fee: Optional[float]

class HawalaBatch(BaseModel):  # this class contain another class for double data 
    transfers: list[HawalaRequest]

system_prompt = """You extract Hawala transfer details from customer messages.
Messages may be in English, Pashto, Dari, or mixed.
Only use information written in the message.
If a detail is missing, leave it empty (null). Never guess.
Write currency as a 3-letter code, like USD, AFN, PKR."""

while True:
    message = input("\nMessage: ")
    if message.lower() == "exit":
        break

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json", #tells Gemini don't chat, just give json
                response_schema=HawalaBatch,
            ),
        )
    except Exception as e: #error occured system show message and should continue replace of crashing
        print("AI error:", e)
        continue

    data = response.parsed
    if data is None or not data.transfers:
        print("⚠ No transfer found in this message.")
        continue

    print(f"Found {len(data.transfers)} transfer(s):")
    for number, transfer in enumerate(data.transfers, start=1):
        print(f"\n--- Transfer {number} ---")
        print(transfer.model_dump_json(indent=2))

        missing = [name for name, value in transfer.model_dump().items() if value is None]
        if missing:
            print("⚠ Missing:", ", ".join(missing))
        if transfer.amount is not None and transfer.amount <= 0:
            print("⚠ Amount must be more than zero")