from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()  # reads GEMINI_API_KEY from .env

system_prompt = """You are a helpful assistant for a money exchange shop in Afghanistan.
Explain things in simple language. Keep answers short.
If you don't know today's exact rate, say so. Never invent numbers."""

history = []

while True:
    question = input("\nYou: ")
    if question.lower() == "exit":
        break

    history.append(types.Content(role="user", parts=[types.Part(text=question)]))

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-latest",
            contents=history,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
    except Exception as e:
        print("\nAssistant: Sorry, the AI is busy right now. Please try again.")
        print("(Error details:", e, ")")
        history.pop()   # remove the question that got no answer
        continue

    answer = response.text
    history.append(types.Content(role="model", parts=[types.Part(text=answer)]))
    print("\nAssistant:", answer)