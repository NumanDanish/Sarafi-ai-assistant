# Sarafi AI Assistant

A step-by-step AI learning project for a money exchange business.

## Level 1: AI Chatbot

A command-line chatbot using the Google Gemini API.

### What I learned

API: your code talks to the AI on someone else's servers, using a secret key.
Messages and roles: a conversation is a list of user and model messages.
System prompt: gives the AI its role, rules, and facts.
Memory: the AI has none; your code keeps and sends the history.
Live data: the AI only knows its training plus what you give it.

### How to run

1. Create a virtual environment and install: pip install google-genai python-dotenv
2. Create a .env file with GEMINI_API_KEY=your-key
3. Run: python chat.py
