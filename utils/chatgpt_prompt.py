import os
import openai
from openai import OpenAI
import sys
sys.path.append("/root/RnD_Project/utils")
from api import OPENAI_KEY
client = OpenAI(api_key=OPENAI_KEY)

def chat_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    prompt = "Who is prime minister of India?"
    response = chat_gpt(prompt)
    print(response)