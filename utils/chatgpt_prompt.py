#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue April 02 17:47:04 2024

@author: soumensmacbookair
"""

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