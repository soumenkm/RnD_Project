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

def chat_gpt(prompt, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    prompt = "What is your gpt version? What is your training cutoff date?"
    response = chat_gpt(prompt, "gpt-4-turbo")
    print(response)