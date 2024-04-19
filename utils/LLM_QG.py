#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri April 12 08:18:57 2024

@author: soumensmacbookair
"""

"""Utils for running question generation."""
import os, sys
import time
from typing import List
from llama_cpp import Llama
sys.path.append("/root/RnD_Project/prompts")
import rarr_prompts

def prompt_model(model, prompt):
    prompt = "[INST]" + prompt + "[/INST]"
    output = model(prompt, max_tokens=256)
    return output['choices'][0]['text']

def parse_api_response(api_response: str) -> List[str]:
    """Extract questions from the mixtral response.

    Our prompt returns questions as a string with the format of an ordered list.
    This function parses this response in a list of questions.

    Args:
        api_response: Question generation response from GPT-3.
    Returns:
        questions: A list of questions.
    """
    Q_search_string = "Q:"
    questions = []
    for response in api_response.split("\n"):
        if Q_search_string in response:
            question = response.split(Q_search_string)[1].strip()
            questions.append(question)

    return questions

def run_rarr_question_generation(
    target_claim: str,
    location: str,
    model: str,
    prompt: str,
    num_rounds: int = 1
) -> List[str]:
    """Generates questions that interrogate the information in a claim.

    Given a piece of text (claim), we use GPT-3 to generate questions that question the
    information in the claim. We run num_rounds of sampling to get a diverse set of questions.

    Args:
        claim: Text to generate questions off of.
        model: Name of the model to use.
        prompt: The prompt template to query
    Returns:
        questions: A list of questions.
    """
    
    if(model == "mixtral8x7b"):
        llm = Llama(
        model_path="/root/llama.cpp/models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",  
        n_ctx=32768,  # The max sequence length to use - note that longer sequence lengths require much more resources
        n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
        n_gpu_layers=35 ,        # The number of layers to offload to GPU, if you have GPU acceleration available
        verbose=False
        ) 
    elif(model == "noushermes8x7b"):
        llm = Llama(
        model_path="../llama.cpp/models/nous-hermes-2-mixtral-8x7b-dpo.Q4_K_M.gguf",  
        n_ctx=2048,  # The max sequence length to use - note that longer sequence lengths require much more resources
        n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
        n_gpu_layers=35 ,        # The number of layers to offload to GPU, if you have GPU acceleration available
        verbose=False
        )
    else:
        print("Model not found!")
    
    llm_input = prompt.format(location=location, target_claim=target_claim).strip()
        
    # print("\nResponse: ")
    questions = set()
    for _ in range(num_rounds):
        response = prompt_model(model = llm, prompt = llm_input)
        cur_round_questions = parse_api_response(response.strip())
        questions.update(cur_round_questions)

    questions = list(sorted(questions))
    return questions

if __name__ == "__main__":

    # target_sent = "Rajinikanth, often referred to as Thalaiva, is an Indian actor and cultural icon primarily working in Tamil cinema, known for his unique style and larger-than-life characters."

    target_sent = "Shah Rukh Khan, often referred to as the King of Bollywood, is an Indian actor and film producer primarily working in Hindi cinema, recognized for his charm and romantic roles."
    location = "Maharashtra"
    t1 = time.time()
    questions = run_rarr_question_generation(
        target_claim=target_sent,
        location=location,
        model="mixtral8x7b",
        prompt=rarr_prompts.QGEN_PROMPT_WITH_LOCATION_mixtral8x7b,
        entity=None)
    t2 = time.time()
    print(f"Question generation module is run in {(t2-t1)/60:.2f} mint")
    print(f"Target sentence: {target_sent}")
    print(f"Questions: {questions}")



    