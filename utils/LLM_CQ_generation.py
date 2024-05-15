#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri March 29 08:18:57 2024

@author: soumensmacbookair
"""

"""Utils for running common question generation."""
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

def run_common_question_generation(
    claim: str,
    model: str,
    prompt: list
) -> List[str]:
    """Generates common question in a claim.

    Given a piece of text (claim), we use Mixtral to generate common question.

    Args:
        claim: Text to generate questions off of.
        model: Name of the OpenAI GPT-3 model to use.
        prompt: The prompt template to query GPT-3 with.
    Returns:
        target sentence and reason
    """
    
    if(model == "mixtral8x7b"):
        llm = Llama(
        model_path="/root/llama.cpp/models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",  
        n_ctx=32768,  # The max sequence length to use - note that longer sequence lengths require much more resources
        n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
        n_gpu_layers=35 ,        # The number of layers to offload to GPU, if you have GPU acceleration available
        verbose=False
        ) 
    else:
        print("Model not found!")
    
    llm_input = prompt[0].format(claim=claim).strip()
    response = prompt_model(model = llm, prompt = llm_input)

    ques_list = response.split("\n")
    ques_list = [i for i in ques_list if "Q:" in i]
    
    out_list = []
    for i in ques_list:
        question = i.replace("Q:","").strip()
        
        rev_llm_input = prompt[1].format(question=question).strip()
        question = prompt_model(model = llm, prompt = rev_llm_input)
        out_list.append(question.split(":")[-1].strip())
    
    return out_list

if __name__ == "__main__":

    data = [{"input_info": 
    {"claim": "Angelina Jolie is an American actress, filmmaker and humanitarian. The recipient of numerous accolades, including an Academy Award and three Golden Globe Awards, she has been named Hollywood's highest-paid actress multiple times.", 
    "location": "Kerala"}},
    {"input_info": 
    {"claim": "On July 10, 2006, concrete ceiling panels and debris fell on a car traveling on the two-lane ramp connecting northbound I-93 to eastbound I-90 in South Boston, killing the passenger and injuring her husband, who was driving.", 
    "location": "Uttarakhand"}},
    {"input_info": {"claim": "Robert John Downey Jr. is an American actor. His career has been characterized by critical success in his youth, followed by a period of substance abuse and legal troubles, and a surge in popular and commercial success later in his career.", 
    "location": "West Bengal"}}]

    claim_id = 1
    claim = data[claim_id]["input_info"]["claim"]
    entity = None
    location = data[claim_id]["input_info"]["location"]

    t1 = time.time()
    common_ques_list = run_common_question_generation(
        claim=claim,
        model="mixtral8x7b",
        prompt=[rarr_prompts.COMMON_QUESTION_GEN_PROMPT_mixtral8x7b,
                rarr_prompts.CHECK_IF_COMMON_QUESTION_PROMPT_mixtral8x7b])
    t2 = time.time()
    
    print(f"Claim: {claim_id}, Common question generation module is run in {(t2-t1)/60:.2f} mint")
    print(common_ques_list)

    