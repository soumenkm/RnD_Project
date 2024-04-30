 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri March 29 08:18:57 2024

@author: soumensmacbookair
"""

"""Utils for running target sentence generation."""
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
    """Extract target sentence from the Mixtral response.

    Args:
        api_response: Target sentence generation response from Mixtral.
    Returns:
        Target sentence and reason
    """
    search_string = "Score:"
    score = 0
    for response in api_response.split("\n"):
        if search_string in response:
            sentence = response.split(search_string)[1].strip()
            if(int(sentence) == 1):
                score = 1
    return score

def run_rarr_target_sentence_cq(
    claim: str,
    model: str,
    prompt: str,
    target_sent: str,
    question: str,
    location: str = None,
) -> List[str]:
    """Generates target sentence based on target location in a claim.

    Given a piece of text (claim), we use Mixtral to generate target sentence.

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
    
    llm_input = prompt.format(target_location=location, target_claim=target_sent, question=question).strip()
    response = prompt_model(model = llm, prompt = llm_input)
    score = parse_api_response(response.strip())
    return score

    