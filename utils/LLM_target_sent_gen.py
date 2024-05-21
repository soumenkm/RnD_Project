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
import ablation_prompts 

def prompt_model(model, prompt):
    prompt = "[INST]" + prompt + "[/INST]"
    output = model(prompt, max_tokens=256)
    return output['choices'][0]['text']

def parse_api_response(api_response: str) -> List[str]:
    """Extract target sentence from the Mixtral response.

    Args:
        api_response: Target sentence generation response from Mixtral.
    Returns:
        Target sentence and reasons
    """
    Sent_search_string = "Target sentence:"
    Reason_search_string = "Reason:"
    sentence = ""
    reason = ""
    for response in api_response.split("\n"):
        if Sent_search_string in response:
            sentence = response.split(Sent_search_string)[1].strip()
        elif Reason_search_string in response:
            reason = response.split(Reason_search_string)[1].strip()
    return sentence, reason

def run_rarr_target_sentence_generation(
    quant: str,
    claim: str,
    model: str,
    prompt: str,
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
    if quant == "Q4":
        model_path = "/root/llama.cpp/models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
    elif quant == "Q6":
        model_path = "/root/llama.cpp/models/mixtral-8x7b-instruct-v0.1.Q6_K.gguf"
    
    if(model == "mixtral8x7b"):
        llm = Llama(
        model_path=model_path,  
        n_ctx=32768,  # The max sequence length to use - note that longer sequence lengths require much more resources
        n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
        n_gpu_layers=35 ,        # The number of layers to offload to GPU, if you have GPU acceleration available
        verbose=False
        ) 
    else:
        print("Model not found!")
    
    llm_input = prompt.format(location=location, claim=claim).strip()
    count_local = llm_input.count(location)
    response = prompt_model(model = llm, prompt = llm_input)
    target_sent, reason = parse_api_response(response.strip())
    return target_sent, reason, count_local

def run_rarr_dynamic_target_sentence_generation(
    quant: str,
    claim: str,
    model: str,
    prompt: str,
    prompt_name: str,
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
    
    if quant == "Q4":
        model_path = "/root/llama.cpp/models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
    elif quant == "Q6":
        model_path = "/root/llama.cpp/models/mixtral-8x7b-instruct-v0.1.Q6_K.gguf"
    
    if(model == "mixtral8x7b"):
        llm = Llama(
        model_path=model_path,  
        n_ctx=32768,  # The max sequence length to use - note that longer sequence lengths require much more resources
        n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
        n_gpu_layers=35 ,        # The number of layers to offload to GPU, if you have GPU acceleration available
        verbose=False
        ) 
    else:
        print("Model not found!")
    
    if "1_shot" in prompt_name:
        llm_input = prompt.format(location=location, 
            claim=claim,
            dyn_ref_claim_1=ablation_prompts.prompt_data[location]["dyn_ref_claim_1"],
            dyn_tar_claim_1=ablation_prompts.prompt_data[location]["dyn_tar_claim_1"],
            dyn_reason_1=ablation_prompts.prompt_data[location]["dyn_reason_1"]).strip()
    elif "2_shot" in prompt_name:
        llm_input = prompt.format(location=location, 
            claim=claim,
            dyn_ref_claim_1=ablation_prompts.prompt_data[location]["dyn_ref_claim_1"],
            dyn_ref_claim_2=ablation_prompts.prompt_data[location]["dyn_ref_claim_2"],
            dyn_tar_claim_1=ablation_prompts.prompt_data[location]["dyn_tar_claim_1"],
            dyn_tar_claim_2=ablation_prompts.prompt_data[location]["dyn_tar_claim_2"],
            dyn_reason_1=ablation_prompts.prompt_data[location]["dyn_reason_1"],
            dyn_reason_2=ablation_prompts.prompt_data[location]["dyn_reason_2"]).strip()
        
    count_local = llm_input.count(location)
    response = prompt_model(model = llm, prompt = llm_input)
    target_sent, reason = parse_api_response(response.strip())
    return target_sent, reason, count_local

def run_rarr_target_sentence_regeneration(
    claim: str,
    model: str,
    prompt: str,
    target_sent: str,
    location: str = None,
    question: str = None
) -> List[str]:
    """Re-generates target sentence based on target location in a claim and common questions.

    Given a piece of text (claim) and common question, we use Mixtral to re-generate target sentence.

    Args:
        claim: Text to generate questions off of.
        model: Name of the OpenAI GPT-3 model to use.
        prompt: The prompt template to query GPT-3 with.
        target_sent: The target sentence to be re-generated.
        location: target location
        question: common question
    Returns:
        revised target sentence, reason
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
    
    llm_input = prompt.format(target_sent = target_sent, location=location, claim=claim, question=question).strip()
    response = prompt_model(model = llm, prompt = llm_input)
    target_sent, reason = parse_api_response(response.strip())
    return target_sent, reason


if __name__ == "__main__":

    data = [{"input_info": 
    {"claim": "Angelina Jolie is an American actress, filmmaker and humanitarian. The recipient of numerous accolades, including an Academy Award and three Golden Globe Awards, she has been named Hollywood's highest-paid actress multiple times.", 
    "location": "Kerala"}},
    {"input_info": 
    {"claim": "On July 10, 2006, concrete ceiling panels and debris fell on a car traveling on the two-lane ramp connecting northbound I-93 to eastbound I-90 in South Boston, killing the passenger and injuring her husband, who was driving.", 
    "location": "Uttarakhand"}},
    {"input_info": {"claim": "Robert John Downey Jr. is an American actor. His career has been characterized by critical success in his youth, followed by a period of substance abuse and legal troubles, and a surge in popular and commercial success later in his career.", 
    "location": "West Bengal"}}]

    claim_id = 0
    claim = data[claim_id]["input_info"]["claim"]
    entity = None
    location = data[claim_id]["input_info"]["location"]

    t1 = time.time()
    target_sent, reason = run_rarr_dynamic_target_sentence_generation(
        claim=claim,
        model="mixtral8x7b",
        prompt=dynamic_prompt_for_target_sent_gen.TARGET_SENT_GEN_PROMPT_WITH_LOCATION_TWO_SHOT_mixtral8x7b,
        location=location)
    t2 = time.time()
    print(f"Claim: {claim_id}, Target sentence generation module is run in {(t2-t1)/60:.2f} mint")
    print(target_sent)

    