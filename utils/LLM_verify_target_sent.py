 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri March 29 14:40:53 2024

@author: soumensmacbookair
"""

"""Utils for verifying target sentence generation."""
import os, sys
import time
from typing import List
from llama_cpp import Llama
sys.path.append("/root/RnD_Project/prompts")
sys.path.append("/root/RnD_Project/utils")
import rarr_prompts
import LLM_target_sent_gen

def prompt_model(model, prompt):
    prompt = "[INST]" + prompt + "[/INST]"
    output = model(prompt, max_tokens=256)
    return output['choices'][0]['text']

def parse_api_response(api_response: str) -> List[str]:
    """Extract decision, reason and correct target sentence from the Mixtral response.

    Args:
        api_response: Target sentence generation response from Mixtral.
    Returns:
        Target sentence and reasons
    """
    Decision_search_string = "Decision:"
    Sent_search_string = "Correct target sentence:"
    Reason_search_string = "Reasons:"
    sentence = ""
    for response in api_response.split("\n"):
        if Decision_search_string in response:
            decision = response.split(Decision_search_string)[1].strip()
        elif Reason_search_string in response:
            reason = response.split(Reason_search_string)[1].strip()
        elif Sent_search_string in response:
            sentence = response.split(Sent_search_string)[1].strip()

    return decision, reason, sentence

def verify_rarr_target_sentence(
    ref_claim: str,
    target_claim: str,
    target_location: str,
    model: str,
    prompt: str,
) -> List[str]:
    """Verifies target sentence based on target location in a claim.

    Given a piece of text (claim), we use Mixtral to verify entity in target sentence.

    Args:
        claim: Text to generate questions off of.
        model: Name of the OpenAI GPT-3 model to use.
        prompt: The prompt template to query GPT-3 with.
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
    else:
        print("Model not found!")
    
    llm_input = prompt.format(target_location=target_location, 
            ref_claim=ref_claim, target_claim=target_claim).strip()
    # print("\nResponse: ")
    
    response = prompt_model(model = llm, prompt = llm_input)
    # print(response)
    decision, reason, correct_target_sent = parse_api_response(response.strip())

    return decision, reason, correct_target_sent


if __name__ == "__main__":

    data = [{"input_info": 
    {"claim": "Angelina Jolie is an American actress, filmmaker and humanitarian. The recipient of numerous accolades, including an Academy Award and three Golden Globe Awards, she has been named Hollywood's highest-paid actress multiple times.", 
    "location": "Maharashtra"}},
    {"input_info": 
    {"claim": "Dwayne Douglas Johnson, also known by his ring name the Rock, is an American actor and professional wrestler currently signed to WWE.", 
    "location": "Kerala"}},
    {"input_info": {"claim": "Robert John Downey Jr. is an American actor. His career has been characterized by critical success in his youth, followed by a period of substance abuse and legal troubles, and a surge in popular and commercial success later in his career.", 
    "location": "West Bengal"}}]

    claim_id = 1
    claim = data[claim_id]["input_info"]["claim"]
    entity = None
    location = data[claim_id]["input_info"]["location"]

    target_sent = "Rajinikanth, often referred to as Thalaiva, is an Indian actor and cultural icon primarily working in Tamil cinema, known for his unique style and larger-than-life characters."

    # target_sent = " Shah Rukh Khan, often referred to as the King of Bollywood, is an Indian actor and film producer primarily working in Hindi cinema, recognized for his charm and romantic roles."

    t1 = time.time()
    decision, reason, correct_target_sent = verify_rarr_target_sentence(
        ref_claim=claim,
        target_claim=target_sent,
        target_location=location,
        model="mixtral8x7b",
        prompt=rarr_prompts.VERIFY_TARGET_ENTITY_PROMPT_mixtral8x7b,
        entity=entity)
    t2 = time.time()
    print(f"Claim: {claim_id}, Verify target sentence module is run in {(t2-t1)/60:.2f} mint")
    print(f"Decision: {decision}")
    print(f"Reason: {reason}")
    print(f"Correct target sentence: {correct_target_sent}")



    