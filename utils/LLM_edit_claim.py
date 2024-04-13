"""Utils for attributing the target sentence."""
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
    """Attribute the target sentence from the Mixtral response.

    Args:
        api_response: Target sentence generation response from Mixtral.
    Returns:
        Target sentence and reasons
    """
    Sent_search_string = "Attributed Claim:"
    sentence = ""
    for response in api_response.split("\n"):
        if Sent_search_string in response:
            sentence = response.split(Sent_search_string)[1].strip()

    return sentence

def edit_target_sentence(
    claim: str,
    evidences: list,
    model: str,
    prompt: str,
) -> List[str]:
    """Attributes target sentence based on evidences.

    Given a piece of text (claim), we use Mixtral to attribute it using evidences.

    Args:
        claim: Text to attribute.
        model: Name of the Mixtral model to use.
        prompt: The prompt template to query Mixtral with.
    Returns:
        edited_claim: Attributed claim.
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
    
    llm_input = prompt.format(evidences=evidences, 
            claim=claim).strip()
    
    response = prompt_model(model = llm, prompt = llm_input)
    attributed_target_sent = response.strip()

    return attributed_target_sent


if __name__ == "__main__":

    # target_sent = "Rajinikanth, often referred to as Thalaiva, is an Indian actor and cultural icon primarily working in Tamil cinema, known for his unique style and larger-than-life characters."

    target_sent = "Shah Rukh Khan, often referred to as the King of Bollywood, is an Indian actor and film producer primarily working in Hindi cinema, recognized for his charm and romantic roles."

    evidence_list = [
        
    ]
    
    evidences = []
    for i,e in enumerate(evidence_list):
        evidences.append(f"{i+1}. {e}")
    
    t1 = time.time()
    attributed_target_sent = edit_target_sentence(
        claim=target_sent,
        evidences=evidences,
        model="mixtral8x7b",
        prompt=rarr_prompts.SINGLE_EDITOR_PROMPT
    )
    t2 = time.time()
    print(f"Editor module is run in {(t2-t1)/60:.2f} mint")
    print(f"Attributed target sentence: {attributed_target_sent}")



    