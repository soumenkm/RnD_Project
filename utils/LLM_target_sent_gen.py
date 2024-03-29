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
        Target sentence and reasons
    """
    Sent_search_string = "Target sentence:"
    Reason_search_string = "Reason:"
    sentence = ""
    for response in api_response.split("\n"):
        if Sent_search_string in response:
            sentence = response.split(Sent_search_string)[1].strip()
        elif Reason_search_string in response:
            reason = response.split(Reason_search_string)[1].strip()

    return sentence, reason

def run_rarr_target_sentence_generation(
    claim: str,
    model: str,
    prompt: str,
    entity: str = None,
    location: str = None,
) -> List[str]:
    """Generates target sentence based on target location in a claim.

    Given a piece of text (claim), we use Mixtral to generate target sentence.

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
    
    if entity:
        llm_input = prompt.format(entity=entity, claim=claim).strip()
    elif location:
        llm_input = prompt.format(location=location, claim=claim).strip()
    else:
        llm_input = prompt.format(claim=claim).strip()
        
    # print("\nResponse: ")
    
    response = prompt_model(model = llm, prompt = llm_input)
    target_sent, reason = parse_api_response(response.strip())

    return target_sent, reason


if __name__ == "__main__":

    data = [{"input_info": 
    {"claim": "Angelina Jolie is an American actress, filmmaker and humanitarian. The recipient of numerous accolades, including an Academy Award and three Golden Globe Awards, she has been named Hollywood's highest-paid actress multiple times.", 
    "location": "Maharashtra"}},
    {"input_info": 
    {"claim": "Dwayne Douglas Johnson, also known by his ring name the Rock, is an American actor and professional wrestler currently signed to WWE.", 
    "location": "Kerala"}},
    {"input_info": {"claim": "Robert John Downey Jr. is an American actor. His career has been characterized by critical success in his youth, followed by a period of substance abuse and legal troubles, and a surge in popular and commercial success later in his career.", 
    "location": "West Bengal"}}]

    claim_id = 0
    claim = data[claim_id]["input_info"]["claim"]
    entity = None
    location = data[claim_id]["input_info"]["location"]

    t1 = time.time()
    target_sent, reason = run_rarr_target_sentence_generation(
        claim=claim,
        model="mixtral8x7b",
        prompt=rarr_prompts.TARGET_SENT_GEN_PROMPT_WITH_LOCATION_mixtral8x7b,
        entity=entity,
        location=location)
    t2 = time.time()
    print(f"Claim: {claim_id}, Target sentence generation module is run in {(t2-t1)/60:.2f} mint")
    print(target_sent)
    print(reason)

    