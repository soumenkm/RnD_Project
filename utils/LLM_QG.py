"""Utils for running question generation."""
import os
import time
from typing import List
from llama_cpp import Llama

def prompt_model(model, prompt):
    output = model(prompt, max_tokens=256)
    return output['choices'][0]['text']

def parse_api_response(api_response: str) -> List[str]:
    """Extract questions from the GPT-3 API response.

    Our prompt returns questions as a string with the format of an ordered list.
    This function parses this response in a list of questions.

    Args:
        api_response: Question generation response from GPT-3.
    Returns:
        questions: A list of questions.
    """
    Sent_search_string = "Target sentence:"
    Q_search_string = "Q:"
    sentence = ""
    questions = []
    for response in api_response.split("\n"):
        # Remove the search string from each question
        if Sent_search_string in response:
            sentence = response.split(Sent_search_string)[1].strip()
        elif Q_search_string in response:
            question = response.split(Q_search_string)[1].strip()
            questions.append(question)

    return sentence, questions

def run_rarr_question_generation(
    claim: str,
    model: str,
    prompt: str,
    temperature: float,
    num_rounds: int,
    entity: str = None,
    location: str = None,
    num_retries: int = 1,
) -> List[str]:
    """Generates questions that interrogate the information in a claim.

    Given a piece of text (claim), we use GPT-3 to generate questions that question the
    information in the claim. We run num_rounds of sampling to get a diverse set of questions.

    Args:
        claim: Text to generate questions off of.
        model: Name of the OpenAI GPT-3 model to use.
        prompt: The prompt template to query GPT-3 with.
        temperature: Temperature to use for sampling questions. 0 represents greedy deconding.
        num_rounds: Number of times to sample questions.
    Returns:
        questions: A list of questions.
    """
    
    if(model == "mixtral8x7b"):
        llm = Llama(
        model_path="/raid/speech/soumenmondal/llama.cpp/models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",  
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
    
    if entity:
        llm_input = prompt.format(entity=entity, claim=claim).strip()
    elif location:
        llm_input = prompt.format(location=location, claim=claim).strip()
    else:
        llm_input = prompt.format(claim=claim).strip()
        
    # print("\nResponse: ")
    
    questions = set()
    for _ in range(num_rounds):
        for _ in range(num_retries):
            response = prompt_model(model = llm, prompt = llm_input)
            sent, cur_round_questions = parse_api_response(response.strip())
            questions.update(cur_round_questions)

    questions = list(sorted(questions))
    # print("Target Sentence: ",sent)
    # print("Questions: ", questions)
    
    return sent, questions