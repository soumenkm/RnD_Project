"""Utils for running the editor."""
import os
import time
from typing import Dict, Union
import os
import time
from typing import List
from llama_cpp import Llama

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

EDITOR_PROMPT = """I will fix some things you said.

1. You said: Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes. This is to prevent a buildup of mucus. It’s called the nasal cycle.
2. I checked: How often do your nostrils switch?
3. I found this article: Although we don’t usually notice it, during the nasal cycle one nostril becomes congested and thus contributes less to airflow, while the other becomes decongested. On average, the congestion pattern switches about every 2 hours, according to a small 2016 study published in the journal PLOS One.
4. This suggests 45 minutes switch time in your statement is wrong.
5. My fix: Your nose switches back and forth between nostrils. When you sleep, you switch about every 2 hours. This is to prevent a buildup of mucus. It’s called the nasal cycle.

1. You said: In the battles of Lexington and Concord, the British side was led by General Thomas Hall.
2. I checked: Who led the British side in the battle of Lexington and Concord?
3. I found this article: Interesting Facts about the Battles of Lexington and Concord. The British were led by Lieutenant Colonel Francis Smith. There were 700 British regulars.
4. This suggests General Thomas Hall in your statement is wrong.
5. My fix: In the battles of Lexington and Concord, the British side was led by Lieutenant Colonel Francis Smith.

1. You said: The Stanford Prison Experiment was conducted in the basement of Encina Hall, Stanford’s psychology building.
2. I checked: Where was Stanford Prison Experiment conducted?
3. I found this article: Carried out August 15-21, 1971 in the basement of Jordan Hall, the Stanford Prison Experiment set out to examine the psychological effects of authority and powerlessness in a prison environment.
4. This suggests Encina Hall in your statement is wrong.
5. My fix: The Stanford Prison Experiment was conducted in the basement of Jordan Hall, Stanford’s psychology building.

1. You said: The Havel-Hakimi algorithm is an algorithm for converting the adjacency matrix of a graph into its adjacency list. It is named after Vaclav Havel and Samih Hakimi.
2. I checked: What is the Havel-Hakimi algorithm?
3. I found this article: The Havel-Hakimi algorithm constructs a special solution if a simple graph for the given degree sequence exists, or proves that one cannot find a positive answer. This construction is based on a recursive algorithm. The algorithm was published by Havel (1955), and later by Hakimi (1962).
4. This suggests the Havel-Hakimi algorithm’s functionality in your statement is wrong.
5. My fix: The Havel-Hakimi algorithm constructs a special solution if a simple graph for the given degree sequence exists, or proves that one cannot find a positive answer. It is named after Vaclav Havel and Samih Hakimi.

1. You said: "Time of My Life" is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Phil Ramone.
2. I checked: Who was the producer of "(I’ve Had) The Time of My Life"?
3. I found this article: On September 8, 2010, the original demo of this song, along with a remix by producer Michael Lloyd , was released as digital files in an effort to raise money for the Patrick Swayze Pancreas Cancer Resarch Foundation at Stanford University.
4. This suggests "Time of My Life" producer name in your statement is wrong.
5. My fix: "Time of My Life" is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Michael Lloyd.

1. You said: Phoenix Market City Pune is located on 21 acres of prime property in Pune. It is spread across four levels with approximately 1.4 million square feet of built-up space. The mall is owned and operated by Phoenix Mills Limited.
2. I checked: What is the area of Phoenix Market City in Pune?
3. I found this article: Phoenix Market City was opened in January 2013 and has the distinction of being the largest mall in the city of Pune, with the area of 3.4 million square feet. It is located in the Viman Nagar area of Pune.
4. This suggests the 1.4 million square feet of built-up space in your statment is wrong.
5. My fix: Phoenix Market City Pune is located on 21 acres of prime property in Pune. It is spread across four levels with approximately 3.4 million square feet of built-up space. The mall is owned and operated by Phoenix Mills Limited.

1. You said: {claim}
2. I checked: {query}
3. I found this article: {evidence}
4. This suggests
""".strip()

def prompt_model(model, prompt):
    output = model(prompt, max_tokens=256)
    return output['choices'][0]['text']

def parse_api_response(api_response: str) -> str:
    """Extract the agreement gate state and the reasoning from the GPT-3 API response.

    Our prompt returns a reason for the edit and the edit in two consecutive lines.
    Only extract out the edit from the second line.

    Args:
        api_response: Editor response from GPT-3.
    Returns:
        edited_claim: The edited claim.
    """
    api_response = api_response.strip().split("\n")
    if len(api_response) < 2:
        print("Editor error.")
        return None
    edited_claim = api_response[1].split("My fix:")[-1].strip()
    return edited_claim

def run_rarr_editor(
    claim: str,
    query: str,
    evidence: str,
    model: str,
    prompt: str,
    num_retries: int = 5,
) -> Dict[str, str]:
    """Runs a GPT-3 editor on the claim given a query and evidence to support the edit.

    Args:
        claim: Text to edit.
        query: Query to guide the editing.
        evidence: Evidence to base the edit on.
        model: Name of the OpenAI GPT-3 model to use.
        prompt: The prompt template to query GPT-3 with.
        num_retries: Number of times to retry OpenAI call in the event of an API failure.
    Returns:
        edited_claim: The edited claim.
    """
    llm_input = prompt.format(claim=claim, query=query, evidence=evidence).strip()

    llm = Llama(
        model_path="/root/llama.cpp/models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",  
        n_ctx=32768,  # The max sequence length to use - note that longer sequence lengths require much more resources
        n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
        n_gpu_layers=35 ,        # The number of layers to offload to GPU, if you have GPU acceleration available
        verbose=False
        ) 

    for _ in range(num_retries):
        response = prompt_model(model = llm, prompt = llm_input)

    edited_claim = parse_api_response(response.strip())
    
    # If there was an error in GPT-3 generation, return the claim.
    if not edited_claim:
        edited_claim = claim
    output = {"text": edited_claim}
    
    return output

if __name__ == "__main__":

    edited_claim = run_rarr_editor(
        claim = "Michael Jordan played for the LA Lakers.",
        query = "Is it true that Michael Jordan played for the LA Lakers?",
        evidence = "Michael Jordan is one of the most popular basketball players in NBA history. During his 15-year professional career, he played for the Chicago Bulls and Washington Wizards, recording and breaking several records along the way.",
        model = "mixtral8x7b",
        context = None,
        prompt = EDITOR_PROMPT,
        num_retries = 1,
    )
