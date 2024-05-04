#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri April 12 06:46:37 2024

@author: soumensmacbookair
"""

import json
import os, time
from typing import Any, Dict, Tuple, List
import json
import jsonlines
import Levenshtein
import tqdm
from prompts import hallucination_prompts, rarr_prompts
from utils import (
    agreement_gate,
    editor,
    evidence_selection,
    hallucination,
    search,
    question_generation,
    LLM_QG,
    api,
    LLM_target_sent_gen,
    LLM_check_target_sent,
    LLM_edit_claim,
    chatgpt_prompt
)
import pandas as pd
from pathlib import Path

number = str(3)
output_name = "100_200"

path_name = "/root/RnD_Project/outputs/chatgpt3.5_few_shot/"+number+"/"
results_output_path = path_name + "outputs_"+output_name+"/results.json"

def get_chatgpt_output(
    claim_id: int,
    claim: str,
    location: str,
    hyperlocality: int):

    # target_sent_prompt = "TARGET_SENT_GEN_PROMPT_WITH_LOCATION_ZERO_SHOT_mixtral8x7b"
    target_sent_prompt = "TARGET_SENT_GEN_PROMPT_WITH_LOCATION_mixtral8x7b"
    prompt = getattr(rarr_prompts, target_sent_prompt)
    prompt = prompt.format(claim=claim,
                           location=location)
    response = chatgpt_prompt.chat_gpt(prompt)
    sent_search_string = "Target sentence:"
    reason_search_string = "Reason:"
    target_sent = ""
    reason = ""
    for text in response.split("\n"):
        if sent_search_string in text:
            target_sent = text.split(sent_search_string)[1].strip()
        if reason_search_string in text:
            reason = text.split(reason_search_string)[1].strip()
    output = {
        "claim_id": claim_id,
        "claim_ref": claim,
        "location": location,
        "hyperlocal score": hyperlocality,
        "claim_target": target_sent,
        "reason": reason
    }
    return output

def write_results_json(data):
    num_claims = len(data)
    output_list = []
    for claim_id, item in enumerate(data):
        claim = data[claim_id]["input_info"]["claim"]
        hyperlocality = int(data[claim_id]["input_info"]["hyperlocality"])
        location = data[claim_id]["input_info"]["location"]
        print("Claim: ", claim_id)
        t1 = time.time()
        output_list.append(get_chatgpt_output(claim_id, claim, location, hyperlocality))
        t2 = time.time()
    # Dump the list into the JSON file
    output_file = Path(results_output_path)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(results_output_path, 'w') as json_file:
        json.dump(output_list, json_file, indent=4)
                        
if __name__ == "__main__":
    eval_data_df = pd.read_csv("/root/RnD_Project/inputs/revised_final_dataset_200.csv")
    data = []
    for i in range(eval_data_df.shape[0]):
        if str(eval_data_df.loc[i, "Reference Sentence"]).lower() != "nan":
            elem_dict = {"input_info": 
                {"claim": eval_data_df.loc[i, "Reference Sentence"], 
                "hyperlocality": eval_data_df.loc[i, "Hyperlocal Score"],
                "location": eval_data_df.loc[i, "Target Location"]}}
            data.append(elem_dict)
        else:
            continue
    
    write_results_json(data[100:200])
