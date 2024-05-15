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
    LLM_verify_target_sent,
    LLM_edit_claim
)
import pandas as pd
from pathlib import Path

path_name = "/root/RnD_Project/outputs/comparison_mixtral_gpt/"

def get_mixtral_output(
    claim_id: int,
    claim: str,
    location: str,
    hyperlocality: int,
    prompt_name: str,
    model: str = "mixtral8x7b",
    temperature_qgen: float = 0.7):

    target_sent_gen_prompt = prompt_name
    prompt_target_sent = getattr(rarr_prompts, target_sent_gen_prompt)
    target_sent, reason_for_target_sent = LLM_target_sent_gen.run_rarr_target_sentence_generation(
        claim=claim,
        model=model,
        prompt=prompt_target_sent,
        location=location
    )
    output = {
        "claim_id": claim_id,
        "claim_ref": claim,
        "location": location,
        "hyperlocal score": hyperlocality,
        "model": model,
        "claim_target": target_sent,
        "reason_for_target_sent": reason_for_target_sent
    }
    return output

def write_results_json(data, results_output_path, prompt_name):
    num_claims = len(data)
    output_list = []
    for claim_id, item in enumerate(data):
        claim = data[claim_id]["input_info"]["claim"]
        hyperlocality = int(data[claim_id]["input_info"]["hyperlocality"])
        location = data[claim_id]["input_info"]["location"]
        t1 = time.time()
        output_list.append(get_mixtral_output(claim_id, claim, location, hyperlocality, prompt_name))
        t2 = time.time()
        print(f"Claim {claim_id} done")

    # Dump the list into the JSON file
    with open(results_output_path, 'w') as json_file:
        json.dump(output_list, json_file, indent=4)
                        
if __name__ == "__main__":
    eval_data_df = pd.read_csv("/root/RnD_Project/inputs/Amazon RnD_ Evaluation Dataset - sample 100.csv")
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
    
    shot = "ZERO_SHOT" # "ZERO_SHOT", "ONE_SHOT", "THREE_SHOT", "FEW_SHOT"
        
    results_output_path = path_name + f"results_mixtral_{shot.lower()}.json"
    prompt_name = f"TARGET_SENT_GEN_PROMPT_WITH_LOCATION_{shot}_mixtral8x7b"
    write_results_json(data, results_output_path, prompt_name)

