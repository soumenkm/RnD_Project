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
from prompts import ablation_prompts
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

target_location_list = ["India", "Mumbai", "Kerala", "Tamil Nadu", "Maharashtra",
                        "West Bengal", "Kolkata", "Delhi", "Karnataka", "Bengaluru",
                        "Chennai", "Andhra Pradesh", "New Delhi", "Gujrat", "Punjab",
                        "Rajasthan", "Hyderabad", "Assam", "Uttar Pradesh", "Goa", 
                        "Uttarakhand", "Pune", "Haryana", "Ahmedabad", "Madhya Pradesh"]

ref_entity_list = [
    "ford motor", "mississippi river",
    "breaking bad", "wall street",
    "ny1", "universal pictures",
    "klamath mountains", "mario cuomo",
    "aretha franklin", "costco",
    "disney", "wtop fm",
    "alberto colombo", "avenged sevenfold",
    "eiffel tower", "stephen king",
    "christopher wilkins", "buckingham palace",
    "bill gates", "changi airport",
    "alan gilbert", "ami vitale",
    "new year", "san diego zoo",
    "king of prussia mall", "sheikh zayed grand mosque",
    "bali kite festival", "bam earthquake",
    "samba", "seattle",
    "the manhattan project", "the driskill",
    "star tv", "clara hughes",
    "everglades national park", "elizabeth freeman",
    "atomic bomb dome", "david bowie",
    "national library of china", "margarita",
    "tiwa savage", "adirondack mountains",
    "bayer", 'chicago "l" train',
    "assistance publique hôpitaux de paris", "rise up coffee",
    "the waltz", "mundipharma",
    "mogao caves", "marble"
]

path_name = "/root/RnD_Project/outputs/mixtral_ablation_results/"

def get_mixtral_output(
    claim_id: int,
    claim: str,
    location: str,
    hyperlocality: int,
    quant: str, 
    initial_prompt: str, 
    shot_example_prompt: str,
    model: str = "mixtral8x7b",
    temperature_qgen: float = 0.7):

    if "dynamic" in shot_example_prompt:
        if location in target_location_list: 
            target_sent_gen_prompt = f"TS_PROMPT_{initial_prompt}_{shot_example_prompt}"
            prompt_target_sent = getattr(ablation_prompts, target_sent_gen_prompt)
            target_sent, reason_for_target_sent, count_local = LLM_target_sent_gen.run_rarr_dynamic_target_sentence_generation(
                quant=quant,
                claim=claim,
                model=model,
                prompt=prompt_target_sent,
                prompt_name=target_sent_gen_prompt,
                location=location
            )
        else:
            target_sent_gen_prompt = f"TS_PROMPT_{initial_prompt}_example_T1_static_3_shot"
            prompt_target_sent = getattr(ablation_prompts, target_sent_gen_prompt)
            target_sent, reason_for_target_sent, count_local = LLM_target_sent_gen.run_rarr_target_sentence_generation(
                quant=quant,
                claim=claim,
                model=model,
                prompt=prompt_target_sent,
                location=location
            )
    elif "static" in shot_example_prompt:
        target_sent_gen_prompt = f"TS_PROMPT_{initial_prompt}_{shot_example_prompt}"
        prompt_target_sent = getattr(ablation_prompts, target_sent_gen_prompt)
        target_sent, reason_for_target_sent, count_local = LLM_target_sent_gen.run_rarr_target_sentence_generation(
            quant=quant,
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
        "reason_for_target_sent": reason_for_target_sent,
        "num_location_in_prompt": count_local-1,
        "quant": quant,
        "prompt_name": target_sent_gen_prompt 
    }
    return output

def write_results_json(data, quant, initial_prompt, shot_example_prompt):
    
    num_claims = len(data)
    output_list = []
    
    for claim_id, item in enumerate(data):
        claim = data[claim_id]["input_info"]["claim"]
        hyperlocality = int(data[claim_id]["input_info"]["hyperlocality"])
        location = data[claim_id]["input_info"]["location"]
        t1 = time.time()
        output_list.append(get_mixtral_output(claim_id, claim, location, hyperlocality, quant, initial_prompt, shot_example_prompt))
        t2 = time.time()
        print(f"Claim {claim_id} done")

    # Dump the list into the JSON file
    results_output_path = path_name + f"target_sent_results_mixtral_{quant}_{initial_prompt}_{shot_example_prompt}.json"
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
    
    quant_list = ["Q4", "Q6"]
    initial_prompt_list = ["initial_A", "initial_B"]
    shot_example_prompt_list = ["static_0_shot" ,"example_T1_static_3_shot",
                           "example_T2_static_3_shot", "dynamic_1_shot",
                           "dynamic_2_shot"]
    
    q = quant_list[1]
    i = initial_prompt_list[0]
    s = shot_example_prompt_list[1]
    
    write_results_json(data, quant=q, initial_prompt=i, shot_example_prompt=s)
    print(f"{q}, {i}, {s} done")
