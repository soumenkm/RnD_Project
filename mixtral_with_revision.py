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
    LLM_edit_claim
)
import pandas as pd
from pathlib import Path

number = str(1)
output_name = "100_200"

path_name = "/root/RnD_Project/mixtral_zero_shot_revised/"+number+"/"
results_output_path = path_name + "outputs_"+output_name+"/results.json"

def get_mixtral_revision(
    claim_id: int,
    claim: str,
    location: str,
    hyperlocality: int,
    common_ques: None,
    target_sent: str,
    model: str = "mixtral8x7b",
    temperature_qgen: float = 0.7):
    
    score_list = []
    target_sent_revised = target_sent
    reason_for_target_sent = ""
    for ques_n in common_ques.keys():
        # prompt for the common ques
        target_sent_check_prompt = "TARGET_SENT_CHECK_PROMPT_WITH_LOCATION_"+model
        prompt_target_sent_check = getattr(rarr_prompts, target_sent_check_prompt)
    
        # run mixtral model
        score, reason = LLM_check_target_sent.run_rarr_target_sentence_cq(
            claim=claim,
            model=model,
            prompt=prompt_target_sent_check,
            target_sent= target_sent_revised,
            location=location,
            question= common_ques[ques_n]
        )
        score_dict = {
        "question": common_ques[ques_n],
        "score": score,
        "reason": reason
        }
        score_list.append(score_dict)
        
        if score != 1:
            target_sent_regen_prompt = "TARGET_SENT_REGEN_PROMPT_WITH_LOCATION_ZERO_SHOT_"+model
            prompt_target_sent_regen = getattr(rarr_prompts, target_sent_regen_prompt)
            target_sent_revised, reason = LLM_target_sent_gen.run_rarr_target_sentence_regeneration(
                claim=claim,
                model=model,
                prompt=prompt_target_sent_regen,
                target_sent= target_sent_revised,
                location=location,
                question= common_ques[ques_n]
            )
            reason_for_target_sent =  reason_for_target_sent + "; " +reason
    
    output = {
        "claim_id": claim_id,
        "claim_ref": claim,
        "location": location,
        "hyperlocal score": hyperlocality,
        "model": model,
        "claim_target": target_sent,
        "common_questions": common_ques,
        "score": score_list,
        "claim_target_revised": target_sent_revised,
        "reason_for_revision": reason_for_target_sent
    }
    return output

def write_results_json(data, mixtral_data):
    num_claims = len(data)
    output_list = []
    for claim_id, item in enumerate(data):
        claim = data[claim_id]["input_info"]["claim"]
        hyperlocality = int(data[claim_id]["input_info"]["hyperlocality"])
        location = data[claim_id]["input_info"]["location"]
        common_ques = data[claim_id]["input_info"]["common_ques"]
        
        for j,item in enumerate(mixtral_data):
            if data[claim_id]["input_info"]["claim"].strip() == item["claim_ref"].strip():
                if data[claim_id]["input_info"]["location"].strip() == item["location"].strip():
                    target_sent = item["claim_target"].strip()
                    
        print("Claim: ", claim_id)
        t1 = time.time()
        output_list.append(get_mixtral_revision(claim_id, claim, location, hyperlocality, common_ques, target_sent))
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
        common_ques = eval_data_df.loc[i, "Correct Common Questions"]
        if str(eval_data_df.loc[i, "Reference Sentence"]).lower() != "nan":
            common_ques = common_ques.replace("(i)", "|")
            common_ques = common_ques.replace("(ii)", "|")
            common_ques = common_ques.replace("(iii)", "|")
            common_ques = common_ques.replace("(iv)", "|")
            common_ques = common_ques.replace("(v)", "|")
            
            common_ques_list = common_ques.split("|")
            common_ques_dict = {}
            for j,k in enumerate(common_ques_list[1:]):
                common_ques_dict[f"ques_{j}"] = k.strip()
            
            elem_dict = {"input_info": 
                {"claim": eval_data_df.loc[i, "Reference Sentence"], 
                "hyperlocality": eval_data_df.loc[i, "Hyperlocal Score"],
                "location": eval_data_df.loc[i, "Target Location"],
                "common_ques": common_ques_dict}}
            data.append(elem_dict)
        else:
            continue
        
    with open("/root/RnD_Project/mixtral_zero_shot/"+number+"/outputs_0_100/results.json", 'r') as json_file:
        mixtral_data_1 = json.load(json_file)
    with open("/root/RnD_Project/mixtral_zero_shot/"+number+"/outputs_100_200/results.json", 'r') as json_file:
        mixtral_data_2 = json.load(json_file)
    mixtral_data = mixtral_data_1 + mixtral_data_2
       
    write_results_json(data[100:200], mixtral_data)
