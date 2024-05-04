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

number = str(3)
output_name = "0_100"

path_name = "/root/RnD_Project/outputs/mixtral_zero_shot_QG_revised/"+number+"/"
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
            target_sent_regen_prompt = "TARGET_SENT_REGEN_PROMPT_WITH_LOCATION_"+model
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
            if(target_sent_revised == ""):
                target_sent_revised = target_sent
    
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

def get_mixtral_CQ(
    claim: str,
    target_sent: str,
    location: str,
    model: str = "mixtral8x7b",
    temperature_qgen: float = 0.7):

    cq_gen_prompt = "COMMON_QUESTION_GEN_PROMPT_"+model
    prompt_cq_gen = getattr(rarr_prompts, cq_gen_prompt)
    common_ques_dict = LLM_QG.run_common_ques_gen(
        claim=claim,
        model=model,
        prompt=prompt_cq_gen,
        target_sent= target_sent,
    )
    
    revised_common_ques_dict = {}
    count = 0
    for key in common_ques_dict.keys():
        check_cq_gen = "CHECK_IF_COMMON_QUESTION_PROMPT_"+model
        prompt_check_cq_gen = getattr(rarr_prompts, check_cq_gen)
        revised_common_ques = LLM_QG.revise_common_ques_gen(
            model=model,
            question = common_ques_dict[key],
            prompt=prompt_check_cq_gen,
        )
        revised_common_ques_dict['ques_'+str(count)] = revised_common_ques
        count +=1
        
    final_common_ques = {}
    count = 0
    for key in revised_common_ques_dict.keys():
        # prompt for the common ques
        target_sent_check_prompt = "TARGET_SENT_CHECK_PROMPT_WITH_LOCATION_"+model
        prompt_target_sent_check = getattr(rarr_prompts, target_sent_check_prompt)
        # run mixtral model
        score, reason = LLM_check_target_sent.run_rarr_target_sentence_cq(
            claim=claim,
            model=model,
            prompt=prompt_target_sent_check,
            target_sent= claim,
            location=location,
            question= revised_common_ques_dict[key]
        )
        if(score == 1):
            final_common_ques['ques_'+str(count)] = revised_common_ques_dict[key]
            count +=1
    
    return final_common_ques

def write_results_json(data, mixtral_data):
    num_claims = len(data)
    output_list = []
    for claim_id, item in enumerate(data):
        claim = data[claim_id]["input_info"]["claim"]
        hyperlocality = int(data[claim_id]["input_info"]["hyperlocality"])
        location = data[claim_id]["input_info"]["location"]
        ref_location = data[claim_id]["input_info"]["ref_location"]
        
        for j,item in enumerate(mixtral_data):
            if data[claim_id]["input_info"]["claim"].strip() == item["claim_ref"].strip():
                if data[claim_id]["input_info"]["location"].strip() == item["location"].strip():
                    target_sent = item["claim_target"].strip()
                    
        common_ques = get_mixtral_CQ(claim, target_sent, ref_location)
                    
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
        if str(eval_data_df.loc[i, "Reference Sentence"]).lower() != "nan":
            elem_dict = {"input_info": 
                {"claim": eval_data_df.loc[i, "Reference Sentence"], 
                "hyperlocality": eval_data_df.loc[i, "Hyperlocal Score"],
                "ref_location": eval_data_df.loc[i, "Reference Location"],
                "location": eval_data_df.loc[i, "Target Location"]}}
            data.append(elem_dict)
        else:
            continue
        
    with open("/root/RnD_Project/outputs/mixtral_zero_shot/"+number+"/outputs_0_100/results.json", 'r') as json_file:
        mixtral_data_1 = json.load(json_file)
    with open("/root/RnD_Project/outputs/mixtral_zero_shot/"+number+"/outputs_100_200/results.json", 'r') as json_file:
        mixtral_data_2 = json.load(json_file)
    mixtral_data = mixtral_data_1 + mixtral_data_2
    
    write_results_json(data[0:100], mixtral_data)
