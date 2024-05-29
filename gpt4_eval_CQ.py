#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun March 24 05:19:45 2024

@author: soumensmacbookair
"""

import json, ast
import os, time
import json
from prompts import ablation_prompts
from utils import (
    chatgpt_prompt,
    LLM_CQ_generation
)
import pandas as pd

path = "/root/RnD_Project/outputs/mixtral_ablation_results/"

def evaluate_target_sent_by_common_ques(claim, gpt4_prompt):
    
    if gpt4_prompt == "gpt4_E1":
        prompt = ablation_prompts.CQ_EVAL_PROMPT_gpt4_E1
    elif gpt4_prompt == "gpt4_E2":
        prompt = ablation_prompts.CQ_EVAL_PROMPT_gpt4_E2
    elif gpt4_prompt == "gpt4_E2R":
        prompt = ablation_prompts.CQ_EVAL_PROMPT_gpt4_E2R
    
    prompt = prompt.format(target_location=claim["target_location"],
                           target_claim=claim[f"target_claim"],
                           common_ques=claim["common_ques"])
    
    response = chatgpt_prompt.chat_gpt(prompt, model="gpt-4-turbo")
    response = response.lower().replace("score:","").strip()
    
    try:
        score = int(response)
    except ValueError:
        score = -1
    
    return score

def get_eval_score(result_path, cq_model, ts_model, gpt4_prompt):
    """cq_model=gpt_corrected or mixtral_corrected"""
    
    # Read the results.json file
    with open(result_path, 'r') as json_file:
        res_data = json.load(json_file)
    
    with open(path+f"common_ques_{cq_model}.json", "r") as f:
        cq_data = json.load(f)
    
    cq_ref_claim_list = [j["ref_claim"] for j in cq_data]
    
    score_list = []
    output_list = []
    for i, elem in enumerate(res_data):
        idx = cq_ref_claim_list.index(elem["claim_ref"])
        for cq in cq_data[idx]["common_ques"]:
            claim = {"target_claim": elem["claim_target"],
                "target_location": elem["location"],
                "common_ques": cq}
            score = evaluate_target_sent_by_common_ques(claim=claim, gpt4_prompt=gpt4_prompt)
            if score != -1:
                score_list.append(score)
                
            output_list.append({**claim, **{"score": score}})
        
        # if i == 2: break
        print(f"cq_model: {cq_model}, claim: {i} done")
    
    avg_score = sum(score_list)/len(score_list)
    output_list.append({"cq_score_sum": sum(score_list), "cq_count": len(score_list)})
    output_list.append({"cq_correctness_avg": avg_score})
    
    with open(path+f"cq_by_{cq_model}_{ts_model}_{gpt4_prompt}.json", 'w') as json_file:
        json.dump(output_list, json_file, indent=4)
        
    return avg_score

if __name__ == "__main__":
    
    cq_model = "mixtral_corrected"
    quant_list = ["Q4", "Q6"]
    initial_prompt_list = ["initial_A", "initial_B"]
    shot_example_prompt_list = ["static_0_shot" ,"example_T1_static_3_shot",
                           "example_T2_static_3_shot", "dynamic_1_shot",
                           "dynamic_2_shot"]
    gpt4_prompt_list = ["gpt4_E1", "gpt4_E2", "gpt4_E2R"]
    
    q = quant_list[1]
    i = initial_prompt_list[1]
    s = shot_example_prompt_list[0]
    g = gpt4_prompt_list[2]
    
    result_path = f"/root/RnD_Project/outputs/mixtral_ablation_results/target_sent_results_mixtral_{q}_{i}_{s}.json"

    cq_score = get_eval_score(result_path=result_path, cq_model=cq_model, ts_model=f"{q}_{i}_{s}", gpt4_prompt=g)
    print(f"cq_score for cq_model: {cq_model} and target_sent_model: {q}_{i}_{s} and gpt4_prompt: {g} is: {cq_score}")
  