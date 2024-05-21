#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun March 24 05:19:45 2024

@author: soumensmacbookair
"""

import json, ast
import os, time
import json
from prompts import rarr_prompts
from utils import (
    chatgpt_prompt,
    LLM_CQ_generation
)
import pandas as pd

path = "/root/RnD_Project/outputs/dynamic_prompt_results/"

def evaluate_target_sent_by_common_ques(claim, gpt4_prompt):
    
    if gpt4_prompt == "GPT4_E1":
        prompt = rarr_prompts.EVAL_BY_SINGLE_COMMON_QUES_PROMPT_GPT4_E1
    elif gpt4_prompt == "GPT4_E2":
        prompt = rarr_prompts.EVAL_BY_SINGLE_COMMON_QUES_PROMPT_GPT4_E2
    
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
    result_path = "/root/RnD_Project/outputs/dynamic_prompt_results/results_mixtral_Q6_dynamic_target_sent_gen.json"
    ts_model = "mixtral_Q6_dynamic"
    gpt4_prompt = "GPT4_E2"
    
    cq_score = get_eval_score(result_path=result_path, cq_model=cq_model, ts_model=ts_model, gpt4_prompt=gpt4_prompt)
    print(f"cq_score for cq_model: {cq_model} and target_sent_model: {ts_model} is: {cq_score}")
  