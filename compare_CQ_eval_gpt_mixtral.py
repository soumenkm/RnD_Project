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

path = "/root/RnD_Project/outputs/comparison_mixtral_gpt/"

def evaluate_target_sent_by_common_ques(claim):
    
    prompt = rarr_prompts.EVAL_BY_SINGLE_COMMON_QUES_PROMPT
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

def get_eval_score(result_path, model):
    """model=gpt_corrected or mixtral_corrected"""
    
    # Read the results.json file
    with open(result_path, 'r') as json_file:
        res_data = json.load(json_file)
    
    with open(path+f"common_ques_{model}.json", "r") as f:
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
            score = evaluate_target_sent_by_common_ques(claim=claim)
            if score != -1:
                score_list.append(score)
                
            output_list.append({**claim, **{"score": score}})
        
        print(f"model: {model}, claim: {i} done")
    
    avg_score = sum(score_list)/len(score_list)
    output_list.append({"cq_score_sum": sum(score_list), "cq_count": len(score_list)})
    output_list.append({"cq_correctness_avg": avg_score})
    
    target_sent_model = "_".join(result_path.split("_")[-3:])
    with open(path+f"cq_by_{model}_{target_sent_model}", 'w') as json_file:
        json.dump(output_list, json_file, indent=4)
        
    return avg_score

if __name__ == "__main__":
    
    cq_model = ["gpt_corrected", "mixtral_corrected"]
    target_sent_model = ["gpt_zero_shot", "gpt_one_shot", "gpt_three_shot", "gpt_few_shot",
                         "mixtral_zero_shot", "mixtral_one_shot", "mixtral_three_shot", "mixtral_few_shot"]
    
    result_path_list = []
    for i in target_sent_model:
        result_path_list.append(f"/root/RnD_Project/outputs/comparison_mixtral_gpt/results_{i}.json")
    
    for i in cq_model:
        for j in result_path_list:
            cq_score = get_eval_score(result_path=j, model=i)
            print(f"cq_score for cq_model: {i} and target_sent_model: {j} is: {cq_score}")
  
    
    
