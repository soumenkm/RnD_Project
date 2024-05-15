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
    
    response = chatgpt_prompt.chat_gpt(prompt)
    response = response.lower().replace("score:","").strip()
    
    try:
        score = int(response)
    except ValueError:
        score = -1
    
    return score

def get_eval_score(res_data, model):
    """model=mixtral or gpt or mixtral_corrected"""
    
    with open(path+f"common_ques_{model}.json", "r") as f:
        cq_data = json.load(f)
    
    cq_ref_claim_list = [j["ref_claim"] for j in cq_data]
    
    score_list = []
    for i, elem in enumerate(res_data):
        idx = cq_ref_claim_list.index(elem["claim_ref"])
        for cq in cq_data[idx]["common_ques"]:
            claim = {"target_claim": elem["claim_target"],
                "target_location": elem["location"],
                "common_ques": cq}
            score = evaluate_target_sent_by_common_ques(claim=claim)
            if score != -1:
                score_list.append(score)
        
        print(f"model: {model}, claim: {i} done")
    
    return sum(score_list)/len(score_list)

if __name__ == "__main__":
        
    # Read the results.json file
    with open("/root/RnD_Project/outputs/mixtral_zero_shot/1/outputs_0_100/results.json", 'r') as json_file:
        res_data_1 = json.load(json_file)
    with open("/root/RnD_Project/outputs/mixtral_zero_shot/1/outputs_100_200/results.json", 'r') as json_file:
        res_data_2 = json.load(json_file)
    
    res_data = res_data_1 + res_data_2 
    
    mixtral_score = get_eval_score(res_data=res_data, model="mixtral_corrected")
    # gpt_score = get_eval_score(res_data=res_data, model="gpt")
    
    print(f"mixtral_score: {mixtral_score}")
  
    
    
