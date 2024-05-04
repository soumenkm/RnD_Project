#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sum March 24 05:19:45 2024

@author: soumensmacbookair
"""

import json, ast
import os, time
import json
from prompts import rarr_prompts
from utils import (
    chatgpt_prompt
)
import pandas as pd

number = str(1)
path = "/root/RnD_Project/outputs/rarr_mixtral_few_shot/"+number+"/outputs_"

def evaluate_target_sent_by_common_ques(claim, model):
    """model = rarr or mixtral"""
    prompt = rarr_prompts.EVAL_BY_COMMON_QUES_PROMPT
    prompt = prompt.format(target_location=claim["target_location"],
                           target_claim=claim[f"target_claim_{model}"],
                           common_ques=claim["common_ques"])
    
    response = chatgpt_prompt.chat_gpt(prompt)
    response = response.lower().replace("scores:","").strip()
    response_dict = ast.literal_eval(response.strip())
    return response_dict, list(response_dict.values())

if __name__ == "__main__":
    
    eval_data_df = pd.read_csv("/root/RnD_Project/inputs/revised_final_dataset_200.csv")
    eval_data = []
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
                {"ref_claim": eval_data_df.loc[i, "Reference Sentence"], 
                "target_location": eval_data_df.loc[i, "Target Location"],
                "common_ques": common_ques_dict}}
            eval_data.append(elem_dict)
        else:
            continue
        
    # Read the results.json file
    with open(path+"0_50/results.json", 'r') as json_file:
        res_data_1 = json.load(json_file)
    with open(path+"50_100/results.json", 'r') as json_file:
        res_data_2 = json.load(json_file)
    with open(path+"100_200/results.json", 'r') as json_file:
        res_data_3 = json.load(json_file)
    res_data = res_data_1 + res_data_2 +res_data_3

    claim_data = []
    for i,elem in enumerate(eval_data):
        for j,item in enumerate(res_data):
            if elem["input_info"]["ref_claim"].strip() == item["claim_ref"].strip():
                if elem["input_info"]["target_location"].strip() == item["location"].strip():
                    elem_dict = {**elem["input_info"], **{
                        "target_claim_mixtral": item["claim_original"],
                        "target_claim_rarr": item["claim_attributed"]}
                    }
                    claim_data.append(elem_dict)

    # index=0
    # df = pd.read_csv('chatgpt.csv')
    # sent= df["chatgpt"]
    # for i,elem in enumerate(eval_data):
    #     elem_dict = {**elem["input_info"], **{
    #                 "target_claim_mixtral": "",
    #                 "target_claim_rarr": sent[index]}
    #             }
    #     index += 1
    #     if(index ==49):
    #         break
    #     claim_data.append(elem_dict)
    
    output_list = []
    rarr_list = []
    mixtral_list = []
    
    for i,item in enumerate(claim_data):
        for _ in range(2):  # try 3 times
            try:
                res_dict_mixtral, mixtral_val = evaluate_target_sent_by_common_ques(item, "mixtral")
                break       # as soon as it works, break out of the loop
            except Exception as e:
                print(e)
                continue    
        for _ in range(2):  # try 3 times
            try:
                res_dict_rarr, rarr_val = evaluate_target_sent_by_common_ques(item, "rarr")
                break       # as soon as it works, break out of the loop
            except Exception as e:
                print(e)
                continue    
        

        output_list.append({**item, 
                            **{"mixtral": res_dict_mixtral}, 
                            **{"rarr": res_dict_rarr}})
        rarr_list.extend(rarr_val)
        mixtral_list.extend(mixtral_val)
        print(f"Claim {i} done")
    
    rarr_metric = sum(rarr_list)/len(rarr_list)
    mixtral_metric = sum(mixtral_list)/len(mixtral_list)
    output_list.append({"rarr_count": sum(rarr_list), "mixtral_count": sum(mixtral_list)})
    output_list.append({"rarr_len": len(rarr_list), "mixtral_len": len(mixtral_list)})
    output_list.append({"rarr_metric": rarr_metric, "mixtral_metric": mixtral_metric})
    
    print("rarr", rarr_metric)
    print("mixtral", mixtral_metric)
    
    with open(path+"100_200/eval_results_1.json", 'w') as json_file:
        json.dump(output_list, json_file, indent=4)
    
    
    
