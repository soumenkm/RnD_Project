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
    chatgpt_prompt
)
import pandas as pd

path = "/root/RnD_Project/outputs/comparison_mixtral_gpt/"

def revise_common_question_by_gpt(question):
    rev_prompt = rarr_prompts.CHECK_IF_COMMON_QUESTION_PROMPT_mixtral8x7b
    rev_prompt = rev_prompt.format(question=question)
    
    response = chatgpt_prompt.chat_gpt(rev_prompt)
    return response

def generate_common_question_by_gpt(claim):
    gen_prompt = rarr_prompts.COMMON_QUESTION_GEN_PROMPT_mixtral8x7b
    gen_prompt = gen_prompt.format(claim=claim)
    
    response = chatgpt_prompt.chat_gpt(gen_prompt)
    ques_list = response.split("\n")
    
    out_list = []
    for i in ques_list:
        question = i.replace("Q:","").strip()
        question = revise_common_question_by_gpt(question)
        out_list.append(question.split(":")[-1].strip())
    
    return out_list

def get_common_question_by_mixtral(eval_data_file_path):
    
    eval_data_df = pd.read_csv(eval_data_file_path)
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
            common_ques_list = [k.strip() for k in common_ques_list[1:]]
            
            elem_dict = {"ref_claim": eval_data_df.loc[i, "Reference Sentence"],
                "common_ques": common_ques_list}
            eval_data.append(elem_dict)
        else:
            continue
    
    return eval_data

if __name__ == "__main__":
    
    eval_data_file_path = "/root/RnD_Project/inputs/revised_final_dataset_200.csv"
    eval_data_df = pd.read_csv(eval_data_file_path)
    eval_data = []
    for i in range(eval_data_df.shape[0]):
        if str(eval_data_df.loc[i, "Reference Sentence"]).lower() != "nan":
            elem_dict = {"input_info": 
                {"ref_claim": eval_data_df.loc[i, "Reference Sentence"]}}
            eval_data.append(elem_dict)
        else:
            continue
    
    common_ques_gpt = []
    for i,item in enumerate(eval_data):
        claim = item["input_info"]["ref_claim"]
        response = generate_common_question_by_gpt(claim=claim)
        res_dict = {"ref_claim": claim,
                    "common_ques": response}
        common_ques_gpt.append(res_dict)
        print(f"CQ for {i} is generated.")
    
    common_ques_mixtral = get_common_question_by_mixtral(eval_data_file_path=eval_data_file_path)
    
    with open(path+"common_ques_gpt.json", 'w') as json_file:
        json.dump(common_ques_gpt, json_file, indent=4)
    with open(path+"common_ques_mixtral.json", 'w') as json_file:
        json.dump(common_ques_mixtral, json_file, indent=4)
    
    
    
