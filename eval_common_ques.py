import json, ast
import os, time
import json
from prompts import rarr_prompts
from utils import (
    chatgpt_prompt
)
import pandas as pd

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
    
    eval_data_df = pd.read_csv("/root/RnD_Project/inputs/Amazon RnD_ Evaluation Dataset - Updated 200 samples with Qs _V3.csv")
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
    with open("/root/RnD_Project/outputs_0-10/results_final.json", 'r') as json_file:
        res_data_1 = json.load(json_file)
    with open("/root/RnD_Project/outputs_20-40/results_final.json", 'r') as json_file:
        res_data_2 = json.load(json_file)
    
    res_data = res_data_1 + res_data_2
    claim_data = []

    for i,elem in enumerate(eval_data):
        for j,item in enumerate(res_data):
            if elem["input_info"]["ref_claim"].strip() == item["claim_ref"].strip():
                elem_dict = {**elem["input_info"], **{
                    "target_claim_mixtral": item["claim_original"],
                    "target_claim_rarr": item["claim_attributed"]}
                }
                claim_data.append(elem_dict)
    
    output_list = []
    rarr_list = []
    mixtral_list = []
    
    for i,item in enumerate(claim_data):

        res_dict_mixtral, rarr_val = evaluate_target_sent_by_common_ques(item, "mixtral")
        res_dict_rarr, mixtral_val = evaluate_target_sent_by_common_ques(item, "rarr")

        output_list.append({**item, 
                            **{"mixtral": res_dict_mixtral}, 
                            **{"rarr": res_dict_rarr}})
        rarr_list.extend(rarr_val)
        mixtral_list.extend(mixtral_val)
        print(f"Claim {i} done")
    
    rarr_metric = sum(rarr_list)/len(rarr_list)
    mixtral_metric = sum(mixtral_list)/len(mixtral_list)
    output_list.append({"rarr_metric": rarr_metric, "mixtral_metric": mixtral_metric})
    
    print("rarr", rarr_metric)
    print("mixtral", mixtral_metric)
    
    with open("eval_results.json", 'w') as json_file:
        json.dump(output_list, json_file, indent=4)
    
    
    