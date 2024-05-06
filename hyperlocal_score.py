import json
import pandas as pd

number = str(3)
path = "/root/RnD_Project/outputs/rarr_mixtral_zero_shot_non_seq/"+number+"/outputs_100_200/"

def get_hyperlocal_results(data):
    score_1_list = []
    score_2_list = []
    score_3_list = []
    score_1 = 0
    score_2 = 0
    score_3 = 0
    for i, item in enumerate(data):
        if(item["hyper_local_score"] == 1):
            for key in item["rarr_result"].keys():
                score_1_list.append(item["rarr_result"][key])
        elif(item["hyper_local_score"] == 2):
            for key in item["rarr_result"].keys():
                score_2_list.append(item["rarr_result"][key])
        elif(item["hyper_local_score"] == 3):
            for key in item["rarr_result"].keys():
                score_3_list.append(item["rarr_result"][key])
    
    if(len(score_1_list)!=0):   
        score_1 = sum(score_1_list)/len(score_1_list)
    if(len(score_2_list)!=0):   
        score_2 = sum(score_2_list)/len(score_2_list)
    if(len(score_3_list)!=0):   
        score_3 = sum(score_3_list)/len(score_3_list)
    return score_1, score_2, score_3

def get_results_data(file_path, eval_data_df, eval_count):
    with open(file_path, 'r') as json_file:
        res_data = json.load(json_file)
    data = []
    for i in range(eval_data_df.shape[0]):    
        for j,item in enumerate(res_data):
            if("ref_claim" in item.keys()):
                if eval_data_df.loc[i, "Reference Sentence"].strip() == item["ref_claim"].strip():
                    if eval_data_df.loc[i, "Target Location"].strip() == item["target_location"].strip():
                        elem_dict = {"ref_claim": eval_data_df.loc[i, "Reference Sentence"], 
                            "target_location": eval_data_df.loc[i, "Target Location"],
                            "hyper_local_score": eval_data_df.loc[i, "Hyperlocal Score"],
                            "rarr_result": item["rarr"]}
                        data.append(elem_dict)
                    
    score_1, score_2, score_3 = get_hyperlocal_results(data)

    output_list = [{"hyperlocal_score_1": score_1, "hyperlocal_score_2": score_2, "hyperlocal_score_3": score_3}]

    print("1: ", score_1)
    print("2: ", score_2)
    print("3: ", score_3)

    new_path = "/root/RnD_Project/outputs/mixtral_few_shot/"+number+"/outputs_100_200/"

    with open(new_path+"hyperlocal_results_"+str(eval_count)+".json", 'w') as json_file:
        json.dump(output_list, json_file, indent=4)
    
    return score_1, score_2, score_3

if __name__ == "__main__":
    
    eval_data_df = pd.read_csv("/root/RnD_Project/inputs/revised_final_dataset_200.csv")
    score_1 = 0
    score_2 = 0
    score_3 = 0
    for i in range(1,4):
        file_path = path+"eval_results_"+str(i)+".json"
        print(file_path)
        s_1, s_2, s_3 = get_results_data(file_path, eval_data_df, i)
        score_1 += s_1
        score_2 += s_2
        score_3 += s_3
        
    print("Final 1: ", score_1/3.0)
    print("Final 2: ", score_2/3.0)
    print("Final 3: ", score_3/3.0)
        


    
    
    
