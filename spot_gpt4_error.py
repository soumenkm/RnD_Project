import json

path_name = "/root/RnD_Project/outputs/mixtral_ablation_results/"

q4_0_shot_path = path_name + "cq_by_mixtral_corrected_Q4_initial_B_static_0_shot_gpt4_E2.json"
q6_0_shot_path = path_name + "cq_by_mixtral_corrected_Q6_initial_B_static_0_shot_gpt4_E2.json"
eval_data_path = path_name + "common_ques_mixtral_corrected.json"

with open(q4_0_shot_path, "r") as f:
    q4_data = json.load(f)
with open(q6_0_shot_path, "r") as f:
    q6_data = json.load(f)
with open(eval_data_path, "r") as f:
    eval_data = json.load(f)

q4_dict = {}   
for elem in q4_data:
    if "score" not in elem.keys(): continue
    q4_dict[elem["common_ques"]] = elem

q6_dict = {}   
for elem in q6_data:
    if "score" not in elem.keys(): continue
    q6_dict[elem["common_ques"]] = elem

q4_q6_dict = {}
for k in q4_dict.keys():
    q4_q6_dict[k] = (q4_dict[k]["score"], q6_dict[k]["score"])

gt_target_claim_dict = {}
for elem in eval_data:
    gt_target_claim_dict = {**gt_target_claim_dict, 
                            **{cq: elem["ground_truth_ann_target_claim"] for cq in elem["common_ques"]}}

res = []
for k, v in q4_q6_dict.items():
    if v[0] != v[1]:
        res.append({"q4_target_claim": q4_dict[k]["target_claim"],
        "q6_target_claim": q6_dict[k]["target_claim"],
        "ground_truth_ann_target_claim": gt_target_claim_dict[k],
        "target_location": q4_dict[k]["target_location"],
        "common_ques": k,
        "q4_score": q4_dict[k]["score"],
        "q6_score": q6_dict[k]["score"]})

with open(path_name + "score_mismatch_between_Q4_and_Q6_initial_B_static_0_shot_gpt4_E2.json", "w") as f:
    json.dump(res, f, indent=4)