import json
from transformers import pipeline

# Initialize the zero-shot classifier
classifier = pipeline('zero-shot-classification', model='roberta-large-mnli')

# Define the passage and the claim
passage = "The sky is blue. It looks clear today."  
claim = "The weather is cloudy." 

# Candidate labels for NLI
candidate_labels = ['contradiction', 'neutral', 'entailment']

# Perform classification
result = classifier(
    passage, 
    candidate_labels, 
    hypothesis_template="The claim '{}' is {}.".format(claim, '{}')
)

print(result)


# with open("/root/RnD_Project/archive/outputs_0-10/evidences.json", "r") as f:
#     evid_data = json.load(f)

# evid_dict = {}
# for claim_id, evid in enumerate(evid_data):
#     evid_set = []
#     for i in evid["evidences"]:
#         evid_set.append(i["text"])
#     evid_dict[f"claim_{claim_id}"] = evid_set

# with open("/root/RnD_Project/archive/outputs_0-10/questions.json", "r") as f:
#     ques_data = json.load(f)

# claim_dict = {}
# for claim_id, ques in enumerate(ques_data):
#     claim_dict[f"claim_{claim_id}"] = ques["claim_target_correct"]

# print(claim_dict)
# print(evid_dict)

