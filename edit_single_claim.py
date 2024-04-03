# from run_editor_sequential import run_editor_one_instance
"""Runs the RARR editor on a JSONL file of claims.

Runs question generation, retrieval, agreement gate, and editing on a file with claims
using GPT-3 and Bing.
"""
import argparse
import json
import os, time
from typing import Any, Dict
import json
import jsonlines
import Levenshtein
import tqdm
from prompts import hallucination_prompts, rarr_prompts
from utils import (
    agreement_gate,
    editor,
    evidence_selection,
    hallucination,
    search,
    question_generation,
    LLM_QG,
    api,
    LLM_target_sent_gen,
    LLM_verify_target_sent,
    chatgpt_prompt
)
import pandas as pd

path_name = "/root/RnD_Project/"

questions_output_path = path_name + "outputs/questions.json"
evidences_output_path = path_name +  "outputs/evidences.json"
agreements_output_path = path_name +  "outputs/agreements.json"
edits_output_path = path_name +  "outputs/edits.json"
results_output_path = path_name +  "outputs/results.json"

def append_to_json_file(data, file_path):
    
    # Check if the file exists
    if os.path.exists(file_path):
        # Read existing data from the file
        with open(file_path, 'r') as json_file:
            existing_data = json.load(json_file)
    else:
        # If the file doesn't exist, start with an empty list
        existing_data = []

    # Append the new data to the existing list
    existing_data.append(data)

    # Write the updated list back to the file
    with open(file_path, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)

def get_questions(
    claim_id: int,
    claim: str,
    entity: str,
    location: str,
    context: str = None,
    model: str = "mixtral8x7b",
    temperature_qgen: float = 0.7,
    num_rounds_qgen: int = 1,
):

    # Generate questions for the claim
    if entity:
        model_prompt = "QGEN_PROMPT_WITH_ENTITY_"+model
    elif location:
        ques_gen_prompt = "QGEN_PROMPT_WITH_LOCATION_"+model
        target_sent_gen_prompt = "TARGET_SENT_GEN_PROMPT_WITH_LOCATION_"+model
        verify_entity_prompt = "VERIFY_TARGET_ENTITY_PROMPT_"+model
    else:
        model_prompt = "QGEN_PROMPT_"+model

    prompt_target_sent = getattr(rarr_prompts, target_sent_gen_prompt)
    prompt_entity_verify = getattr(rarr_prompts, verify_entity_prompt)
    prompt_ques_gen = getattr(rarr_prompts, ques_gen_prompt)
    
    target_sent, reason_for_target_sent = LLM_target_sent_gen.run_rarr_target_sentence_generation(
        claim=claim,
        model=model,
        prompt=prompt_target_sent,
        entity=entity,
        location=location
    )
    
    decision, reason_for_entity, correct_target_sent, response = LLM_verify_target_sent.verify_rarr_target_sentence(
        ref_claim=claim,
        target_claim=target_sent,
        target_location=location,
        model=model,
        prompt=prompt_entity_verify,
        entity=entity
    )
    
    questions = LLM_QG.run_rarr_question_generation(
        target_claim=target_sent,
        location=location,
        model=model,
        prompt=prompt_ques_gen,
        entity=entity
    )
    
    if "na" in correct_target_sent.lower():
        sent = target_sent
    elif correct_target_sent == "":
        sent = target_sent
    else:
        sent = correct_target_sent
        
    output = {
        "claim_id": claim_id,
        "claim_ref": claim,
        "entity": entity,
        "location": location,
        "model": model,
        "response": response,
        "claim_target": target_sent,
        "reason_for_target_sent": reason_for_target_sent,
        "claim_target_correct": sent,
        "decision_for_entity_verification": decision,
        "reason_for_entity_verification": reason_for_entity,
        "questions": questions
    }

    # Dump the list into the JSON file
    append_to_json_file(output, file_path=questions_output_path)

    return sent, questions, output

def get_evidence(
    claim_id: int,
    claim: str,
    entity: str,
    location: str,
    questions: list,
    max_search_results_per_query: int = 3,
    max_sentences_per_passage: int = 4,
    sliding_distance: int = 1,
    max_passages_per_search_result: int = 1,
    max_evidences_per_question: int = 1,
):

    # Run search on generated question for the claim
    evidences_for_questions = [
        search.run_search(
            query=query,
            max_search_results_per_query=max_search_results_per_query,
            max_sentences_per_passage=max_sentences_per_passage,
            sliding_distance=sliding_distance,
            max_passages_per_search_result_to_return=max_passages_per_search_result,
            sub_key = api.SUBSCRIPTION_KEY
        )
        for query in questions
    ]

    # Flatten the evidences per question into a single list.
    used_evidences = [
        e
        for cur_evids in evidences_for_questions
        for e in cur_evids[:max_evidences_per_question]
    ]

    output = {
        "claim_id": claim_id,
        "claim_target": claim,
        "entity": entity,
        "location": location,
        "questions": questions,
        "evidences": used_evidences
    }

    # Dump the list into the JSON file
    append_to_json_file(output, file_path=evidences_output_path)

    return used_evidences, output

def check_agreement(
    claim_id: int,
    claim: str,
    entity: str,
    location: str,
    evid_id: int,
    query: str,
    evidence: str,
    context: str = None,
    model: str = "mixtral8x7b",
):

    # Run the agreement gate on the current (claim, context, query, evidence) tuple
    prompt = rarr_prompts.CONTEXTUAL_AGREEMENT_GATE_PROMPT if context else rarr_prompts.AGREEMENT_GATE_PROMPT
    gate = agreement_gate.run_agreement_gate(
        claim=claim,
        context=context,
        query=query,
        evidence=evidence,
        model=model,
        prompt=prompt,
    )

    output = {
        "claim_id": claim_id,
        "claim_target": claim,
        "entity": entity,
        "location": location,
        "evid_id": evid_id,
        "query": query,
        "evidence": evidence,
        "model": model,
        "gate": gate
    }

    # Dump the list into the JSON file
    append_to_json_file(output, file_path=agreements_output_path)

    return gate, output

def edit_claim(
    claim_id: int,
    claim: str,
    entity: str,
    location: str,
    evid_id: int,
    query: str,
    evidence: str,
    context: str = None,
    model: str = "mixtral8x7b",
):

    # Run the editor gate
    prompt = rarr_prompts.CONTEXTUAL_EDITOR_PROMPT if context else rarr_prompts.EDITOR_PROMPT
    edited_claim = editor.run_rarr_editor(
        claim=claim,
        context=context,
        query=query,
        evidence=evidence,
        model=model,
        prompt=prompt
    )["text"]

    output = {
        "claim_id": claim_id,
        "claim_target": claim,
        "entity": entity,
        "location": location,
        "evid_id": evid_id,
        "query": query,
        "evidence": evidence,
        "model": model,
        "edited_claim": edited_claim
    }

    # Dump the list into the JSON file
    append_to_json_file(output, file_path=edits_output_path)

    return edited_claim, output

def write_questions_json(
    data: dict
):

    num_claims = len(data)
    output_list = []
    for claim_id, item in enumerate(data):
        claim = data[claim_id]["input_info"]["claim"]
        entity = None
        location = data[claim_id]["input_info"]["location"]

        t1 = time.time()
        output_list.append(get_questions(claim_id, claim, entity, location)[-1])
        t2 = time.time()
        print(f"Claim: {claim_id}, Question generation module is run in {(t2-t1)/60:.2f} mint")

    # Dump the list into the JSON file
    with open(questions_output_path.replace(".json","_final.json"), 'w') as json_file:
        json.dump(output_list, json_file, indent=4)

def write_evidences_json():

    # Read the questions.json file
    with open(questions_output_path.replace(".json","_final.json"), 'r') as json_file:
        data = json.load(json_file)

    num_claims = len(data)
    output_list = []
    for claim_id, item in enumerate(data):
        claim = item["claim_target_correct"]
        entity = None
        location = item["location"]
        questions = item["questions"]
        print(f"Claim: {claim_id}, Question is taken from file")
    
        t1 = time.time()
        output_list.append(get_evidence(claim_id, claim, entity, location, questions)[-1])
        t2 = time.time()
        print(f"Claim: {claim_id}, Evidence module is run in {(t2-t1)/60:.2f} mint")

    # Dump the list into the JSON file
    with open(evidences_output_path.replace(".json","_final.json"), 'w') as json_file:
        json.dump(output_list, json_file, indent=4)

def write_agreements_json():

    # Read the evidences.json file
    with open(evidences_output_path.replace(".json","_final.json"), 'r') as json_file:
        data = json.load(json_file)

    num_claims = len(data)
    output_list = []
    for claim_id, item in enumerate(data):
        claim = item["claim_target"]
        entity = None
        location = item["location"]
        evidence_data = item["evidences"]
        print(f"Claim: {claim_id}, Evidence is taken from file")

        for evid_id, evid in enumerate(evidence_data):
            t1 = time.time()
            output_list.append(check_agreement(claim_id, claim, entity, location, evid_id, evid['query'], evid['text'])[-1])
            t2 = time.time()
            print(f"Claim: {claim_id}, Evidence: {evid_id}, Agreement module is run in {(t2-t1)/60:.2f} mint")

    # Dump the list into the JSON file
    with open(agreements_output_path.replace(".json","_final.json"), 'w') as json_file:
        json.dump(output_list, json_file, indent=4)

def write_edits_json():

    # Read the questions.json file
    with open(questions_output_path.replace(".json","_final.json"), 'r') as json_file:
        ques_data = json.load(json_file)

    # Read the evidences.json file
    with open(evidences_output_path.replace(".json","_final.json"), 'r') as json_file:
        evid_data = json.load(json_file)

    num_claims = len(evid_data)
    
    # Read the agreements.json file
    with open(agreements_output_path.replace(".json","_final.json"), 'r') as json_file:
        data = json.load(json_file)

    agreement_data = [None]*num_claims
    for i in range(num_claims):
        evids = []
        for item in data:
            if item["claim_id"] == i:
                evids.append(item)      
        agreement_data[i] = evids

    output_list = []
    results_list = []

    for claim_id, ag_item in enumerate(agreement_data):
        original_claim = evid_data[claim_id]["claim_target"]
        edit_rev_list = []
        claim = original_claim

        for evid_id, evid_item in enumerate(ag_item):
            entity = None
            location = evid_item["location"]
            evid = evid_item["evidence"]
            query = evid_item["query"]
            gate = evid_item["gate"]
            print(f"Claim: {claim_id}, Evidence: {evid_id}, Agreement is taken from file")

            # Run the editor gate if the agreement gate is open
            claim_input = claim
            if gate["is_open"]:
                t1 = time.time()
                edited_claim, output = edit_claim(claim_id, claim, entity, location, evid_id, query, evid)
                t2 = time.time()

                # Don't keep the edit if the editor makes a huge change
                max_edit_ratio = 50
                if Levenshtein.distance(claim, edited_claim) / len(claim) <= max_edit_ratio:
                    claim = edited_claim
            
                print(f"Claim: {claim_id}, Evidence: {evid_id}, Editor module is run in {(t2-t1)/60:.2f} mint")
            else:
                output = None
                print(f"Claim: {claim_id}, Evidence: {evid_id}, Editor module is skipped")

            edit_rev_list.append({
                "claim": claim_input,
                "query": query,
                "evidence": evid,
                "is_open": gate["is_open"],
                "decision": gate["decision"],
                "edited_claim": claim
            })
        
        results = {
            "claim_id": ques_data[claim_id]["claim_id"],
            "claim_ref": ques_data[claim_id]["claim_ref"],
            "claim_target": ques_data[claim_id]["claim_target"],
            "is_target_claim_ok": ques_data[claim_id]["decision_for_entity_verification"],
            "claim_target_correct": ques_data[claim_id]["claim_target_correct"],
            "questions": ques_data[claim_id]["questions"],
            "edit_revisions": edit_rev_list,
            "claim_ref": ques_data[claim_id]["claim_ref"],
            "entity": ques_data[claim_id]["entity"],
            "location": ques_data[claim_id]["location"],
            "claim_original": original_claim,
            "claim_attributed": claim
        }

        output_list.append(output)
        results_list.append(results)

    # Dump the list into the JSON file
    with open(results_output_path.replace(".json","_final.json"), 'w') as json_file:
        json.dump(results_list, json_file, indent=4)

if __name__ == "__main__":

    eval_data_df = pd.read_csv("/root/RnD_Project/inputs/Amazon RnD_ Evaluation Dataset - Updated 200 samples with Qs _V3.csv")
    data = []
    for i in range(eval_data_df.shape[0]):
        elem_dict = {"input_info": 
        {"claim": eval_data_df.loc[i, "Reference Sentence"], 
        "location": eval_data_df.loc[i, "Target Location"]}}
        data.append(elem_dict)
    
    # data = [{"input_info": 
    # {"claim": "Angelina Jolie is an American actress, filmmaker and humanitarian. The recipient of numerous accolades, including an Academy Award and three Golden Globe Awards, she has been named Hollywood's highest-paid actress multiple times.", 
    # "location": "Maharashtra"}},
    # {"input_info": 
    # {"claim": "Dwayne Douglas Johnson, also known by his ring name the Rock, is an American actor and professional wrestler currently signed to WWE.", 
    # "location": "Kerala"}},
    # {"input_info": {"claim": "Robert John Downey Jr. is an American actor. His career has been characterized by critical success in his youth, followed by a period of substance abuse and legal troubles, and a surge in popular and commercial success later in his career.", 
    # "location": "West Bengal"}}]

    write_questions_json(data[40:60])
    # write_evidences_json()
    # write_agreements_json()
    # write_edits_json()
    
    # DONE
    # 0-10, 20-40
