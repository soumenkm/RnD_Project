#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri April 12 06:46:37 2024

@author: soumensmacbookair
"""

import json
import os, time
from typing import Any, Dict, Tuple, List
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
    LLM_edit_claim
)
import pandas as pd
from pathlib import Path

number = str(1)
output_name = "50_100"

path_name = "/root/RnD_Project/rarr_with_non_seq/"+number+"/"
questions_output_path = "/root/RnD_Project/mixtral_outputs/"+number+"/outputs_"+output_name+"/questions.json"
evidences_output_path = path_name +  "outputs_"+output_name+"/evidences.json"
agreements_output_path = path_name +  "outputs_"+output_name+"/agreements.json"
edits_output_path = path_name +  "outputs_"+output_name+"/edits.json"
results_output_path = path_name +  "outputs_"+output_name+"/results.json"

def get_questions(
    claim_id: int,
    claim: str,
    location: str,
    is_verify: bool,
    model: str = "mixtral8x7b",
    temperature_qgen: float = 0.7):

    # Generate questions for the claim
    ques_gen_prompt = "QGEN_PROMPT_WITH_LOCATION_"+model
    target_sent_gen_prompt = "TARGET_SENT_GEN_PROMPT_WITH_LOCATION_"+model
    verify_entity_prompt = "VERIFY_TARGET_ENTITY_PROMPT_"+model

    prompt_target_sent = getattr(rarr_prompts, target_sent_gen_prompt)
    prompt_entity_verify = getattr(rarr_prompts, verify_entity_prompt)
    prompt_ques_gen = getattr(rarr_prompts, ques_gen_prompt)
    
    target_sent, reason_for_target_sent = LLM_target_sent_gen.run_rarr_target_sentence_generation(
        claim=claim,
        model=model,
        prompt=prompt_target_sent,
        location=location
    )
    
    if is_verify:
        decision, reason_for_entity, correct_target_sent = LLM_verify_target_sent.verify_rarr_target_sentence(
            ref_claim=claim,
            target_claim=target_sent,
            target_location=location,
            model=model,
            prompt=prompt_entity_verify,
        )

        if decision.lower() == "accepted":
            sent = correct_target_sent
        elif decision.lower() == "rejected":
            sent = target_sent
        else:
            print(f"Decision is something else! {decision}")
            sent = target_sent
    else:
        correct_target_sent = target_sent
        decision = "Not Applicable"
        reason_for_entity = "Not Applicable"
        sent = correct_target_sent
    
    questions = LLM_QG.run_rarr_question_generation(
        target_claim=sent,
        location=location,
        model=model,
        prompt=prompt_ques_gen
    )
    
    output = {
        "claim_id": claim_id,
        "claim_ref": claim,
        "location": location,
        "model": model,
        "claim_target": target_sent,
        "reason_for_target_sent": reason_for_target_sent,
        "claim_target_correct": sent,
        "decision_for_entity_verification": decision,
        "reason_for_entity_verification": reason_for_entity,
        "questions": questions
    }
    
    return sent, questions, output

def get_evidence(
    claim_id: int,
    claim: str,
    location: str,
    questions: list,
    max_passages_per_search_result_to_score: int,
    ranking_model: str,
    max_search_results_per_query: int = 3,
    max_sentences_per_passage: int = 4,
    sliding_distance: int = 1,
    max_passages_per_search_result: int = 1,
    max_evidences_per_question: int = 1):

    # Run search on generated question for the claim
    evidences_for_questions = [
        search.run_search(
            query=query,
            max_search_results_per_query=max_search_results_per_query,
            max_sentences_per_passage=max_sentences_per_passage,
            sliding_distance=sliding_distance,
            max_passages_per_search_result_to_return=max_passages_per_search_result,
            sub_key = api.SUBSCRIPTION_KEY,
            max_passages_per_search_result_to_score=max_passages_per_search_result_to_score,
            ranking_model=ranking_model
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
        "location": location,
        "questions": questions,
        "evidences": used_evidences
    }
    
    return used_evidences, output

def check_agreement(
    claim_id: int,
    claim: str,
    location: str,
    evid_id: int,
    query: str,
    evidence: str,
    model: str = "mixtral8x7b",):

    # Run the agreement gate on the current (claim, context, query, evidence) tuple
    prompt = rarr_prompts.AGREEMENT_GATE_PROMPT
    gate = agreement_gate.run_agreement_gate(
        claim=claim,
        query=query,
        evidence=evidence,
        model=model,
        prompt=prompt,
    )

    output = {
        "claim_id": claim_id,
        "claim_target": claim,
        "location": location,
        "evid_id": evid_id,
        "query": query,
        "evidence": evidence,
        "model": model,
        "gate": gate
    }

    return gate, output

def edit_claim(
    claim_id: int,
    claim: str,
    location: str,
    evid_id: int,
    query: str,
    evidence: str,
    model: str = "mixtral8x7b",):

    # Run the editor gate
    prompt = rarr_prompts.EDITOR_PROMPT
    edited_claim = editor.run_rarr_editor(
        claim=claim,
        query=query,
        evidence=evidence,
        model=model,
        prompt=prompt
    )["text"]

    output = {
        "claim_id": claim_id,
        "claim_target": claim,
        "location": location,
        "evid_id": evid_id,
        "query": query,
        "evidence": evidence,
        "model": model,
        "edited_claim": edited_claim
    }

    return edited_claim, output

def write_questions_json(
    data: dict,
    is_verify: bool):

    num_claims = len(data)
    output_list = []
    for claim_id, item in enumerate(data):
        claim = data[claim_id]["input_info"]["claim"]
        location = data[claim_id]["input_info"]["location"]

        t1 = time.time()
        output_list.append(get_questions(claim_id, claim, location, is_verify)[-1])
        t2 = time.time()
        print(f"Claim: {claim_id}, Question generation module is run in {(t2-t1)/60:.2f} mint")
        break

    # Dump the list into the JSON file
    # output_file = Path(questions_output_path)
    # output_file.parent.mkdir(exist_ok=True, parents=True)
    # with open(questions_output_path, 'w+') as json_file:
    #     json.dump(output_list, json_file, indent=4)
 
def write_evidences_json(
    max_passages_per_search_result_to_score: int,
    ranking_model: str):
    
    # Read the questions.json file
    with open(questions_output_path, 'r') as json_file:
        data = json.load(json_file)

    num_claims = len(data)
    print(num_claims)
    output_list = []
    for claim_id, item in enumerate(data):
        claim = item["claim_target_correct"]
        location = item["location"]
        questions = item["questions"]
        print(f"Claim: {claim_id}, Question is taken from file")
    
        t1 = time.time()
        try:
            res = get_evidence(claim_id, claim, location, questions,
                                max_passages_per_search_result_to_score,
                                ranking_model)[-1]
        except Exception as e:
            print(e)
            res = f"error: {e}"
        
        output_list.append(res)
        t2 = time.time()
        print(f"Claim: {claim_id}, Evidence module is run in {(t2-t1)/60:.2f} mint")

    # Dump the list into the JSON file
    output_file = Path(evidences_output_path)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(evidences_output_path, 'w') as json_file:
        json.dump(output_list, json_file, indent=4)

def write_agreements_json():

    # Read the evidences.json file
    with open(evidences_output_path, 'r') as json_file:
        data = json.load(json_file)

    num_claims = len(data)
    output_list = []
    for claim_id, item in enumerate(data):
        if type(item) == str:
            output_list.append(item)
            continue
        claim = item["claim_target"]
        location = item["location"]
        evidence_data = item["evidences"]
        print(f"Claim: {claim_id}, Evidence is taken from file")

        for evid_id, evid in enumerate(evidence_data):
            t1 = time.time()
            output_list.append(check_agreement(claim_id, claim, location, evid_id, evid['query'], evid['text'])[-1])
            t2 = time.time()
            print(f"Claim: {claim_id}, Evidence: {evid_id}, Agreement module is run in {(t2-t1)/60:.2f} mint")

    # Dump the list into the JSON file
    output_file = Path(agreements_output_path)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(agreements_output_path, 'w') as json_file:
        json.dump(output_list, json_file, indent=4)

def write_edits_json(
    is_sequential_edit: bool):

    # Read the questions.json file
    with open(questions_output_path, 'r') as json_file:
        ques_data = json.load(json_file)

    # Read the evidences.json file
    with open(evidences_output_path, 'r') as json_file:
        evid_data = json.load(json_file)

    num_claims = len(evid_data)
    
    # Read the agreements.json file
    with open(agreements_output_path, 'r') as json_file:
        data = json.load(json_file)

    agreement_data = [None]*num_claims
    for i in range(num_claims):
        evids = []
        for item in data:
            if type(item) == str:
                continue
            if item["claim_id"] == i:
                evids.append(item)      
        agreement_data[i] = evids

    if is_sequential_edit:
        output_list = []
        results_list = []

        for claim_id, ag_item in enumerate(agreement_data):
            flag = 1
            claim = ""
            original_claim = ""
            if type(ag_item) == str:
                flag = 0
            elif type(evid_data) == str:
                flag = 0
            elif type(evid_data[claim_id]) == str:
                flag = 0
            
            edit_rev_list = []
            if(flag == 1):
                original_claim = evid_data[claim_id]["claim_target"]
                claim = original_claim
                for evid_id, evid_item in enumerate(ag_item):
                    location = evid_item["location"]
                    evid = evid_item["evidence"]
                    query = evid_item["query"]
                    gate = evid_item["gate"]
                    print(f"Claim: {claim_id}, Evidence: {evid_id}, Agreement is taken from file")

                    # Run the editor gate if the agreement gate is open
                    claim_input = claim
                    if gate["is_open"]:
                        t1 = time.time()
                        edited_claim, output = edit_claim(claim_id, claim, location, evid_id, query, evid)
                        t2 = time.time()

                        # Don't keep the edit if the editor makes a huge change
                        max_edit_ratio = 50
                        if(len(claim) > 0):
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
                "location": ques_data[claim_id]["location"],
                "claim_original": original_claim,
                "claim_attributed": claim
            }

            output_list.append(output)
            results_list.append(results)

        # Dump the list into the JSON file
        output_file = Path(edits_output_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        with open(edits_output_path, 'w') as json_file:
            json.dump(output_list, json_file, indent=4)
        output_file = Path(results_output_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        with open(results_output_path, 'w') as json_file:
            json.dump(results_list, json_file, indent=4)
    
    else:
        evid_dict = {}
        for claim_id, ag_item in enumerate(agreement_data):
            if type(ag_item) == str:
                continue
            if type(evid_data) == str:
                continue
            if type(evid_data[claim_id]) == str:
                continue
            original_claim = evid_data[claim_id]["claim_target"]
            claim = original_claim

            evid_list = []
            for evid_id, evid_item in enumerate(ag_item):
                evid = evid_item["evidence"]
                query = evid_item["query"]
                gate = evid_item["gate"]["is_open"]
                if gate:
                    evid_list.append({
                        "query": query,
                        "evidence": evid,
                        "gate": gate
                    })
            evid_dict[f"claim_{claim_id}"] = evid_list
        
        evidence_dict = {}
        for c_key, c_val in evid_dict.items():
            evidence_list = []
            i = 0
            for item in c_val:
                evidence_list.append(f"{i+1}: " + item["evidence"])
                i = i + 1
            evidence_dict[c_key] = evidence_list
        
        output_list = []
        results_list = []
        for claim_id, evidences in evidence_dict.items():
            t1 = time.time()
            claim = evid_data[int(claim_id.split("_")[-1])]["claim_target"]
            if evidences:
                attributed_target_sent = LLM_edit_claim.edit_target_sentence(
                    claim=claim,
                    evidences=evidences,
                    model="mixtral8x7b",
                    prompt=rarr_prompts.SINGLE_EDITOR_PROMPT
                )
                # Don't keep the edit if the editor makes a huge change
                max_edit_ratio = 100
                if(len(claim)> 0):
                    if not Levenshtein.distance(claim, attributed_target_sent) / len(claim) <= max_edit_ratio:
                        attributed_target_sent = claim
                    
            else:
                attributed_target_sent = claim
            
            t2 = time.time()
            print(f"{claim_id}: Editor module is run in {(t2-t1)/60:.2f} mint")

            output_list.append({
                "claim": claim,
                "evidence": evid_dict[claim_id],
                "edited_claim": attributed_target_sent
            })

            claim_id = int(claim_id.split("_")[-1])
            results = {
                "claim_id": ques_data[claim_id]["claim_id"],
                "claim_ref": ques_data[claim_id]["claim_ref"],
                "claim_target": ques_data[claim_id]["claim_target"],
                "is_target_claim_ok": ques_data[claim_id]["decision_for_entity_verification"],
                "claim_target_correct": ques_data[claim_id]["claim_target_correct"],
                "questions": ques_data[claim_id]["questions"],
                "location": ques_data[claim_id]["location"],
                "claim_original": claim,
                "claim_attributed": attributed_target_sent
            }
            results_list.append(results)
    
        # Dump the list into the JSON file
        with open(edits_output_path, 'w') as json_file:
            json.dump(output_list, json_file, indent=4)
        with open(results_output_path, 'w') as json_file:
            json.dump(results_list, json_file, indent=4)
                        
if __name__ == "__main__":

    eval_data_df = pd.read_csv("/root/RnD_Project/inputs/revised_final_dataset_200.csv")
    data = []
    for i in range(eval_data_df.shape[0]):
        elem_dict = {"input_info": 
        {"claim": eval_data_df.loc[i, "Reference Sentence"], 
        "location": eval_data_df.loc[i, "Target Location"]}}
        data.append(elem_dict)
    
    write_questions_json(data[50:100], is_verify=False)
    write_evidences_json(max_passages_per_search_result_to_score=30,
                        ranking_model="cross_encoder")
    write_agreements_json()
    write_edits_json(is_sequential_edit=False)
    
    # write_questions_json(data[0:30], is_verify=True)
    # write_evidences_json(max_passages_per_search_result_to_score=-1,
    #                     ranking_model="cohere") # 30, "cross_encoder"
    # write_agreements_json()
    # write_edits_json(is_sequential_edit=False)
    
    # DONE
