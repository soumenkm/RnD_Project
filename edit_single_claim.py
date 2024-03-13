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
    LLM_QG
)

questions_output_path = "/raid/speech/soumenmondal/RnD_project/RARR/outputs/questions.json"
evidences_output_path = "/raid/speech/soumenmondal/RnD_project/RARR/outputs/evidences.json"
agreements_output_path = "/raid/speech/soumenmondal/RnD_project/RARR/outputs/agreements.json"
edits_output_path = "/raid/speech/soumenmondal/RnD_project/RARR/outputs/edits.json"
results_output_path = "/raid/speech/soumenmondal/RnD_project/RARR/outputs/results.json"

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
        model_prompt = "QGEN_PROMPT_WITH_LOCATION_"+model
    else:
        model_prompt = "QGEN_PROMPT_"+model

    prompt = getattr(rarr_prompts, model_prompt)
    sent, questions = LLM_QG.run_rarr_question_generation(
        claim=claim,
        entity=entity,
        location=location,
        model=model,
        prompt=prompt,
        temperature=temperature_qgen,
        num_rounds=num_rounds_qgen,
    )

    output = {
        "claim_id": claim_id,
        "claim": claim,
        "entity": entity,
        "location": location,
        "model": model,
        "claim_target": sent,
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
    with open(questions_output_path, 'r') as json_file:
        data = json.load(json_file)

    num_claims = len(data)
    output_list = []
    for claim_id, item in enumerate(data):
        claim = item["claim_target"]
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
    with open(evidences_output_path, 'r') as json_file:
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
                max_edit_ratio = 100
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
            "claim_ref": ques_data[claim_id]["claim"],
            "entity": ques_data[claim_id]["entity"],
            "location": ques_data[claim_id]["location"],
            "claim_target": ques_data[claim_id]["claim_target"],
            "questions": ques_data[claim_id]["questions"],
            "edit_revisions": edit_rev_list,
            "claim_original": original_claim,
            "claim_attributed": claim
        }

        output_list.append(output)
        results_list.append(results)

        # rarr_result = {
        #     "context": None,
        #     "text": original_claim,
        #     "questions": ques_data[claim_id]["questions"],
        #     "evidences_for_questions": evid_data["claim_id"]["evidences_for_questions"],
        #     "revisions": [
        #         {
        #             "original_text": original_claim,
        #             "revised_text": revision_steps[-1]["text"],
        #             "evidences": evid_data["claim_id"]["evidences"],
        #             "agreement_gates": agreement_gates,
        #             "revision_steps": revision_steps,
        #         }
        #     ],
        # }
        
        # selected_evidences = evidence_selection.select_evidences(rarr_result)
        # result["selected_evidences"] = selected_evidences

    # Dump the list into the JSON file
    with open(results_output_path.replace(".json","_final.json"), 'w') as json_file:
        json.dump(results_list, json_file, indent=4)

def run_editor_one_instance(
    claim_id: int,
    claim: str,
    entity: str,
    location: str,
    context: str = None,
    model: str = "mixtral8x7b",
    temperature_qgen: float = 0.7,
    num_rounds_qgen: int = 1,
    max_search_results_per_query: int = 3,
    max_sentences_per_passage: int = 4,
    sliding_distance: int = 1,
    max_passages_per_search_result: int = 1,
    max_evidences_per_question: int = 1,
    max_edit_ratio: float = 100
) -> Dict[str, Any]:
    """Runs query generation, search, agreement gating, and editing on a claim.

    Args:
        claim: Text to check the validity of.
        model: Name of the OpenAI GPT-3 model to use.
        temperature_qgen: Sampling temperature to use for query generation.
        num_rounds_qgen: Number of times to sample questions.
        max_search_results_per_query: Maximum number of search results per query.
        max_sentences_per_passage: Maximum number of sentences for each passage.
        sliding_distance: Sliding window distance over the sentences of each search
            result. Used to extract passages.
        max_passages_per_search_result:  Maximum number of passages to return for
            each search result. A passage ranker is applied first.
        max_evidences_per_question: Maximum number of evidences to return per question.
        max_edit_ratio: Maximum edit ratio between claim and edit for each round.
    Returns:
        result: All revision information, including the queries generated, search
            results, agreement gate information, and each revision step done on the
            claim.
    """
    
    # Generate questions for the claim
    run_ques = False
    if run_ques:
        t1 = time.time()
        claim_target, questions = get_questions(claim_id, claim, entity, location, context, model, temperature_qgen, num_rounds_qgen)
        t2 = time.time()
        print(f"Claim: {claim_id}, Question generation module is run in {(t2-t1)/60:.2f} mint")
    else:
        with open(questions_output_path, 'r') as json_file:
            questions_data = json.load(json_file)
        for item in questions_data:
            if int(item["claim_id"]) == claim_id:
                claim_target = item["claim_target"]
                questions = item["questions"]
        print(f"Claim: {claim_id}, Question is taken from file")

    # Set the claim_target as the claim now
    claim_ref = claim
    claim = claim_target
    original_claim = claim

    # Run search on generated question for the claim
    run_search = False
    if run_search:
        t1 = time.time()
        used_evidences = get_evidence(claim_id, claim, entity, location, questions)
        t2 = time.time()
        print(f"Claim: {claim_id}, Evidence module is run in {(t2-t1)/60:.2f} mint")
    else:
        with open(evidences_output_path, 'r') as json_file:
            evidences_data = json.load(json_file)
        for item in evidences_data:
            if int(item["claim_id"]) == claim_id:
                used_evidences = item["evidences"]
        print(f"Claim: {claim_id}, Evidences is taken from file")

    # Iterative editing over each evidence
    revision_steps = []
    agreement_gates = []
    run_ag = [False, False, True, True]

    for e, evid in enumerate(used_evidences):
        
        # Run the agreement gate on the current (claim, context, query, evidence) tuple
        if run_ag[e]:
            t1 = time.time()
            gate = check_agreement(claim_id, claim, entity, location, e+1, evid['query'], evid['text'], context, model)
            t2 = time.time()
            print(f"Claim: {claim_id}, Evidence: {e+1}, Agreement module is run in {(t2-t1)/60:.2f} mint")
        else:
            with open(agreements_output_path, 'r') as json_file:
                agreements_data = json.load(json_file)
            for item in agreements_data:
                if int(item["claim_id"]) == claim_id and int(item["evid_id"]) == e+1:
                    gate = item["gate"]
            print(f"Claim: {claim_id}, Evidence: {e+1}, Agreement is taken from file")

        agreement_gates.append(gate)

        # Run the editor gate if the agreement gate is open
        if gate["is_open"]:
            t1 = time.time()
            edited_claim = edit_claim(claim_id, claim, entity, location, e+1, evid["query"], evid["text"], context, model)
            t2 = time.time()
            # Don't keep the edit if the editor makes a huge change
            if Levenshtein.distance(claim, edited_claim) / len(claim) <= max_edit_ratio:
                claim = edited_claim
        
            print(f"Claim: {claim_id}, Evidence: {e+1}, Editor module is run in {(t2-t1)/60:.2f} mint")
        else:
            print(f"Claim: {claim_id}, Evidence: {e+1}, Editor module is skipped")

        revision_steps.append({"text": claim})

    result = {
        "context": context,
        "text": original_claim,
        "questions": questions,
        "evidences_for_questions": evidences_for_questions,
        "revisions": [
            {
                "original_text": original_claim,
                "revised_text": revision_steps[-1]["text"],
                "evidences": used_evidences,
                "agreement_gates": agreement_gates,
                "revision_steps": revision_steps,
            }
        ],
    }
    selected_evidences = evidence_selection.select_evidences(result)
    result["selected_evidences"] = selected_evidences
    return result

if __name__ == "__main__":

    data = [{"input_info": 
        {"claim": "On July 10, 2006, concrete ceiling panels and debris fell on a car traveling on the two-lane ramp connecting northbound I-93 to eastbound I-90 in South Boston, killing the passenger and injuring her husband, who was driving.", 
        "location": "Uttarakhand"}},
        {"input_info": 
        {"claim": "A train derailment occurred on February 3, 2023, at 8:55 p.m. EST, when 38 cars of a Norfolk Southern freight train carrying hazardous materials derailed in East Palestine, Ohio, United States.", 
        "location": "Andhra Pradesh"}},
        {"input_info": 
        {"claim": "Angelina Jolie is an American actress, filmmaker and humanitarian. The recipient of numerous accolades, including an Academy Award and three Golden Globe Awards, she has been named Hollywood's highest-paid actress multiple times.", 
        "location": "Kerala"}}
    ]

    # write_questions_json(data)
    # write_evidences_json()
    # write_agreements_json()
    write_edits_json()
