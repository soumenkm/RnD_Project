#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri April 12 06:46:37 2024

@author: soumensmacbookair
"""

import json
import os, time
import pandas as pd
from pathlib import Path

target_location_list = ["India", "Mumbai", "Kerala", "Tamil Nadu", "Maharashtra",
                        "West Bengal", "Kolkata", "Delhi", "Karnataka", "Bengaluru",
                        "Chennai", "Andhra Pradesh", "New Delhi", "Gujarat", "Punjab",
                        "Rajasthan", "Hyderabad", "Assam", "Uttar Pradesh", "Goa", 
                        "Uttarakhand", "Pune", "Haryana", "Ahmedabad", "Madhya Pradesh"]

data_100_df = pd.read_csv("/root/RnD_Project/inputs/Amazon RnD_ Evaluation Dataset - sample 100.csv")
data_1100_df = pd.read_csv("/root/RnD_Project/inputs/Amazon RnD_ Evaluation Dataset - 1100 Final Dataset.csv")

data_100_df.iloc[:, 4] = data_100_df.iloc[:, 4].str.lower().str.strip()
data_1100_df.iloc[:, 4] = data_1100_df.iloc[:, 4].str.lower().str.strip()
data_100_df.iloc[:, 3] = data_100_df.iloc[:, 3].str.lower().str.strip()
data_1100_df.iloc[:, 3] = data_1100_df.iloc[:, 3].str.lower().str.strip()

loc = 13
    
fil_1100_df = data_1100_df[data_1100_df["Target Location"] == target_location_list[loc].lower().strip()]
fil_100_df = data_100_df[data_100_df["Target Location"] == target_location_list[loc].lower().strip()]

present_primary_key = []
absent_primary_key = []
for i in range(fil_1100_df.shape[0]):
    if fil_1100_df.iloc[i, 3] in fil_100_df.iloc[:, 3].tolist():
        index = data_1100_df[data_1100_df["Reference Entity"] == fil_1100_df.iloc[i, 3]].index
        present_primary_key.append(index[0])
    else:
        index = data_1100_df[data_1100_df["Reference Entity"] == fil_1100_df.iloc[i, 3]].index
        absent_primary_key.append(index[0])

final_df = data_1100_df.iloc[absent_primary_key, [0,2,3,4,5,8,10]]
final_df.to_excel(f"/root/RnD_Project/prompts/data_{loc}.xlsx", index=False)