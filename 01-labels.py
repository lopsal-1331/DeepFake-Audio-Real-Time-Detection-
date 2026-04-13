'''
Generate models: training, test and threshold-test
Objectives: 
    - Load metadata
    - Classify among the three classes based on the metadata
    - Search independent audios for each set
        - Training
        - Test 
        - Threshold test 
'''


# imports 
import os 
import sys 
import pandas as pd 
from pathlib import Path 
from tqdm import tqdm 
