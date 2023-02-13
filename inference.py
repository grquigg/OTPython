import pandas as pd
import numpy as np
import math
def generate_constraint_dict(num_features, max_features=3): #these are only going to be one consonant at a time
    pass
    constraints = {}
    for i in range(len(num_features)):
        for j in range(len(num_features)):
            for k in range(len(num_features)):
                array = [0 for i in range(num_features)]

def convert_to_base_3(vector, num_features):
    pass
consonants_df = pd.read_csv("EnglishFeatures.txt", delimiter='\t') #this is the **feature** dataset
consonant_headers = consonants_df.columns
dataset = []
data_df = pd.read_csv("./data.csv", header=None, delimiter=',')
data = data_df.to_numpy()
likelihoods = [entry[1] for entry in data]
likelihoods = sorted(likelihoods, reverse=True)
counts = likelihoods.copy()
likelihoods = likelihoods / np.sum(likelihoods)
print(len(consonant_headers))