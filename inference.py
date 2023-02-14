import pandas as pd
import numpy as np
import math

def generate_combinations(constraints, i, j, k, num_features):
    for p in range(3):
        for q in range(3):
            for r in range(3):
                array = [0 for i in range(num_features)]
                array[i] = p
                array[j] = q
                array[k] = r
                num_hash = convert_vector_to_num(array, num_features)
                if(num_hash not in constraints):
                    constraints[num_hash] = array
def generate_constraint_dict(num_features, max_features=3): #these are only going to be one consonant at a time
    pass
    constraints = {}
    for i in range(num_features):
        for j in range(num_features):
            for k in range(num_features):
                generate_combinations(constraints, i, j, k, num_features)
    return constraints
def convert_vector_to_num(vector, num_features):
    total = 0
    base = 0
    for i in range(num_features, 0, -1):
        total += (vector[i-1] * 3**base)
        base += 1
    return total
consonants_df = pd.read_csv("EnglishFeatures.txt", delimiter='\t') #this is the **feature** dataset
consonant_headers = consonants_df.columns
dataset = []
data_df = pd.read_csv("./data.csv", header=None, delimiter=',')
data = data_df.to_numpy()
consonants = consonants_df.to_numpy()
likelihoods = [entry[1] for entry in data]
likelihoods = sorted(likelihoods, reverse=True)
counts = likelihoods.copy()
likelihoods = likelihoods / np.sum(likelihoods)

print(convert_vector_to_num([0,0,0,1], 4))
print(convert_vector_to_num([2,2,2,2], 4))
constraints =  generate_constraint_dict(len(consonant_headers))
possible_constraints = [key for key in constraints.keys()]

#convert data features
for i in range(consonants.shape[0]):
    for j in range(consonants.shape[1]):
        if(consonants[i][j] == "-"):
            consonants[i][j] = 2
        elif(consonants[i][j] == "+"):
            consonants[i][j] = 1
        elif(consonants[i][j] == "0"):
            consonants[i][j] = 0
print(consonants)