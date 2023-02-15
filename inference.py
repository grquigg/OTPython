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

constraints =  generate_constraint_dict(len(consonant_headers)-1)
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
violation_table = np.zeros((len(consonants), len(possible_constraints)), dtype=int)
#map consonants
consonants_dict = {}
for consonant in consonants:
    consonants_dict[consonant[0]] = consonant[1:]

#evaluate constraint violations
for j in range(len(possible_constraints)):
    value = constraints[possible_constraints[j]]
    #this is where things get hairy with the math
    for i in range(len(consonants)):
        val = consonants[i][1:]
        #multiply the vector representations of constraint and value
        #in the case of this specific problem, candidates that violate a constraint will be marked by 
        #either 1 or 4 (1^2 and 2^2, respectively)
        result = np.multiply(value, val)
        #if 1 or 4 in result, then the constraint is violated
        #however, it needs to violate all parts of the constraint
        if(1 in result or 4 in result):
            violation = True
            for i in range(len(value)):
                if(value[i] != val[i] and (value == 1 or value == 2)):
                    violation = False
            if(violation):
                violation_table[i][j] += 1
print("DONE")
sample = 5
print(possible_constraints[sample])
print(constraints[possible_constraints[sample]])
print(violation_table[:,sample])
