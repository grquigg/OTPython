import pandas as pd
import numpy as np
import math
import random

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

def generate_full_constraint_space(constraints, possible_constraints):
    constraint_space = {}
    for i in range(len(possible_constraints)):
        for j in range(len(possible_constraints)):
            for k in range(len(possible_constraints)):
                value = (constraints[possible_constraints[i]] + 
                         constraints[possible_constraints[j]] +
                         constraints[possible_constraints[k]])
                constraint_space[(i,j,k)] = value

def convert_vector_to_num(vector, num_features):
    total = 0
    base = 0
    for i in range(num_features, 0, -1):
        total += (vector[i-1] * 3**base)
        base += 1
    return total

def featurize_data(data, consonants_dict):
    featurized_data = {}
    for d in data:
        vec = d[0].split(" ")
        featurized_data[d[0]] = []
        for i in range(len(vec)):
            featurized_data[d[0]] = np.concatenate((featurized_data[d[0]], consonants_dict[vec[i]]))
    return featurized_data

consonants_df = pd.read_csv("EnglishFeatures.txt", delimiter='\t') #this is the **feature** dataset
consonant_headers = consonants_df.columns
dataset = []
data_df = pd.read_csv("EnglishLearningData.txt", header=None, delimiter='\t')
data = data_df.to_numpy()
consonants = consonants_df.to_numpy()
likelihoods = [entry[1] for entry in data]
likelihoods = sorted(likelihoods, reverse=True)
counts = likelihoods.copy()
likelihoods = likelihoods / np.sum(likelihoods)
for i in range(consonants.shape[0]):
    for j in range(consonants.shape[1]):
        if(consonants[i][j] == "-"):
            consonants[i][j] = 2
        elif(consonants[i][j] == "+"):
            consonants[i][j] = 1
        elif(consonants[i][j] == "0"):
            consonants[i][j] = 0
#map consonants
consonants_dict = {}
for consonant in consonants:
    consonants_dict[consonant[0]] = consonant[1:]

featurized = featurize_data(data, consonants_dict)
constraints =  generate_constraint_dict(len(consonant_headers)-1)
possible_constraints = [key for key in constraints.keys()]
#this takes way too long
# full_constraints = generate_full_constraint_space(constraints, possible_constraints)
#convert data features
violation_table = np.zeros((len(data), len(possible_constraints)), dtype=int)


#evaluate constraint violations
for j in range(len(possible_constraints)):
    value = constraints[possible_constraints[j]]
    #this is where things get hairy with the math
    for i in range(len(data)):
        val = featurized[data[i][0]]
        #multiply the vector representations of constraint and value
        #in the case of this specific problem, candidates that violate a constraint will be marked by 
        #either 1 or 4 (1^2 and 2^2, respectively)
        result = np.multiply(value, val)
        #if 1 or 4 in result, then the constraint is violated
        #however, it needs to violate all parts of the constraint
        if(1 in result or 4 in result):
            violation = True
            for i in range(len(value)):
                #if the two values are not the same and the values 
                if(value[i] != val[i] and (value[i] == 1 or value[i] == 2)):
                    violation = False
            if(violation):
                violation_table[i][j] += 1
print("DONE")
sample = random.randint(1,1000)
total_violations = np.sum(violation_table, axis=0)
print(total_violations)
#now we get the full constraint space

#TO-DO: Figure out whether there's a way to simplify constraint space at least a small amount

