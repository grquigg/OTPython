import pandas as pd
import numpy as np
import math
import random
import copy
def generate_salad(cdf, pdf, grammar, length=1000):
    salad = []
    for i in range(length):
        index = random.random()
        entry = recursiveRandomSample(index, cdf, pdf)
        salad.append(entry)
    return salad

def recursiveRandomSample(value, cdf, pdf):
    middle = math.floor(len(cdf) / 2)
    if(len(cdf) == 1):
        return pdf[0]
    if(value < cdf[middle]):
        return recursiveRandomSample(value, cdf[0:middle], pdf[0:middle])
    elif(value >= cdf[middle]):
        return recursiveRandomSample(value, cdf[middle:], pdf[middle:])
    
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
#don't include "repeat" constraints
def generate_full_constraint_space(constraints, possible_constraints):
    constraint_space = {}
    #TO-DO: make this more efficient

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
NUM_FEATURES = len(consonant_headers)-1
featurized = featurize_data(data, consonants_dict)
constraints =  generate_constraint_dict(NUM_FEATURES)
possible_constraints = [key for key in constraints.keys()]
#this is formally a stack

#this takes way too long
# full_constraints = generate_full_constraint_space(constraints, possible_constraints)
#convert data features
violation_table = np.zeros((len(data), len(possible_constraints)), dtype=int)


#evaluate constraint violations
for j in range(2, 3):
    value = constraints[possible_constraints[j]]
    #this is where things get hairy with the math
    for i in range(len(data)):
        val = featurized[data[i][0]]
        print(val)
        for k in range(0, len(val), NUM_FEATURES):
            #multiply the vector representations of constraint and value
            #in the case of this specific problem, candidates that violate a constraint will be marked by 
            #either 1 or 4 (1^2 and 2^2, respectively)
            print(val[k:k+NUM_FEATURES])
            print(value)
            result = np.multiply(value, val[k:k+NUM_FEATURES])
            print("Result")
            print(result)
            #if 1 or 4 in result, then the constraint is violated
            #however, it needs to violate all parts of the constraint
            if(1 in result or 4 in result):
                violation = True
                for n in range(len(value)):
                    #if the two values are not the same and the values 
                    if(value[n] != val[n] and (value[n] == 1 or value[n] == 2)):
                        violation = False
                if(violation):
                    print("violation")
                    violation_table[i][j] += 1
print("DONE")
total_violations = np.sum(violation_table, axis=0)
print(total_violations)

#TO-DO: Figure out whether there's a way to simplify constraint space a significant amount
#determine which constraints actually might have violations based on our data
#TO-DO: produce output file called blick

def generate_possible_combinations(consonants, combos_dict, max_length=3):
    full_combos = copy.deepcopy(combos_dict)
    c_space = consonants[:,0].tolist()
    print(c_space)
    #treat the list of possible combinations like a stack
    current_combos = len(c_space)-1
    index = len(c_space)-1
    #ensure that everything is as it should be
    for cons in c_space:
        if(cons not in full_combos):
            full_combos[cons] = consonants_dict[cons]
    while(index >= 0):
        #look through the entirety of the stack to see if there's any combinations we can have
        current_val = c_space[index]
        for i in range(len(c_space)-1, 0, -1):
            string = current_val + " " + c_space[i]
            if(len(string.split(' ')) > max_length):
                continue
            vec = np.concatenate((full_combos[current_val], full_combos[c_space[i]]))
            if(string not in c_space):
                c_space.append(string)
            if(string not in full_combos):
                full_combos[string] = vec
        index -= 1


    pass
possible_combos = generate_possible_combinations(consonants, featurized, max_length=3)
combination_pdf = []
combination_cdf = []
total = 0
#randomly generate constraints
# print(len(consonants[0]))
# print(consonants_dict)
# num_features = 15
# num_candidate_features = 100
# for key in possible_combos.keys():
#     combination_pdf.append(key)
#     combination_cdf.append(total)
#     total += (1/len(possible_combos))
# G = []
# salad = generate_salad(combination_cdf, combination_pdf, G)
# print(salad[0])