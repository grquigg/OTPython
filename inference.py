import pandas as pd
import numpy as np
import math
import random
import copy
from predict import fit, predict_probabilities

random.seed(42)

def generate_cdf(probs):
    cdf = []
    val = 0
    for i in range(len(probs)):
        val+= probs[i]
        cdf.append(val)
    return cdf

def recursiveRandomSample(value, cdf, pdf):
    middle = math.floor(len(cdf) / 2)
    if(len(cdf) == 1):
        return pdf[0]
    if(value < cdf[middle]):
        return recursiveRandomSample(value, cdf[0:middle], pdf[0:middle])
    elif(value >= cdf[middle]):
        return recursiveRandomSample(value, cdf[middle:], pdf[middle:])
    
def generate_salad(consonants, grammar, probs, length=1000, max_length=3):
    salad = []
    if(len(grammar) == 0):
        for i in range(length):
            l = random.randint(1, max_length)
            str = ""
            for j in range(l):
                index = random.randint(0,len(consonants)-1)
                str += consonants[index] + " "
            salad.append(str[:-1])
    else:
        print(grammar)
        print(probs)
        nums = [i for i in range(len(probs))]
        cdf = generate_cdf(probs)
        for i in range(length):
            l = random.randint(1, max_length)
            str = ""
            for j in range(l):
                index = random.random()
                id = recursiveRandomSample(index, cdf, nums)
                elem = consonants[id]
                le = len(str.split(' ')) + len(elem.split(' '))
                while(le > max_length):
                    index = random.random()
                    id = recursiveRandomSample(index, cdf, nums)
                    elem = consonants[id]
                    le = len(str.split(' ')) + len(elem.split(' '))
                str += elem + " "
            salad.append(str[:-1])
    return salad

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

def featurize_data(data, consonants_dict):
    featurized_data = {}
    for d in data:
        print(d)
        vec = d.split(" ")
        featurized_data[d] = []
        for i in range(len(vec)):
            featurized_data[d] = np.concatenate((featurized_data[d], consonants_dict[vec[i]]))
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
constraints =  generate_constraint_dict(NUM_FEATURES)
possible_constraints = [key for key in constraints.keys()]

#TO-DO: Figure out whether there's a way to simplify constraint space a significant amount
#determine which constraints actually might have violations based on our data
#TO-DO: produce output file called blick

#recursive function for generating full list of possible combinations
#runs in approximately O(n^3) time, where n is the number of features
#obviously any speed up possible is going to result in improved performance

c_space = consonants[:,0].tolist()

def generate_candidate_constraints(possible_constraints, constraints, grammar, num=1000, max_length=3):
    constraint_list = {}
    while(len(constraint_list.keys()) < num): #this is sampling WITHOUT replacement
        length = random.randint(1, max_length)
        constraint = []
        vec = []
        for i in range(length):
            sample = random.randint(0, len(possible_constraints)-1)
            constraint.append(possible_constraints[sample])
            vec = np.concatenate((vec, constraints[possible_constraints[sample]]))
        constraint = tuple(constraint)
        if(constraint not in constraint_list and constraint not in grammar):
            constraint_list[constraint] = vec
    return constraint_list

def generate_constraint_violations(cans, candidates, features, salad):
    violations = np.zeros((len(salad), len(cans)))
    for j in range(len(cans)):
        constraint = candidates[cans[j]]
        #this is where things get hairy with the math
        for i in range(len(salad)):
            val = features[salad[i]]
            if(len(constraint) > len(val)):
                violations[i][j] = 0
                continue
            if(len(constraint) == len(val)):
                result = np.multiply(constraint, val)
                if(1 in result or 4 in result):
                    violation = True
                    for n in range(len(constraint)):
                        #if the two values are not the same and the values 
                        if(constraint[n] != val[n] and (constraint[n] == 1 or constraint[n] == 2)):
                            violation = False
                    if(violation):
                        violations[i][j] += 1
            else:
                for k in range(0, len(constraint), NUM_FEATURES):
                    #multiply the vector representations of constraint and value
                    #in the case of this specific problem, candidates that violate a constraint will be marked by 
                    #either 1 or 4 (1^2 and 2^2, respectively)
                    d = val[k:k+len(constraint)]
                    result = np.multiply(constraint, d)
                    #if 1 or 4 in result, then the constraint is violated
                    #however, it needs to violate all parts of the constraint
                    if(1 in result or 4 in result):
                        violation = True
                        for n in range(len(constraint)):
                            #if the two values are not the same and the values 
                            if(constraint[n] != d[n] and (constraint[n] == 1 or constraint[n] == 2)):
                                violation = False
                        if(violation):
                            violations[i][j] += 1
    return violations

def find_constraint(a, salad, candidates, consonants, freqs):
    features = featurize_data(salad, consonants_dict)

    cans = [key for key in candidates.keys()]
    violations = generate_constraint_violations(cans, candidates, features, salad)
    print(violations.shape)
    input = np.concatenate((freqs, violations), axis=1)
    feed = {"Input1": input}
    weights = np.ones((1000,))
    probs = predict_probabilities(weights, feed)
    expected_totals = np.dot(probs, violations)
    featurized_data = featurize_data(data[:,0], consonants_dict)
    observed_violations = generate_constraint_violations(cans, candidates, featurized_data, data[:,0])
    observed_totals = np.dot(data[:,1]/sum(data[:,1])*1000, observed_violations)
    #determine the best constraints given the set of conditions
    best_accuracy = a
    max_diff = 0
    accuracy = observed_totals - expected_totals
    print("Results")
    print(np.min(accuracy))
    constraint = np.argmin(accuracy)
    print(constraint)
    print(observed_violations[:,constraint])
    return constraint, observed_violations[:,constraint]

def get_unique(salad):
    unique_salad = []
    count = {}
    for word in salad:
        if(word not in count):
            count[word] = 0
            unique_salad.append(word)
        count[word] += 1
    return unique_salad, count

A = [0.001, 0.01, 0.1, 0.2, 0.3] #accuracy schedule
#algorithm as outlined in Hayes and Wilson 2008
#start with an empty grammar G
tableau = np.array(data[:,1]).reshape((len(data), 1))
input = {"Input1": tableau}
probs = np.full_like((len(data), 1), 1/len(data))
G = []
salad = generate_salad(c_space, G, probs)
# print(salad)
unique_salad, count = get_unique(salad)
freqs = [value for value in count.values()]
freqs = np.array(freqs).reshape((len(freqs), 1))
#for every accuracy level a in A

for a in A[:1]:
    candidate_constraints = generate_candidate_constraints(possible_constraints, constraints, G)
    #select the most general constraint with accuracy less than a and add it to the grammar
    constraint, violations = find_constraint(a, unique_salad, candidate_constraints, consonants_dict, freqs)
    constraint = constraint.item()
    violations = np.reshape(violations, (input["Input1"].shape[0], 1))
    input["Input1"] = np.concatenate((input["Input1"], violations), axis=1)
    print(input["Input1"].shape)
    assert input["Input1"].shape[1] == 2
    #fit model with new constraint
    weights = fit(input, num_constraints=1, mu_scalar=0, sigma_scalar=10)
    probs = predict_probabilities(weights, input)
    print(probs)
    while(constraint != None):
        G.append(constraint)
        salad = generate_salad(data[:,0], G, probs)
        unique_salad, count = get_unique(salad)
        freqs = [value for value in count.values()]
        freqs = np.array(freqs).reshape((len(freqs), 1))
        candidate_constraints = generate_candidate_constraints(possible_constraints, constraints, G)
        #select the most general constraint with accuracy less than a and add it to the grammar
        constraint, violations = find_constraint(a, unique_salad, candidate_constraints, consonants_dict, freqs)
        break

    