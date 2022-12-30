import pandas as pd
import numpy as np
import copy
from scipy.special import softmax
from scipy.optimize import minimize
def load_data(file, delimiter='\t'):
    data = pd.read_csv(file, delimiter=delimiter, header=None)

    full_names = data.iloc[0][3:]
    abbr_names = data.iloc[1][3:]
    candidate_rows = data[data[1].notnull()]
    candidate_rows = candidate_rows.to_numpy()
    full_candidates = candidate_rows[:,2:].astype(float)
    candidates = {}
    for i in range(len(candidate_rows)):
        if(pd.isna(candidate_rows[i][0])):
            candidate_rows[i,0] = candidate_rows[i-1,0]
    for j in range(len(candidate_rows)):
        if(candidate_rows[j,0] not in candidates):
            candidates[candidate_rows[j,0]] = []
        candidates[candidate_rows[j,0]].append(candidate_rows[j,2:])

    return full_names, abbr_names, candidates, full_candidates

def compute_probabilities(constraint_weights, data):
    data = np.array(data).astype(float)
    freqs = data[:,2:]
    h = -np.dot(constraint_weights, data[:,0:2])
    h_soft = softmax(h)

def calculate_data_likelihood(constraint_weights, data):
    data = np.array(data).astype(float)
    freqs = data[:,2:]
    h = -np.dot(constraint_weights, data[:,0:2])
    h_soft = softmax(h)
    log_candidate_probs = np.log(h_soft)
    log_probs = np.sum(np.dot(log_candidate_probs, freqs))
    return log_probs, h_soft

def compute_gradient(weights, probs, data, bias_params):
    freqs = data[:,2:]
    constraint_data = data[:,:2]
    dot = np.dot(np.transpose(probs), constraint_data)
    mult = np.multiply(freqs, constraint_data)
    total_constraints = np.sum(mult, axis=0)
    loss = total_constraints - dot[0]
    if(bias_params != None):
        mus = bias_params[0]
        sigmas = bias_params[1]
        top = (weights - mus)
        bottom = 2 * sigmas**2
        bias = np.sum(top / bottom)
        loss += bias
    return loss

def calculate_bias(mus, sigmas, constraint_weights):
    top = (constraint_weights - mus)**2
    bottom = 2 * sigmas**2
    bias = np.sum(top / bottom)
    return bias

def log_likelihood(constraint_weights, args, bias_params):
    l = 0
    data = args[0]
    full_data = args[1]
    total_probs = np.empty((0,1))
    for key, value in data.items():
        loss, probs = calculate_data_likelihood(constraint_weights, value)
        l += loss
        total_probs = np.concatenate([total_probs, np.transpose(probs)])
    if(bias_params != None):
        l += - calculate_bias(bias_params[0], bias_params[1], constraint_weights)
    return l, total_probs

def convert(weights, data):
    weights = weights.flatten()
    dat = []
    for key, value in data.items():
        dat += value
    pass

def load_bias_file(bias_file, sep="\t"):
    inBias = pd.read_csv(bias_file, delimiter=sep, header=None)
    inBias = inBias.to_numpy()
    print(inBias)
    mus = inBias[:,1]
    print(mus)
    sigmas = inBias[:,2]
    print(sigmas)
    return (mus, sigmas)

def optimize(input_file, bias_file=None, constraint_weights=None, learning_rate=0.1, iterations=1000):

    input = load_data(input_file)
    if(bias_file != None):
        bias_params = load_bias_file(bias_file)
    else:
        bias_params = None
    long_names = input[0]
    abbr_names = input[1]
    data = input[2]
    full_data = input[3]
    weights = np.ones((1, len(long_names)))

    #optimization code
    for i in range(iterations):
        log, probs = log_likelihood(weights, (data, full_data,), bias_params)
        print(log)
        gradient = compute_gradient(weights, probs, full_data, bias_params)
        weights = weights - learning_rate * gradient
    print(weights)


optimize("sample_data_file.txt", bias_file="sample_constraint_file.txt")