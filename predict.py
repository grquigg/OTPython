import pandas as pd
import numpy as np
import copy
import argparse
from scipy.special import softmax
from scipy.optimize import minimize
np.set_printoptions(suppress=True)
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
    freqs = data[:,0:1]
    h = -np.dot(constraint_weights, data[:,0:2])
    h_soft = softmax(h)

def calculate_data_likelihood(constraint_weights, data):
    data = np.array(data).astype(float)
    freqs = data[:,0:1]
    h = -np.dot(data[:,1:], constraint_weights)
    h_soft = softmax(h)
    log_candidate_probs = np.log(h_soft)
    log_probs = np.sum(np.dot(log_candidate_probs, freqs))
    return log_probs, h_soft
    
def calculate_bias(mus, sigmas, constraint_weights):
    top = (constraint_weights - mus)**2
    bottom = 2 * sigmas**2
    bias = np.sum(top / bottom)
    return bias

def log_likelihood(constraint_weights, data, bias_params):
    l = 0
    total_probs = []
    for key, value in data.items():
        loss, probs = calculate_data_likelihood(constraint_weights, value)
        l += loss
        total_probs = np.concatenate([total_probs, probs])
    if(bias_params != None):
        l += - calculate_bias(bias_params[0], bias_params[1], constraint_weights)
    return -l

def predict_probabilities(constraint_weights, data):
    total_probs = []
    for key, value in data.items():
        loss, probs = calculate_data_likelihood(constraint_weights, value)
        total_probs = np.concatenate([total_probs, probs])
    return total_probs

def load_bias_file(bias_file, sep="\t"):
    inBias = pd.read_csv(bias_file, delimiter=sep, header=None)
    inBias = inBias.to_numpy()
    print(inBias)
    mus = inBias[:,1]
    print(mus.shape)
    sigmas = inBias[:,2]
    print(sigmas.shape)
    return (mus, sigmas)

def process_bias_arguments(bias_file, mu_scalar, mu_vector, sigma_scalar, sigma_vector, num_constraints):
    bias_params = None
    if(bias_file != None):
        if(not(mu_scalar == None and mu_vector == None and sigma_scalar == None and sigma_vector == None)):
            raise ValueError("Both bias file and scalar constraints were provided. Ignoring the scalars and using parameters from file")
        bias_params = load_bias_file(bias_file)
    elif((mu_vector != None or mu_scalar != None) and (sigma_scalar != None or sigma_vector != None)):
        bias_params = [np.ndarray((num_constraints,)), np.ndarray((num_constraints,))]
        if(mu_vector != None):
            if(mu_scalar != None):
                raise ValueError("Ignoring scalar value and using vector parameters")
        
            if(len(mu_vector) != num_constraints):
                raise ValueError("This does not work")
            bias_params[0] = np.array(mu_vector)
        else:
            mu = np.full((num_constraints,), mu_scalar)
            bias_params[0] = mu
        
        if(sigma_vector != None):
            if(sigma_scalar != None):
                raise ValueError("Ignoring scalar avlue and using vector parameters")
            if(len(sigma_vector) != num_constraints):
                raise ValueError("Bad value")
            bias_params[1] = np.array(sigma_vector)
        else:
            sigma = np.full((num_constraints,), sigma_scalar)
            bias_params[1] = sigma
    elif(mu_vector != None or mu_scalar != None or sigma_scalar != None or sigma_vector != None):
        raise ValueError("Must specify values for both sigma and mu")
    else:
        print("Proceed with no mu or sigma")
    return bias_params

def fit(input_data, num_constraints=None, constraint_weights=None, mu_scalar=None, mu_vector=None, sigma_scalar=None, sigma_vector=None):
    if(constraint_weights != None):
        weights = constraint_weights
        num_constraints = len(weights)
    else:
        weights = np.ones((1, num_constraints))

    bias_params = process_bias_arguments(None, mu_scalar, mu_vector, sigma_scalar, sigma_vector, num_constraints)

    result = minimize(log_likelihood, weights, args=(input_data,bias_params,))
    new_weights = np.reshape(result.x, (1,num_constraints))
    probs = predict_probabilities(result.x, input_data)
    return result.x

def optimize(input_file, bias_file=None, constraint_weights=None, mu_scalar=None, mu_vector=None, sigma_scalar=None, sigma_vector=None):
    input = load_data(input_file)
    #variables
    long_names = input[0]
    num_constraints = len(long_names)
    abbr_names = input[1]
    data = input[2]
    full_data = input[3]
    #bias params
    bias_params = process_bias_arguments(bias_file, mu_scalar, mu_vector, sigma_scalar, sigma_vector, num_constraints)
    if(constraint_weights != None):
        weights = constraint_weights
    else:
        weights = np.ones((1, num_constraints))
    #optimization code
    result = minimize(log_likelihood, weights, args=(data,bias_params,))
    new_weights = np.reshape(result.x, (1,len(long_names)))
    probs = predict_probabilities(result.x, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--constraint_file", type=str, help="Input data file for contraints")
    parser.add_argument("-f", "--bias_file")
    args = parser.parse_args()
    optimize(args.constraint_file, mu_scalar=0, sigma_scalar=10, bias_file=args.bias_file)