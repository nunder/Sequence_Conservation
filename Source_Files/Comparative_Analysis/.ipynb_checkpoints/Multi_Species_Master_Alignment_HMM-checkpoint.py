import os
import pandas as pd
import subprocess
import seaborn as sns
import shutil
from tqdm import tqdm
import numpy as np
from Bio import Entrez, SeqIO, AlignIO, pairwise2, Align, Seq, motifs
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.Align.Applications import MuscleCommandline
from pathlib import Path
from joblib import Parallel, delayed
import random
from statistics import mean, stdev
import math
from scipy import linalg
import scipy.stats as ss
from scipy.stats import binom
from . import Utilities as util
from . import HMM as hmm
from . import Sequence_Analysis_Routines as sar
from . import Alignment as align


    
class Multi_Species_Master_Alignment_HMM:
    
    def __init__(self, pairwise_observations):
        self.pairwise_observation_dictionary = {}
        self.num_states = 2
        self.initial_state_probabilities = [1.0/self.num_states]*self.num_states
        self.observation_ids = list(range(len(pairwise_observations)))
        for i in self.observation_ids:
            observation = pairwise_observations[i]
            temp = np.zeros((len(observation),2 , len(observation[0][0])))
            for j in range(len(observation)):
                for l in range(len(observation[0][0])):
                    temp[j, 0, l] = observation[j][0][l]
                    temp[j, 1, l] = observation[j][1][l] + observation[j][2][l]  
            self.pairwise_observation_dictionary[i] = temp
        
    def sum_logs(self, p, q):
        if p>9999 and q>99999:
            ans = math.log(math.exp(p) + math.exp(q))
        else:
            if p > q:
                ans =  p + math.log(1 + math.exp(q - p))
            else:
                ans =  q + math.log(1 + math.exp(p - q))
        return ans
    
    def calculate_observation_probs(self, mutation_probabilities, observations):    
        num_sequences = len(observations)
        num_observations = len(observations[0][0])
        observation_probs =  np.zeros((self.num_states, num_observations))
        for i in range(num_observations):
            for s in range(self.num_states):
                temp = 1
                for k in range(num_sequences):
                    temp = temp * (mutation_probabilities[s][k]*observations[k][0][i] + (1-mutation_probabilities[s][k])*observations[k][1][i])
                observation_probs[s, i] =  temp
        return observation_probs

    def calculate_match_probs(self, observations):    
        num_sequences = len(observations)
        num_observations = len(observations[0][0])
        match_probs = []
        for i in range(num_observations):
            matches = []
            for k in range(num_sequences):
                matches.append(observations[k][0][i])
            match_probs.append(matches)
        return match_probs
    
    def convert_alignment_hmm_to_parameters(self, transition_probabilities, mutation_probabilities):
        return [transition_probabilities[0,0], transition_probabilities[1,1], mutation_probabilities[0], mutation_probabilities[1]]
    
    def alignment_hmm_model_inputs(self, params):
        a = params[0]
        b = 1 - a
        d = params[1]
        c = 1-d
        transition_probabilities = np.array([[a,b],[c,d]])
        mutation_probabilities = params[2:]
        return transition_probabilities, mutation_probabilities

    def alignment_hmm_log_likelihood(self, params, num_subsets, subset_num, offset, min_length):
        total_probability = 0
        transition_probabilities, mutation_probabilities = self.alignment_hmm_model_inputs(params)
        ids = util.chunk_list(self.observation_ids, num_subsets, subset_num)
        for group_id in ids:
            observations = self.pairwise_observation_dictionary[group_id]
            num_sequences = len(observations)
            num_observations = len(observations[0][0])
            non_cds = observations[:,:,offset:(num_observations - offset)]  
            if num_observations < min_length + 2 * offset:
                continue
            observation_probabilities = self.calculate_observation_probs(mutation_probabilities, non_cds)
            hm_model = hmm.HMM(self.initial_state_probabilities, transition_probabilities, observation_probabilities, termination = False)
            hm_model.calculate_probabilities()
            total_probability += hm_model.forward_ll * -1
        return total_probability

    def EM_update_parameters(self, num_subsets, subset_num, offset, min_length, mutation_probabilities, transition_probabilities):
        ids = util.chunk_list(self.observation_ids, num_subsets, subset_num)
        total_probability = 0
        transition_counts = np.zeros((self.num_states, self.num_states))
        match_emission_counts = np.zeros((self.num_states, len(mutation_probabilities[0])))
        match_total_counts = np.zeros((self.num_states, len(mutation_probabilities[0])))
        for group_id in ids:
            observations = self.pairwise_observation_dictionary[group_id]
            num_sequences = len(observations)
            num_observations = len(observations[0][0])
            non_cds = observations[:,:,offset:(num_observations - offset)] 
            if num_observations <= min_length + 2 * offset:
                continue
            match_probs =  self.calculate_match_probs(non_cds)    
            observation_probabilities = self.calculate_observation_probs(mutation_probabilities, non_cds)
            observation_length = observation_probabilities.shape[1]
            hm_model = hmm.HMM(self.initial_state_probabilities, transition_probabilities, observation_probabilities, termination = False)
            hm_model.calculate_probabilities()
            total_probability += hm_model.forward_ll * -1
            prob_observation = hm_model.forward_ll
            for s in range(self.num_states):
                for t in range(self.num_states):
                    temp = 0
                    for i in range(observation_length - 1):
                        if i == 0:
                            temp = hm_model.forward_probabilities[s, i] + math.log(transition_probabilities[s, t]) + math.log(observation_probabilities[t, i+1]) + hm_model.backward_probabilities[t, i+1]
                        else:
                            temp = self.sum_logs(temp, hm_model.forward_probabilities[s, i] + math.log(transition_probabilities[s, t]) + math.log(observation_probabilities[t, i+1]) + hm_model.backward_probabilities[t, i+1])
                    transition_counts[s, t] += math.exp(temp - prob_observation)

            for s in range(self.num_states):
                for i in range(observation_length - 1):
                    for k in range(num_sequences):
                        match_emission_counts[s][k] += hm_model.state_probabilities[s][i] * match_probs[i][k]
                        match_total_counts[s][k] += hm_model.state_probabilities[s][i]
                 
        return transition_counts, match_emission_counts, match_total_counts, total_probability
    
    def EM_update(self, num_subsets, params, offset, min_length):
        subset_numbers = list(range(1, num_subsets+1))
        for iternum in tqdm(range(300)):
            total_probability = 0
            
            if iternum == 0:
                transition_probabilities, mutation_probabilities = self.alignment_hmm_model_inputs(params)
            else:
                transition_probabilities = transition_counts
                mutation_probabilities = match_emission_counts
            num_sequences = len(mutation_probabilities[0])
            parallel_output = Parallel(n_jobs=-1)(delayed(self.EM_update_parameters)(num_subsets, subset_num, offset, min_length, mutation_probabilities, transition_probabilities) 
                                                  for subset_num in subset_numbers)
            #parallel_output = []
            #for subset_num in subset_numbers:
            #    parallel_output.append(self.EM_update_parameters(num_subsets, subset_num, offset, min_length, mutation_probabilities, transition_probabilities))
            transition_counts = np.zeros((self.num_states, self.num_states))
            match_emission_counts = np.zeros((self.num_states, num_sequences))
            match_total_counts = np.zeros((self.num_states, num_sequences))
            for i in range(len(parallel_output)):
                for s in range(self.num_states):
                    for t in range(self.num_states):
                        transition_counts[s,t] += (parallel_output[i][0])[s,t]
                    for k in range(num_sequences):
                        match_emission_counts[s][k] += (parallel_output[i][1])[s][k]
                        match_total_counts[s][k] += (parallel_output[i][2])[s][k]
                total_probability += parallel_output[i][3]
            
            for s in range(self.num_states):
                temp_1 = 0
                for t in range(self.num_states):
                    temp_1 += transition_counts[s, t]
                for t in range(self.num_states):
                    transition_counts[s, t] = transition_counts[s, t] / temp_1
            for s in range(self.num_states):
                for k in range(num_sequences):
                    match_emission_counts[s][k] = match_emission_counts[s][k]  / match_total_counts[s][k]
            if iternum > 1 and ((abs(total_probability - prev_total_probability) < 0.001) or (total_probability > prev_total_probability)):
                break
            prev_total_probability = total_probability
        return(transition_counts, match_emission_counts, total_probability, self.convert_alignment_hmm_to_parameters(transition_counts, match_emission_counts))
    
    