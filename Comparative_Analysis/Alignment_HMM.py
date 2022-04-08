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


    
class Alignment_HMM:
    
    def __init__(self, num_symbols, num_states, alignment_dir, master_species_id, species_order = []):
        self.initial_state_probabilities = [1.0/num_states]*num_states
        self.num_symbols = num_symbols 
        self.num_states = num_states
        alignment_ids = util.list_files(alignment_dir + '/')
        self.alignment_ids = [int(i.split('.')[0]) for i in alignment_ids]
        self.alignment_dict = {}
        for group_id in tqdm(self.alignment_ids):
            alignment = align.Alignment(alignment_dir + '/' + str(group_id) + '.fasta', master_species_id, 'NT', insert_symbol = '-', species_order = species_order)
            alignment.modify_sequence(1, False, False)
            self.alignment_dict[group_id] = alignment
        
    def sum_logs(self, p, q):
        if p>9999 and q>99999:
            ans = math.log(math.exp(p) + math.exp(q))
        else:
            if p > q:
                ans =  p + math.log(1 + math.exp(q - p))
            else:
                ans =  q + math.log(1 + math.exp(p - q))
        return ans
    
    def calculate_observation_probs(self, mutation_probabilities, alignment_list, alignment, all_species = True, comparison_species = ''):    
        align_list =  alignment_list
        len_align_list = len(alignment_list[0])
        num_sequences = len(alignment_list)
        observation_probs =  np.zeros((self.num_states, len_align_list))
        master_species_index = alignment.master_species_index
        for i in range(len_align_list):
            master_species_symbol = alignment_list[master_species_index][i]
            for s in range(self.num_states):
                temp_1 = 0
                temp_2 = 0
                for k in range(num_sequences):
                    if k == master_species_index or (all_species == False and not (k == alignment.species_index(comparison_species))):
                        continue
                    else:
                        aligned_symbol = alignment_list[k][i]
                        temp_1 += 1
                        if aligned_symbol == master_species_symbol:
                            temp_2 += 1
                observation_probs[s, i] =  binom.pmf(temp_2,  temp_1, mutation_probabilities[s])
        return observation_probs

    def calculate_match_probs(self, alignment_list, alignment, all_species = True, comparison_species = ''):    
        align_list =  alignment_list
        len_align_list = len(alignment_list[0])
        num_sequences = len(alignment_list)
        match_probs = []
        master_species_index = alignment.master_species_index
        for i in range(len_align_list):
            master_species_symbol = alignment_list[master_species_index][i]
            total_count = 0
            matches = 0
            for k in range(num_sequences):
                if k == master_species_index or (all_species == False and not (k == alignment.species_index(comparison_species))):
                    continue
                else:
                    total_count += 1
                    aligned_symbol = alignment_list[k][i]
                    if aligned_symbol == master_species_symbol:
                        matches += 1
            match_probs.append(matches/total_count)
        return match_probs
    
    def convert_alignment_hmm_to_parameters(self, transition_probabilities, mutation_probabilities):
        return [transition_probabilities[0,0], transition_probabilities[0,1]/(1-transition_probabilities[0,0]), transition_probabilities[1,1], transition_probabilities[1,0]/(1-transition_probabilities[1,1]), 
                transition_probabilities[2,2], transition_probabilities[2,0]/(1-transition_probabilities[2,2]), mutation_probabilities[0], mutation_probabilities[1], mutation_probabilities[2]]
    
    def alignment_hmm_model_inputs(self, params):
        a = params[0]
        b = (1-params[0])*(params[1])
        c = 1-a-b
        e = params[2]
        d = (1-params[2])*(params[3])
        f = 1-e-d
        i = params[4]
        g = (1-params[4])*(params[5])
        h = 1 - i - g
        transition_probabilities = np.array([[a,b,c],[d,e,f],[g,h,i]])
        mutation_probabilities = params[6:]
        return transition_probabilities, mutation_probabilities

    def alignment_hmm_log_likelihood(self, params, num_subsets, subset_num, offset, min_length, all_species = True, comparison_species = ''):
        total_probability = 0
        transition_probabilities, mutation_probabilities = self.alignment_hmm_model_inputs(params)
        ids = util.chunk_list(self.alignment_ids, num_subsets, subset_num)
        for group_id in ids:
            alignment = self.alignment_dict[group_id]
            align_list =  alignment.modified_sequence_list
            align_names = alignment.sequence_names
            len_align_list = len(align_list[0])
            non_cds = [x[offset:len_align_list - offset] for x in align_list]
            if len(non_cds[0]) < min_length:
                continue
            observation_probabilities = self.calculate_observation_probs(mutation_probabilities, non_cds, alignment, all_species, comparison_species)
            hm_model = hmm.HMM(self.initial_state_probabilities, transition_probabilities, observation_probabilities, termination = False)
            hm_model.calculate_probabilities()
            total_probability += hm_model.forward_ll * -1
        return total_probability

    def EM_update_parameters(self, num_subsets, subset_num, offset, min_length, mutation_probabilities, transition_probabilities, all_species, comparison_species):
        ids = util.chunk_list(self.alignment_ids, num_subsets, subset_num)
        total_probability = 0
        transition_counts = np.zeros((self.num_states, self.num_states))
        match_emission_counts = np.zeros(self.num_states)
        match_total_counts = np.zeros(self.num_states)
        for group_id in ids:
            alignment = self.alignment_dict[group_id]
            align_list =  alignment.modified_sequence_list
            align_names = alignment.sequence_names
            len_align_list = len(align_list[0])
            non_cds = [x[offset:len_align_list - offset] for x in align_list]
            if len(non_cds[0]) < min_length:
                continue
            match_probs =  self.calculate_match_probs(non_cds, alignment, all_species, comparison_species)    
            observation_probabilities = self.calculate_observation_probs(mutation_probabilities, non_cds, alignment, all_species, comparison_species)
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
                    match_emission_counts[s] += hm_model.state_probabilities[s][i] * match_probs[i]
                    match_total_counts[s] += hm_model.state_probabilities[s][i]
        return transition_counts, match_emission_counts, match_total_counts, total_probability
    
    def EM_update(self, num_subsets, params, offset, min_length, all_species = True, comparison_species = ''):
        subset_numbers = list(range(1, num_subsets+1))
        for iternum in tqdm(range(300)):
            total_probability = 0
            
            if iternum == 0:
                transition_probabilities, mutation_probabilities = self.alignment_hmm_model_inputs(params)
            else:
                transition_probabilities = transition_counts
                mutation_probabilities = match_emission_counts
            parallel_output = Parallel(n_jobs=-1)(delayed(self.EM_update_parameters)(num_subsets, subset_num, offset, min_length, mutation_probabilities, transition_probabilities, all_species, comparison_species) for subset_num in subset_numbers)
            transition_counts = np.zeros((self.num_states, self.num_states))
            match_emission_counts = np.zeros(self.num_states)
            match_total_counts = np.zeros(self.num_states)
            for i in range(len(parallel_output)):
                for s in range(self.num_states):
                    for t in range(self.num_states):
                        transition_counts[s,t] += (parallel_output[i][0])[s,t]
                    match_emission_counts[s] += (parallel_output[i][1])[s]
                    match_total_counts[s] += (parallel_output[i][2])[s]
                total_probability += parallel_output[i][3]
            
            for s in range(self.num_states):
                temp_1 = 0
                for t in range(self.num_states):
                    temp_1 += transition_counts[s, t]
                for t in range(self.num_states):
                    transition_counts[s, t] = transition_counts[s, t] / temp_1
            for s in range(self.num_states):
                match_emission_counts[s] = match_emission_counts[s]  / match_total_counts[s]
            if iternum > 1 and ((abs(total_probability - prev_total_probability) < 0.001) or (total_probability > prev_total_probability)):
                break
            prev_total_probability = total_probability

        print(transition_counts, match_emission_counts, total_probability)
        return(transition_counts, match_emission_counts, total_probability, self.convert_alignment_hmm_to_parameters(transition_counts, match_emission_counts))
    
    