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
from . import Utilities as util
    
class HMM:
    def __init__(self, initial_state_probabilities, transition_probabilities, observation_probabilities, termination = False):
        self.initial_state_probabilities = initial_state_probabilities
        self.transition_probabilities = transition_probabilities
        self.observation_probabilities = observation_probabilities
        self.num_states = observation_probabilities.shape[0]
        self.observation_length = observation_probabilities.shape[1]
        self.viterbi_path = np.zeros(self.observation_length, dtype='int16')
        self.viterbi_probability = 0
        self.forward_probabilities = []
        self.backward_probabilities = []
        self.state_probabilities = []
        self.forward_ll = 0
        self.backward_ll = 0
    
    def viterbi(self):
        max_probs = np.zeros((self.num_states, self.observation_length))
        pointers = np.zeros((self.num_states, self.observation_length), dtype='int16')
        for s in range(self.num_states):
            max_probs[s, 0] = math.log(self.initial_state_probabilities[s]) + math.log(self.observation_probabilities[s, 0])
        for i in range(1, self.observation_length):
            for t in range(self.num_states):
                max_state = 0
                max_val = -np.inf
                for s in range(self.num_states):
                    temp = max_probs[s, i-1] + math.log(self.transition_probabilities[s, t]) + math.log(self.observation_probabilities[t, i])
                    if temp > max_val:
                        max_state = s
                        max_val = temp
                max_probs[t, i] = max_val
                pointers[t, i] = max_state
        max_state = 0
        max_val = -np.inf
        for t in range(self.num_states):
            if max_probs[t, self.observation_length - 1] > max_val:
                max_state = t
                max_val = max_probs[t, self.observation_length - 1]
        self.viterbi_log_probability = max_val

        #  Traceback
        for i in reversed(range(self.observation_length)):
            self.viterbi_path[i] = max_state
            max_state = pointers[max_state, i]
    
    def sum_logs(self, p, q):
        if p>9999 and q>99999:
            ans = math.log(math.exp(p) + math.exp(q))
        else:
            if p > q:
                ans =  p + math.log(1 + math.exp(q - p))
            else:
                ans =  q + math.log(1 + math.exp(p - q))
        return ans
    
    def forward(self):
        self.forward_probabilities = np.zeros((self.num_states, self.observation_length))
        for s in range(self.num_states):
            self.forward_probabilities[s, 0] = math.log(self.initial_state_probabilities[s]) + math.log(self.observation_probabilities[s, 0])
        for i in range(1, self.observation_length):
            for t in range(self.num_states):
                temp = 0
                for s in range(self.num_states):
                    if s == 0:
                        temp = math.log(self.transition_probabilities[s, t]) + self.forward_probabilities[s, i-1]
                    else:
                        temp = self.sum_logs(temp, math.log(self.transition_probabilities[s, t]) + self.forward_probabilities[s, i-1])
                self.forward_probabilities[t, i] = temp + math.log(self.observation_probabilities[t, i])
        temp = 0
        for t in range(self.num_states):
            if t == 0:
                temp = self.forward_probabilities[t, self.observation_length -1]
            else:
                temp = self.sum_logs(temp, self.forward_probabilities[t, self.observation_length -1])
        self.forward_ll = temp
        
    def backward(self):
        self.backward_probabilities = np.zeros((self.num_states, self.observation_length))
        for s in range(self.num_states):
            self.backward_probabilities[s, self.observation_length - 1] = 0 #math.log(self.observation_probabilities[s, self.observation_length - 1])
        for i in reversed(range(0, self.observation_length - 1)):
            for s in range(self.num_states):
                temp = 0
                for t in range(self.num_states):
                    if t == 0:
                        temp = self.backward_probabilities[t, i+1] + math.log(self.transition_probabilities[s, t]) + math.log(self.observation_probabilities[t, i+1])
                    else:
                        temp = self.sum_logs(temp, self.backward_probabilities[t, i+1] + math.log(self.transition_probabilities[s, t]) + math.log(self.observation_probabilities[t, i+1]))
                self.backward_probabilities[s, i] = temp
        temp = 0
        for t in range(self.num_states):
            if t == 0:
                temp = math.log(self.initial_state_probabilities[t]) + self.backward_probabilities[t, 0] + math.log(self.observation_probabilities[t,0])
            else:
                temp = self.sum_logs(temp, math.log(self.initial_state_probabilities[t]) + self.backward_probabilities[t, 0] + math.log(self.observation_probabilities[t,0]))
        self.backward_ll = temp
    
    def calculate_state_probabilities(self):
        for s in range(self.num_states):
            c = self.forward_probabilities[s] + self.backward_probabilities[s]
            self.state_probabilities.append([math.exp(x - self.forward_ll) for x in c])
    
    def calculate_probabilities(self):
        self.viterbi(); self.forward(); self.backward(); self.calculate_state_probabilities()
    
def cons_mutation_probs(params, alignment_list, alignment_names, num_symbols, sequence_name_dict, master_species_index):    
    num_states = 3
    align_list =  alignment_list
    len_align_list = len(alignment_list[0])
    num_sequences = len(alignment_list)
    observation_probs =  np.zeros((num_states, len_align_list))
    for i in range(len_align_list):
        for a_name in alignment_names:
            j = sequence_name_dict[a_name]
            if j == master_species_index:
                master_species_symbol = alignment_list[j][i]
        for s in range(num_states):
            ans = 1
            for a_name in alignment_names:
                j = sequence_name_dict[a_name]
                if j == master_species_index:
                    continue
                else:
                    aligned_symbol = alignment_list[j][i]
                    if aligned_symbol == master_species_symbol:
                        ans = ans * (params[s])
                    else:
                        ans = ans * (1-params[s])
            observation_probs[s, i] = ans
    return observation_probs

def mutation_probs(rates, alignment_list, alignment_names, master_tree, num_symbols):
    num_states = len(rates)
    align_list =  alignment_list
    len_align_list = len(alignment_list[0])
    num_sequences = len(alignment_list)
    observation_probs =  np.zeros((num_states, len_align_list))
    for i in range(len_align_list):
        temp = []
        temp.append([x for x in alignment_names])
        temp.append([x[i] for x in alignment_list])
        for j in range(num_states):
            observation_probs[j, i] = felsenstein_probability (temp, num_symbols, master_tree, rates[j]) 
    return observation_probs

def felsenstein_probability (state_list, num_symbols, master_tree, length_scalar):
    if num_symbols == 4:
        alphabet = ['A','C','G','T']
    else:
        alphabet = ['A','C','G','T','-']
    initial_states = {}
    prior_probabilities = [1/num_symbols] * num_symbols
    for i in range(len(state_list[0])):
        initial_states[state_list[0][i]] = alphabet.index(state_list[1][i])
    nodes_under_consideration = []
    info_dict = {}
    num_nodes = 0
    for node in master_tree.traverse():
        num_nodes+=1
        if node.is_leaf():
            nodes_under_consideration.append(node)
            temp_probs = []
            for s in range(num_symbols):
                if initial_states[node.name] == s:
                    temp_probs.append(1)
                else:
                    temp_probs.append(0)
            info_dict[node] = temp_probs
    while(len(nodes_under_consideration) < num_nodes):
        for n in nodes_under_consideration:
            if n.up in info_dict:
                continue
            sibling_group = [n]
            for p in n.get_sisters():
                sibling_group.append(p)
            num_not_in_dict = 0
            for x in sibling_group:
                if not(x in info_dict):
                    num_not_in_dict +=1
            if num_not_in_dict == 0:
                new_probs = []
                for s in range(num_symbols):
                    temp_prob = 1
                    for x in sibling_group:
                        branch_length = x.dist
                        probs = info_dict[x]
                        temp_prob_2 = 0
                        for t in range(num_symbols):
                            jc_prob = math.exp(-1.0*num_symbols*branch_length*length_scalar/(num_symbols -1))
                            if s == t:
                                transition_probability = 1.0/num_symbols + (num_symbols-1)/num_symbols * jc_prob
                            else:
                                transition_probability = 1.0/num_symbols - 1.0/num_symbols * jc_prob
                            temp_prob_2 += transition_probability * probs[t]
                        temp_prob = temp_prob * temp_prob_2
                    new_probs.append(temp_prob)
                info_dict[n.up] = new_probs
        nodes_under_consideration = list(info_dict.keys()) 
    for node in master_tree.traverse():
        if node.is_root():
            ans = 0
            probs = info_dict[node]
            for s in range(num_symbols):
                ans = ans + prior_probabilities[s] * probs[s] 
    return ans

def fit_phylo_hmm(tree, num_symbols, num_states, params, group_ids, align_dict, num_subsets, subset_num, offset, min_length):
    initial_state_probabilities = [1.0/num_states]*num_states
    total_probability = 0
    a = params[0]
    b = (1-params[0])
    c = 1 - (params[1])
    d = params[1]
    transition_probabilities = np.array([[a,b],[c,d]])
    ids = util.chunk_list(group_ids, num_subsets, subset_num)
    for group_id in ids:
        alignment = align_dict[group_id]
        align_list =  alignment.modified_sequence_list
        align_names = alignment.sequence_names
        len_align_list = len(align_list[0])
        non_cds = [x[offset:len_align_list - offset] for x in align_list]
        if len(non_cds[0]) < min_length:
            continue
        #observation_probabilities = mutation_probs(params[2:4], non_cds, align_names, tree, num_symbols)
        observation_probabilities = mutation_probs([params[2],params[3],params[3]], non_cds, align_names, tree, num_symbols)
        trial_hmm = HMM(initial_state_probabilities, transition_probabilities, observation_probabilities)
        #trial_hmm.viterbi()
        #total_probability += trial_hmm.viterbi_log_probability * -1
        trial_hmm.forward()
        total_probability += trial_hmm.forward_ll * -1
    return total_probability

def fit_cons_hmm(num_symbols, num_states, params, group_ids, align_dict, num_subsets, subset_num, offset, min_length, sequence_name_dict, master_species_index):
    initial_state_probabilities = [1.0/num_states]*num_states
    total_probability = 0
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
    ids = util.chunk_list(group_ids, num_subsets, subset_num)
    for group_id in ids:
        alignment = align_dict[group_id]
        align_list =  alignment.modified_sequence_list
        align_names = alignment.sequence_names
        len_align_list = len(align_list[0])
        non_cds = [x[offset:len_align_list - offset] for x in align_list]
        if len(non_cds[0]) < min_length:
            continue
        observation_probabilities = cons_mutation_probs(params[6:], non_cds, align_names, num_symbols, sequence_name_dict, master_species_index)
        trial_hmm = HMM(initial_state_probabilities, transition_probabilities, observation_probabilities)
        trial_hmm.forward()
        total_probability += trial_hmm.forward_ll * -1
    return total_probability