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
from . import HMM as hmm
from . import Sequence_Analysis_Routines as sar


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

def phylo_mutation_probs(rates, alignment_list, alignment_names, master_tree, num_symbols):
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
        #observation_probabilities = phylo_mutation_probs(params[2:4], non_cds, align_names, tree, num_symbols)
        observation_probabilities = phylo_mutation_probs([params[2],params[3],params[3]], non_cds, align_names, tree, num_symbols)
        trial_hmm = HMM(initial_state_probabilities, transition_probabilities, observation_probabilities)
        #trial_hmm.viterbi()
        #total_probability += trial_hmm.viterbi_log_probability * -1
        trial_hmm.forward()
        total_probability += trial_hmm.forward_ll * -1
    return total_probability