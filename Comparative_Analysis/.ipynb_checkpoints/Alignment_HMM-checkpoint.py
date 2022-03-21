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
from . import Alignment as align

class Alignment_HMM:
    
    def __init__(self, num_symbols, num_states, alignment_dir, master_species_id):
        self.num_symbols = num_symbols 
        self.num_states = num_states
        alignment_ids = util.list_files(alignment_dir + '/')
        self.alignment_ids = [int(i.split('.')[0]) for i in alignment_ids]
        self.alignment_dict = {}
        for group_id in tqdm(self.alignment_ids):
            alignment = align.Alignment(alignment_dir + '/' + str(group_id) + '.fasta', master_species_id, 'NT')
            alignment.modify_sequence(1, False, False)
            self.alignment_dict[group_id] = alignment
        
    def alignment_hmm_mutation_probs(self, params, alignment_list, alignment):    
        align_list =  alignment_list
        len_align_list = len(alignment_list[0])
        num_sequences = len(alignment_list)
        observation_probs =  np.zeros((self.num_states, len_align_list))
        master_species_index = alignment.master_species_index
        for i in range(len_align_list):
            master_species_symbol = alignment_list[master_species_index][i]
            for s in range(self.num_states):
                ans = 1
                for k in range(num_sequences):
                    if k == master_species_index:
                        continue
                    else:
                        aligned_symbol = alignment_list[k][i]
                        if aligned_symbol == master_species_symbol:
                            ans = ans * (params[s])
                        else:
                            ans = ans * (1-params[s])
                observation_probs[s, i] = ans
        return observation_probs

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

    def fit_alignment_hmm(self, params, num_subsets, subset_num, offset, min_length):
        initial_state_probabilities = [1.0/self.num_states]*self.num_states
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
            observation_probabilities = self.alignment_hmm_mutation_probs(mutation_probabilities, non_cds, alignment)
            trial_hmm = hmm.HMM(initial_state_probabilities, transition_probabilities, observation_probabilities)
            trial_hmm.forward()
            total_probability += trial_hmm.forward_ll * -1
        return total_probability

