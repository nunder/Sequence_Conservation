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

def align_and_build(id_list, num_subsets, subset_num, source_data, length_field, seq_field, out_loc, min_species): 
    muscle_exe = 'C:/Users/nicho/Muscle/muscle3.8.31_i86win32.exe'
    ids = util.chunk_list(id_list, num_subsets, subset_num)
    for j in ids:
        temp_df = source_data[source_data['group_id'] == j]
        num_non_zero = 0
        num_zero = 0
        for i, r in temp_df.iterrows():
            if (r[length_field] > 0):
                num_non_zero += 1
            else:
                num_zero += 1
        if ((num_non_zero >= min_species)):
            ofile = open(out_loc+'temp'+str(j)+'.fasta', "w")
            for i, r in temp_df.iterrows():
                if (r[length_field] > 0):
                    ofile.write(">" + r['species'] + "\n" + r[seq_field] + "\n")
                else:
                    ofile.write(">" + r['species'] + "\n" + "\n")
            ofile.close() 
            cline = MuscleCommandline(muscle_exe, input=out_loc +'temp'+str(j)+'.fasta', out=out_loc+str(j)+'.fasta')
            try:
                stdout, stderr = cline()
            except Exception as e:
                pass
            util.delete_if_exists(out_loc +'temp'+str(j)+'.fasta')
            

def relative_entropy(sequence_list, alphabet_name = 'NT', *args, **kwargs):
    alphabet_list = kwargs.get('alphabet_list',[])
    background_probabilities =kwargs.get('background_probabilities',[]) 
    element_wise = kwargs.get('element_wise',True)
    insertions_as_background = kwargs.get('insertions_as_background',True)
    exclude_insertions = kwargs.get('exclude_insertions',False)
    insertion_character = kwargs.get('insertion_character','-')
    if alphabet_name == 'AA':
        alphabet_list = ['D', 'I', 'A', 'S', 'P', 'Y', 'V', 'Q', 'T', '*', 'H', 'G', 'R', 'F', 'W', 'N', 'C', 'K', 'L', 'M', 'E']
    elif alphabet_name == 'NT':
        alphabet_list =['A','C','G','T']
    if insertions_as_background == False:
        alphabet_list.append(insertion_character)
    if len(background_probabilities) > 0:
        background_probs = background_probabilities
    else:
        background_probs = [1/len(alphabet_list)]*len(alphabet_list)
    num_symbols = len(alphabet_list)
    num_sequences = len(sequence_list)
    sequence_length = len(sequence_list[0])
    relent_list = []
    cumulative_relent = 0
    symbol_entropies = [[] for k in range(num_symbols)]
    for i in range(sequence_length):
        relent = 0
        vals = [v[i] for v in sequence_list]
        if (exclude_insertions == False) or (not(insertion_character in vals)):
            for j in range(num_symbols):
                if insertions_as_background == True:
                    ct = vals.count(alphabet_list[j]) + vals.count(insertion_character) * background_probs[j]
                else:
                    ct = vals.count(alphabet_list[j]) 
                if ct == 0:
                    temp = 0
                else:
                    temp = (ct/num_sequences) * math.log((ct/num_sequences)/background_probs[j],2)
                symbol_entropies[j].append(temp)
                relent = relent + temp
            cumulative_relent = cumulative_relent + relent  
            relent_list.append(relent)
    if element_wise == True:
        return relent_list, symbol_entropies
    else:
        return cumulative_relent, symbol_entropies

    
class Alignment:
    def __init__(self, fileloc, master_species, alphabet_name, insert_symbol = '-'): 
        temp = util.read_fasta_to_arrays(fileloc)
        self.alphabet_name = alphabet_name
        self.non_insert_symbols = []
        if self.alphabet_name == 'NT':
            self.non_insert_symbols = ['A','C','G','T']
        self.insert_symbol = insert_symbol
        self.sequence_names = temp[0]
        self.sequence_list = temp[1]
        self.modified_sequence_list = temp[1]
        self.num_sequences = len(self.sequence_names)
        self.sequence_length = len(self.sequence_list[0])
        self.modified_sequence_length = len(self.sequence_list[0])
        self.master_species = master_species
        self.master_species_index = self.sequence_names.index(self.master_species)  
        self.relative_entropy = []
        self.symbol_entropies = []
        self.mvave_relative_entropy = [] 
        self.master_species_modified_sequence_insertions = []
        self.master_species_modified_sequence = self.modified_sequence_list[self.master_species_index]
        self.replaced_indels = []
       
    def modify_sequence(self, consensus, delete_insert_sites = False, randomize_insert_sites = False):
        self.modified_sequence_list = []
        for j in range(self.num_sequences):
            self.modified_sequence_list.append([])
        for i in range(self.sequence_length):
            temp = [x[i] for x in self.sequence_list]
            if delete_insert_sites == True:
                if temp.count(self.insert_symbol) > 0:
                    continue
            if ((temp[self.master_species_index] == self.insert_symbol) and (temp.count(self.insert_symbol) >= consensus)):
                continue
            if randomize_insert_sites == True:
                num_replacements = 0
                for i in range(len(temp)):
                    if not (temp[i] in self.non_insert_symbols):
                        temp[i] = self.non_insert_symbols[np.where(np.random.default_rng().multinomial(1, np.array([0.25, 0.25, 0.25, 0.25]), size=None) == 1)[0][0]]
                        num_replacements += 1
                self.replaced_indels.append(num_replacements)
            for j in range(self.num_sequences):
                self.modified_sequence_list[j].append(temp[j])
        for i in range(self.num_sequences):
            self.modified_sequence_list[i] = ''.join(self.modified_sequence_list[i])
        self.modified_sequence_length = len(self.modified_sequence_list[0])
        self.master_species_modified_sequence = self.modified_sequence_list[self.master_species_index]
        self.master_species_modified_sequence_insertions = []
        other_sequences = []
        for i in range(self.num_sequences):
            if not ((i == self.master_species_index)):
                other_sequences.append(self.modified_sequence_list[i])
        for i in range(self.modified_sequence_length):
            if not (self.master_species_modified_sequence[i] == self.insert_symbol):
                sli = [x[i] for x in other_sequences]
                if sli.count(self.insert_symbol) >=1:   # == self.num_sequences - 1:
                    self.master_species_modified_sequence_insertions.append([i, sli.count(self.insert_symbol)])
        
    def calculate_entropies(self, mvave_len = 1, modified=True):
        if modified == True:
            self.relative_entropy, self.symbol_entropies = relative_entropy(self.modified_sequence_list,alphabet_name = self.alphabet_name)
        else:
            self.relative_entropy, self.symbol_entropies = relative_entropy(self.sequence_list,alphabet_name = self.alphabet_name)
        self.mvave_relative_entropy = []
        for k in range(len(self.relative_entropy)):
            mv_temp = max(int(mvave_len/2),1)
            if ((k + mv_temp <= len(self.relative_entropy)) and (k-mv_temp >= 0)):
                self.mvave_relative_entropy.append(mean(self.relative_entropy[k-mv_temp:k+mv_temp]))
            else:
                self.mvave_relative_entropy.append(-np.inf)
     
    def alignment_position(self, pos):
        if pos < 0:
            temp_range = reversed(range(self.sequence_length))
            pos = pos * - 1
        else:
            temp_range = range(self.sequence_length)
        num_chars = 0
        ans = -1
        for i in temp_range:
            temp =  self.sequence_list[self.master_species_index][i]
            if not(temp == self.insert_symbol):
                num_chars += 1
            if num_chars == pos:
                ans = i 
                break
        return ans   

    def find_pattern(self, search_str_list, start_pos, end_pos, min_entropy, max_mismatches, in_frame = False, frame_start = 0, method = 'count'):
        match_starts = []
        if method == 'entropy':
            self.calculate_entropies()
            for search_str in search_str_list:
                search_len = len(search_str)
                search_positions = []
                for i in range(search_len):
                    if search_str[i] == 'N':
                        search_positions.append(-1)
                    else:
                        search_positions.append(self.non_insert_symbols.index(search_str[i]))
                i = start_pos
                while i <= end_pos - search_len:
                    if (in_frame == True) and not((i - frame_start)%3 == 0):
                        i += 1
                        continue
                    num_mismatches = 0
                    for j in range(search_len):
                        if search_positions[j] == -1:
                            pass
                        elif self.symbol_entropies[search_positions[j]][i+j] < min_entropy:
                            num_mismatches += 1
                        else:
                            pass
                    if num_mismatches > max_mismatches:
                        pass
                    else:
                        match_starts.append(i)
                    i += 1
        else:
                i = start_pos
                search_len = len(search_str_list[0])
                while i <= end_pos - search_len:
                    if (in_frame == True) and not((i - frame_start)%3 == 0):
                        i += 1
                        continue
                    num_mismatches = 0
                    for j in range(self.num_sequences):
                        matched = 0
                        for search_str in search_str_list:
                            test_seq = self.modified_sequence_list[j][i:i+search_len]
                            if test_seq == search_str:
                                matched = 1
                        if matched == 0:
                            num_mismatches += 1
                    if num_mismatches > max_mismatches:
                        pass
                    else:
                        match_starts.append(i)
                    i += 1
        return match_starts   
    