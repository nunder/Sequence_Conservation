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
import copy

###    File routines ###

def wslname(windows_loc):
    outstr = '/mnt/'+windows_loc[0].lower()+windows_loc[2:]
    return outstr

def list_dirs(dir):
    r = []
    s = []
    for root, dirs, files in os.walk(dir):
        for name in dirs:
            if name == '.ipynb_checkpoints':
                pass
            else:
                s.append(name)
    return s

def list_files(dir):
    r = []
    s = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name == '.ipynb_checkpoints':
                pass
            else:
                s.append(name)
    return s

def delete_if_exists(filename): 
    if os.path.exists(filename):
        os.remove(filename)

#def chunk_list(id_list, num_subsets, subset_num):
#    len_ids = len(id_list)
#    subset_size = int(len_ids / num_subsets)
#    if subset_num == num_subsets:
#        ids = id_list[(subset_num - 1) * subset_size:]
#    else:
#        ids = id_list[(subset_num - 1) * subset_size: (subset_num ) * subset_size]
#    return ids

def chunk_list(id_list, num_subsets, subset_num):
    len_ids = len(id_list)
    subset_size = int(len_ids / num_subsets)
    remainder = len_ids % num_subsets
    if subset_num <= remainder:
        ids = id_list[(subset_num - 1) * (subset_size + 1): subset_num * (subset_size + 1)]
    else:
        ids = id_list[len_ids - (num_subsets - (subset_num -1))*subset_size : len_ids - (num_subsets - subset_num)*subset_size]
    return ids

def repeat_fn(num_cores, inputs):
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def g(id_list, num_subsets, subset_number):
                id_subset = chunk_list(id_list, num_subsets, subset_number)
                temp = []
                for i in id_subset:
                    args = (i,)
                    temp.append(func(*args, **kwargs))
                return temp
            core_numbers = list(range(1, num_cores+1))
            parallel_output = Parallel(n_jobs=-1)(delayed(g)(inputs, num_cores, core_number) for core_number in core_numbers)
            return [item for sublist in parallel_output for item in sublist]
        return wrapper
    return decorate   

def parallelize(func, group_ids, other_args, num_cores):
    def g(id_list, num_subsets, subset_number):
            id_subset = chunk_list(id_list, num_subsets, subset_number)
            arg_list = copy.deepcopy(other_args)
            arg_list.insert(0, 0)
            temp = []
            for i in id_subset:
                arg_list[0] = i
                temp.append(func(*arg_list))
            return temp
    core_numbers = list(range(1, num_cores+1))
    parallel_output = Parallel(n_jobs=-1)(delayed(g)(group_ids, num_cores, core_number) for core_number in core_numbers)
    return [item for sublist in parallel_output for item in sublist]

def concatenate_fasta(directory, file_list, output_file):
    sequence_dict = {}
    for filename in file_list:
        f = directory + '/' + filename
        sequence_count = 0
        with open(f,'r') as ofile:
                first_seq = 0
                for l in ofile:
                    m = l.strip('\n')
                    if m[0] == '>':
                        sequence_count += 1
                        if first_seq == 0:
                            sequence_name = m[1:]
                            outstr = ''
                        else:
                            sequence_dict[(filename,sequence_name)] = outstr
                            sequence_name = m[1:]
                            outstr = ''
                    else:
                        first_seq = 1
                        outstr += m
                sequence_dict[(filename,sequence_name)] = outstr
                sequence_name = m[1:]
    name_list = list(set([i[1] for i in list(sequence_dict.keys())]))
    with open(output_file,'w') as outfile:
        for name in name_list:
            outfile.write(">" + name + "\n")      
            outstring = []
            for filename in file_list:
                outstring.append(sequence_dict[(filename, name)])
            outfile.write(''.join(outstring) + "\n")
    
def read_fasta_to_array(filename, species_order = []):
    with open(filename,'r') as ofile: 
            sequence_names = []
            sequence_list = []
            first_seq = 0
            for l in ofile:
                m = l.strip('\n')
                if m[0] == '>':
                    if first_seq > 0:
                        sequence_list.append(outstr)
                    outstr = ''
                    sequence_names.append(m[1:])
                else:
                    first_seq = 1
                    outstr += m
            sequence_list.append(outstr)
            if len(species_order) == 0:
                pass
            else:
                ordered_sequence_list = []
                ordered_sequence_names = []
                for seq_name in species_order:
                    if seq_name in sequence_names:
                        pos = sequence_names.index(seq_name)
                        ordered_sequence_list.append(sequence_list[pos])
                        ordered_sequence_names.append(sequence_names[pos])
    if len(species_order) == 0:
        return [sequence_names, sequence_list]
    else:
        return [ordered_sequence_names, ordered_sequence_list]
  
 
class Translator:
    def __init__(self):
        self.codon_dict = {}
    with open('D:/Project_Data/Project_3/Datasets/Reference_Tables/Standard_Code.txt') as f:
        for l in f:
            pass
            #self.codon_dict[str(l[1:4])] = l[5]
        
    def translate_sequence(self, input_seq, strand, rf, separate_start_symbol = False, separate_stop_symbol = False):
        output_seq = ''
        if strand == 1:
            seq = input_seq[rf:]
        else:
            seq = align.reverse_complement(input_seq)[rf:]
        for i in range(0,len(seq)-2,3):
            if separate_start_symbol == True and seq[i:(i+3)] in ['ATG','GTG','TTG']:
                output_seq += 'Z'
            elif separate_stop_symbol == True and seq[i:(i+3)] in ['TAG','TGA','TAA']:
                output_seq += 'J'
            else:
                if seq[i:(i+3)] in self.codon_dict:
                    output_seq += self.codon_dict[seq[i:(i+3)]]
                else:
                    output_seq += 'X'
        return output_seq
