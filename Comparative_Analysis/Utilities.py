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
    
def read_fasta_to_arrays(filename):
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
    num_sequences = len(sequence_names) 
    return [sequence_names, sequence_list]