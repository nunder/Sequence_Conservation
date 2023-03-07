import os
import pandas as pd
import subprocess
import seaborn as sns
import shutil
from tqdm import tqdm
import numpy as np
import seaborn as sns
from Bio import Entrez, SeqIO, AlignIO, pairwise2, Align, Seq, motifs
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.Align.Applications import MuscleCommandline
from pathlib import Path
from joblib import Parallel, delayed
import random
from statistics import mean, stdev
from scipy.stats import binom
import math
from scipy import linalg
import scipy.stats as ss
from . import Utilities as util
import copy
import pickle

genome_datasets_file = 'D:/Project_Data/Project_8/Datasets/Actinobacteria_Ref_Rep_Lev_Complete/GCF_000195955.2_ASM19595v2_genomic.gbff'
mutation_counts_file = 'F:/Project_Data/Project_9/alt_mutation_counts.pkl'

class ORF_Finder:
    def __init__(self, sequence):
        self.full_sequence = sequence
        
    
    def max_orf(self, seq_start, seq_stop, output_orfs = 'None', min_orf_length = 0):
        max_len = 0
        orfs_found = []
        start_pos = -999
        end_pos = -999
        for frame in ['Forward', 'Reverse']:
            if frame == 'Forward':
                temp = (self.full_sequence[seq_start: seq_stop])
            else:
                temp = util.reverse_complement(self.full_sequence[seq_start: seq_stop])
            seq_len = len(temp)
            for i in range(seq_len - 2):
                orf_length = 0
                test_codon = temp[i: i+3] 
                if test_codon in ['ATG','GTG','TTG']:  #Missed out CTG as doesn't seem to be used very much at all
                    for j in range(i + 3, seq_len - 2, 3):
                        test_codon_2 = temp[j: j+3] 
                        if test_codon_2 in ['TAG','TGA','TAA']:
                            orf_length = j - i
                            break
                    if orf_length > 0:
                        if frame == 'Forward':
                            orf_start =  seq_start + i
                            orf_end = seq_start + j+3
                            orf_strand = 1
                        else:
                            orf_start =  seq_start + seq_len-(j+3)
                            orf_end = seq_start + seq_len-i
                            orf_strand = -1
                        
                        if orf_length >= min_orf_length:
                            orfs_found.append((orf_start, orf_end, orf_strand, orf_length))

                    if orf_length > max_len and orf_length >= min_orf_length:                                           
                        max_len = orf_length
                        start_pos = orf_start
                        end_pos = orf_end
                        strand = orf_strand 

        if output_orfs == 'All':
            sorted_orfs = sorted(orfs_found, key=lambda x: x[3], reverse=True)
            return sorted_orfs    
        
        elif output_orfs == 'Nested':
            morf_dict = {}
            for x in orfs_found:
                if x[2] == 1:
                    if not (x[1] in morf_dict) or ((x[1] in morf_dict) and (x[3] > morf_dict[x[1]][3])):
                        morf_dict[x[1]] = x

                else:
                    if not (x[0] in morf_dict) or ((x[0] in morf_dict) and (x[3] > morf_dict[x[0]][3])):
                        morf_dict[x[0]] = x
            results = []
            for k, v in morf_dict.items():
                results.append(v)
            results.sort(key = lambda x: x[0])
            return results
        
        elif start_pos == -999:
            return(0,0,0)
        else:
            return(start_pos, end_pos, strand)   

class H37Rv_ORF_Finder:
    def __init__(self):
        genome_record = next(SeqIO.parse(genome_datasets_file, "genbank"))
        self.full_sequence = genome_record.seq
        with open(mutation_counts_file, 'rb') as f:
            mutation_counts = pickle.load(f)  
            mutation_counts.sort(key = lambda x: int(x[0]))
            self.mutation_count_list = []
            for (start, stop, counts) in mutation_counts:
                for count in counts:
                    self.mutation_count_list.append(count)
        
        
    def bin_formula(self, max_bin_counts, tot_bin_counts):
        return 1- binom.cdf(max_bin_counts-1, tot_bin_counts,1/3)

    def mutation_bin_probability(self, mutation_counts):
        bin_counts = [0,0,0]
        for i, c in enumerate(mutation_counts):
            bin_counts[i % 3] += c
        if sum(bin_counts) == 0:
            return 2
        else:
            return self.bin_formula(bin_counts[2], sum(bin_counts))  
    
    def max_orf(self, seq_start, seq_stop, p_value, output_orfs = 'None', min_orf_length = 0):
        max_len = 0
        orfs_found = []
        start_pos = -999
        end_pos = -999
        for frame in ['Forward', 'Reverse']:
            if frame == 'Forward':
                temp = (self.full_sequence[seq_start: seq_stop])
            else:
                temp = util.reverse_complement(self.full_sequence[seq_start: seq_stop])
            seq_len = len(temp)
            for i in range(seq_len - 2):
                orf_length = 0
                test_codon = temp[i: i+3] 
                if test_codon in ['ATG','GTG','TTG']:  #Missed out CTG as doesn't seem to be used very much at all
                    for j in range(i + 3, seq_len - 2, 3):
                        test_codon_2 = temp[j: j+3] 
                        if test_codon_2 in ['TAG','TGA','TAA']:
                            orf_length = j - i
                            break
                    if orf_length > 0:
                        if frame == 'Forward':
                            orf_start =  seq_start + i
                            orf_end = seq_start + j+3
                            orf_strand = 1
                        else:
                            orf_start =  seq_start + seq_len-(j+3)
                            orf_end = seq_start + seq_len-i
                            orf_strand = -1
                        if orf_strand == 1:
                            prob = self.mutation_bin_probability(self.mutation_count_list[orf_start:orf_end])
                        else:
                            prob = self.mutation_bin_probability(reversed(self.mutation_count_list[orf_start:orf_end]))
                        if prob < p_value and orf_length >= min_orf_length:
                            orfs_found.append((orf_start, orf_end, orf_strand, orf_length, prob))

                    if orf_length > max_len and prob< p_value and orf_length >= min_orf_length:                                           
                        max_len = orf_length
                        start_pos = orf_start
                        end_pos = orf_end
                        strand = orf_strand 

        if output_orfs == 'All':
            sorted_orfs = sorted(orfs_found, key=lambda x: x[3], reverse=True)
            return sorted_orfs        
        elif output_orfs == 'Nested':
            morf_dict = {}
            for x in orfs_found:
                if x[2] == 1:
                    if not (x[1] in morf_dict) or ((x[1] in morf_dict) and (x[3] > morf_dict[x[1]][3])):
                        morf_dict[x[1]] = x

                else:
                    if not (x[0] in morf_dict) or ((x[0] in morf_dict) and (x[3] > morf_dict[x[0]][3])):
                        morf_dict[x[0]] = x
            results = []
            for k, v in morf_dict.items():
                results.append(v)
            results.sort(key = lambda x: x[0])
            return results
        
        elif start_pos == -999:
            return(0,0,0)
        else:
            if strand == 1:
                prob = self.mutation_bin_probability(self.mutation_count_list[start_pos:end_pos])
            else:
                prob = self.mutation_bin_probability(reversed(self.mutation_count_list[start_pos:end_pos]))
            return(start_pos, end_pos, strand, prob)   

