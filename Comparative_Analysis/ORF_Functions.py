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

cryptic_output_path = "D:/Project_Data/CRYPTIC_DATA/Cryptic_Data_Analysis"
genome_datasets_dir = 'D:/Project_Data/Project_6/Datasets/NCBI_Datasets'

class ORF_Finder:
    def __init__(self):
        genome_record = next(SeqIO.parse(genome_datasets_dir + '/GCF_000195955.2/genomic.gbff', "genbank"))
        self.full_sequence = genome_record.seq
        variant_count_df = pd.read_csv(cryptic_output_path + '/filtered_variant_summary_df.csv')
        variant_count_df = variant_count_df[variant_count_df['MUTATION_PCT'] < 0.1]
        temp = variant_count_df.groupby(['GENOME_INDEX'])[['MYKROBE_LINEAGE_NAME_2']].count().reset_index()
        temp_dict = dict(zip(temp.GENOME_INDEX, temp.MYKROBE_LINEAGE_NAME_2))
        self.mutation_counts_dict = {}
        for i in range(len(self.full_sequence)):
            if (i+1) in temp_dict:
                self.mutation_counts_dict[i] = temp_dict[(i+1)]
            else:
                self.mutation_counts_dict[i] = 0
        
        
    def bin_formula(self, max_bin_counts, tot_bin_counts, in_frame = False):
        return 1- binom.cdf(max_bin_counts-1, tot_bin_counts,1/3)

    def mutation_bin_probability(self, start, end, strand):
        mutations = []
        for i in range(start,end):
            for j in range(self.mutation_counts_dict[i]):
                mutations.append(i)
        bin_counts = [0,0,0]
        for m in mutations:
            if strand == 1:
                bin_counts[(m-(start))%3] +=1
            else:
                bin_counts[((end-1)-m)%3] +=1
        if sum(bin_counts) == 0:
            return (2)
        else:
            return (self.bin_formula(bin_counts[2], sum(bin_counts)))  
    
    def max_orf(self, seq_start, seq_stop, p_value, output_all_orfs = False, min_orf_length = 0):
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
                        prob = self.mutation_bin_probability(orf_start, orf_end, orf_strand)
                        if prob < p_value and orf_length >= min_orf_length:
                            orfs_found.append((orf_start, orf_end, orf_strand, orf_length, prob))

                    if orf_length > max_len and prob< p_value and orf_length >= min_orf_length:                                           
                        max_len = orf_length
                        start_pos = orf_start
                        end_pos = orf_end
                        strand = orf_strand 

        if output_all_orfs == True:
            sorted_orfs = sorted(orfs_found, key=lambda x: x[3], reverse=True)
            return sorted_orfs                
        elif start_pos == -999:
            return(0,0,0)
        else:
            return(start_pos, end_pos, strand, max_len, self.mutation_bin_probability(start_pos, end_pos, strand))   