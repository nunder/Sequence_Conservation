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
from . import Alignment_HMM as alignment_hmm

class Alignment_Analysis:
    
    def __init__(self, alignment, num_states, non_cds_offset, seq_type, group_id, fitted_parameters, project_dir, Alignment_HMM_Model, seq_data):
        self.group_id = group_id
        self.alignment = alignment 
        self.alignment.modify_sequence(1,False,False)
        self.alignment.calculate_entropies(mvave_len = 10)
        
        utr_upstream_dict = {}
        utrs = pd.read_csv(project_dir + '/Datasets/Data_From_Publications/strict_3UTRs.csv', header=0)
        for i, r in utrs.iterrows():
            utr_upstream_dict[r['upstream']] = [r['utr'], r['start']-1, r['stop']-1, r['strand'], r['downstream']]
        
        initial_state_probabilities = [1.0/num_states]*num_states
        transition_probabilities, mutation_probabilities = Alignment_HMM_Model.alignment_hmm_model_inputs(fitted_parameters)
        observation_probabilities = Alignment_HMM_Model.alignment_hmm_mutation_probs(mutation_probabilities, self.alignment.modified_sequence_list, alignment)
        self.hmm_model = hmm.HMM(initial_state_probabilities, transition_probabilities, observation_probabilities)
        self.hmm_model.calculate_probabilities()
        self.buffer_end = non_cds_offset - 1
        self.target_end = self.alignment.modified_sequence_length - non_cds_offset
        if seq_type == 'Downstream':
            self.start = seq_data.master_species_info(self.group_id, 'non_cds_offset_start')
            self.end = seq_data.master_species_info(self.group_id, 'non_cds_offset_end')
            self.locus_tag = seq_data.master_species_info(self.group_id, 'locus_tag')
            self.locus_strand = seq_data.master_species_info(self.group_id, 'strand')
            if self.locus_strand == 1:
                self.locus_tag_2 = seq_data.master_species_info(self.group_id, 'next_locus_tag')
                self.locus_strand_2 = seq_data.master_species_info(self.group_id, 'next_strand')
            else:
                self.locus_tag_2 = seq_data.master_species_info(self.group_id, 'previous_locus_tag')
                self.locus_strand_2 = seq_data.master_species_info(self.group_id, 'prev_strand')
        else:
            self.start = seq_data.master_species_info(self.group_id, 'upstream_non_cds_offset_start')
            self.end = seq_data.master_species_info(self.group_id, 'upstream_non_cds_offset_end')
            self.locus_tag_2 = seq_data.master_species_info(self.group_id, 'locus_tag')
            self.locus_strand_2 = seq_data.master_species_info(self.group_id, 'strand')
            if self.locus_strand_2 == 1:
                self.locus_tag = seq_data.master_species_info(self.group_id, 'previous_locus_tag')
                self.locus_strand = seq_data.master_species_info(self.group_id, 'prev_strand')
            else:
                self.locus_tag = seq_data.master_species_info(self.group_id, 'next_locus_tag')
                self.locus_strand = seq_data.master_species_info(self.group_id, 'next_strand')

        self.utr_start = 0
        self.utr_end = 0
        if self.locus_tag in utr_upstream_dict:
            utr_data = utr_upstream_dict[self.locus_tag]
            if self.locus_strand == 1:
                self.utr_start = utr_data[1] - self.start
                self.utr_end = utr_data[2] - self.start
            else:
                self.utr_starts = self.end - utr_data[2]
                self.utr_end = self.end - utr_data[1]        
        
 