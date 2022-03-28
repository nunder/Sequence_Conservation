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
import logomaker as lm
from . import Utilities as util
from . import HMM as hmm
from . import Sequence_Analysis_Routines as sar
from . import Alignment_HMM as alignment_hmm

class Alignment_Analysis:
    
    def __init__(self, analysis_type, alignment, num_states, non_cds_offset, group_id, fitted_parameters, project_dir, Alignment_HMM_Model, Master_Alignment_HMM_Model, pairwise_fitted_parameters, master_fitted_parameters, seq_data):
        self.analysis_type = analysis_type
        self.group_id = group_id
        self.alignment = alignment 
        self.alignment.modify_sequence(1,False,False)
        self.alignment.calculate_entropies(mvave_len = 10)
        
        self.species_name_dict = {}
        for i, r in seq_data.species_info().iterrows():
            self.species_name_dict[r['species']] = 'M.'+ (r['name'].split()[1]) 
        
        self.insertion_locations = {}
        for i, seq in enumerate(self.alignment.modified_sequence_list):
            if i == self.alignment.master_species_index:
                continue
            temp = []
            for j, symbol in enumerate(seq):
                if symbol =='-':
                    temp.append(j)
            self.insertion_locations[self.alignment.sequence_names[i]] = temp
        
        utr_upstream_dict = {}
        utrs = pd.read_csv(project_dir + '/Datasets/Data_From_Publications/strict_3UTRs.csv', header=0)
        for i, r in utrs.iterrows():
            utr_upstream_dict[r['upstream']] = [r['utr'], r['start']-1, r['stop']-1, r['strand'], r['downstream']]
        
        initial_state_probabilities = [1.0/num_states]*num_states
        transition_probabilities, mutation_probabilities = Alignment_HMM_Model.alignment_hmm_model_inputs(fitted_parameters)
        observation_probabilities = Alignment_HMM_Model.calculate_observation_probs(mutation_probabilities, self.alignment.modified_sequence_list, alignment)
        self.hmm_model = hmm.HMM(initial_state_probabilities, transition_probabilities, observation_probabilities)
        self.hmm_model.calculate_probabilities()
       
        self.species_names = []
        self.hmm_model_list = []
        pairwise_state_probabilities = []
        for params in pairwise_fitted_parameters:
            transition_probabilities, mutation_probabilities = Alignment_HMM_Model.alignment_hmm_model_inputs(params[1])
            observation_probabilities = Alignment_HMM_Model.calculate_observation_probs(mutation_probabilities, self.alignment.modified_sequence_list, alignment, 
                                                                                        all_species=False, comparison_species = params[0])
            self.hmm_model_list.append(hmm.HMM(initial_state_probabilities, transition_probabilities, observation_probabilities))
            self.hmm_model_list[-1].calculate_probabilities()
            self.species_names.append(params[0])
            pairwise_state_probabilities.append(self.hmm_model_list[-1].state_probabilities)
        
        num_master_states = 2
        master_initial_state_probabilities = [1.0/num_master_states]*num_master_states
        transition_probabilities, mutation_probabilities = Master_Alignment_HMM_Model.alignment_hmm_model_inputs(master_fitted_parameters)
        observation_probabilities = Master_Alignment_HMM_Model.calculate_observation_probs(mutation_probabilities, pairwise_state_probabilities)
        self.master_hmm_model = hmm.HMM(master_initial_state_probabilities, transition_probabilities, observation_probabilities)
        self.master_hmm_model.calculate_probabilities()
        
            
        self.buffer_end = non_cds_offset - 1
        self.target_end = self.alignment.modified_sequence_length - non_cds_offset
        if analysis_type == 'Downstream':
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
        
    def display_analysis(self):
        
        plot_length = self.end - self.start
        counts_df = lm.alignment_to_matrix(sequences = self.alignment.modified_sequence_list, to_type = 'counts', characters_to_ignore = '-', pseudocount=0)
        background_probs = [0.25, 0.25, 0.25, 0.25]
        for i, r in counts_df.iterrows():
            temp_relent = []
            num_gaps = self.alignment.num_sequences
            for k in range(4):
                num_gaps = num_gaps - r.iloc[k]
            for k in range(4):
                ct = r.iloc[k] + num_gaps*background_probs[k]
                if ct == 0:
                    temp_relent.append(0)
                else:
                    temp_relent.append((ct /self.alignment.num_sequences) * math.log((ct /self.alignment.num_sequences)/background_probs[k],2))
            for k in range(4):
                r.iloc[k] = temp_relent[k]

        y = -1        
        text_offset = 0.06 * plot_length
        
        seqlogo = lm.Logo(counts_df, figsize = [30,6])
        seqlogo.style_spines(visible=False)
        seqlogo.style_spines(spines=['left'], visible=True, bounds=[0, 2])
        seqlogo.ax.set_xticks([])
        seqlogo.ax.set_yticks([0,2])
        seqlogo.ax.set_ylim([-7, 2])
        seqlogo.ax.axhline(y, color = 'k', linewidth = 1)
        if self.analysis_type == 'Upstream':
            seqlogo.ax.set_title(self.analysis_type + ' of ' + self.locus_tag_2)
        else:
            seqlogo.ax.set_title(self.analysis_type + ' of ' + self.locus_tag)
        # Start and end regions
        seqlogo.ax.plot([-0.5, self.buffer_end+0.5], [y,y], color='skyblue', linewidth=10, solid_capstyle='butt')
        seqlogo.ax.plot([self.target_end-0.5, self.alignment.modified_sequence_length +0.5], [y,y], color='skyblue', linewidth=10, solid_capstyle='butt')
        
        #Old method Insertions
        #for i in self.alignment.master_species_modified_sequence_insertions:
        #    seqlogo.ax.plot([i[0], i[0]+1], [y-2,y-2], color='red', linewidth=3*i[1], solid_capstyle='butt')
        
        #Pribnow
        for i in self.alignment.find_pattern(['TANNNT'] , 0 , self.alignment.modified_sequence_length,1.3,0, method = 'entropy'):
            seqlogo.ax.plot([i, i+5], [y,y], color='orange', linewidth=5, solid_capstyle='butt')
        for i in self.alignment.find_pattern(['ANNNTA'] , 0 , self.alignment.modified_sequence_length,1.3,0, method = 'entropy'):
            seqlogo.ax.plot([i, i+5], [y-0.5,y-0.5], color='orange', linewidth=5, solid_capstyle='butt')
        
        
        seqlogo.ax.plot([self.target_end, self.target_end], [y,y], color='blue', linewidth=5, solid_capstyle='butt')
        seqlogo.ax.plot([self.buffer_end, self.buffer_end], [y,y], color='blue', linewidth=5, solid_capstyle='butt')
        
        tolerance = 4
        # Stop, Start codons in frame
        if self.analysis_type == 'Upstream': 
            for i in self.alignment.find_pattern(['ATG','GTG','TTG','CTG'],0,self.alignment.modified_sequence_length,1,tolerance,in_frame = True, frame_start = self.target_end, method = 'count'):
                seqlogo.ax.plot([i-0.5, i+2.5], [y,y], color='green', linewidth=5, solid_capstyle='butt')
            for i in self.alignment.find_pattern(['TAG','TGA','TAA'],0,self.alignment.modified_sequence_length,1,tolerance,in_frame = True, frame_start = self.target_end, method = 'count'):
                seqlogo.ax.plot([i-0.5, i+2.5], [y,y], color='red', linewidth=5, solid_capstyle='butt')
                
            if self.locus_strand == self.locus_strand_2:
                for i in self.alignment.find_pattern(['ATG','GTG','TTG','CTG'],0,self.alignment.modified_sequence_length,1,tolerance,in_frame = True, frame_start = self.buffer_end-2, method = 'count'):
                    seqlogo.ax.plot([i-0.5, i+2.5], [y-0.5,y-0.5], color='green', linewidth=5, solid_capstyle='butt')
                for i in self.alignment.find_pattern(['TAG','TGA','TAA'],0,self.alignment.modified_sequence_length,1,tolerance,in_frame = True, frame_start = self.buffer_end-2, method = 'count'):
                    seqlogo.ax.plot([i-0.5, i+2.5], [y-0.5,y-0.5], color='red', linewidth=5, solid_capstyle='butt')
            else:
                for i in self.alignment.find_pattern(['CAT','CAC','CAA','CAG'],0,self.alignment.modified_sequence_length,1,tolerance,in_frame = True, frame_start = self.buffer_end-2, method = 'count'):
                    seqlogo.ax.plot([i-0.5, i+2.5], [y-0.5,y-0.5], color='green', linewidth=5, solid_capstyle='butt')
                for i in self.alignment.find_pattern(['CTA','TCA','TTA'],0,self.alignment.modified_sequence_length,1,tolerance,in_frame = True, frame_start = self.buffer_end-2, method = 'count'):
                    seqlogo.ax.plot([i-0.5, i+2.5], [y-0.5,y-0.5], color='red', linewidth=5, solid_capstyle='butt')
        else:
            for i in self.alignment.find_pattern(['ATG','GTG','TTG','CTG'],0,self.alignment.modified_sequence_length,1,tolerance,in_frame = True, frame_start = self.buffer_end-2, method = 'count'):
                    seqlogo.ax.plot([i-0.5, i+2.5], [y,y], color='green', linewidth=5, solid_capstyle='butt')
            for i in self.alignment.find_pattern(['TAG','TGA','TAA'],0,self.alignment.modified_sequence_length,1,tolerance,in_frame = True, frame_start = self.buffer_end-2, method = 'count'):
                seqlogo.ax.plot([i-0.5, i+2.5], [y,y], color='red', linewidth=5, solid_capstyle='butt')
            if self.locus_strand == self.locus_strand_2:
                for i in self.alignment.find_pattern(['ATG','GTG','TTG','CTG'],0,self.alignment.modified_sequence_length,1,tolerance,in_frame = True, frame_start = self.target_end, method = 'count'):
                    seqlogo.ax.plot([i-0.5, i+2.5], [y-0.5,y-0.5], color='green', linewidth=5, solid_capstyle='butt')
                for i in self.alignment.find_pattern(['TAG','TGA','TAA'],0,self.alignment.modified_sequence_length,1,tolerance,in_frame = True, frame_start = self.target_end, method = 'count'):
                    seqlogo.ax.plot([i-0.5, i+2.5], [y-0.5,y-0.5], color='red', linewidth=5, solid_capstyle='butt')
            else:
                for i in self.alignment.find_pattern(['CAT','CAC','CAA','CAG'],0,self.alignment.modified_sequence_length,1,tolerance,in_frame = True, frame_start = self.target_end, method = 'count'):
                    seqlogo.ax.plot([i-0.5, i+2.5], [y-0.5,y-0.5], color='green', linewidth=5, solid_capstyle='butt')
                for i in self.alignment.find_pattern(['CTA','TCA','TTA'],0,self.alignment.modified_sequence_length,1,tolerance,in_frame = True, frame_start = self.target_end, method = 'count'):
                    seqlogo.ax.plot([i-0.5, i+2.5], [y-0.5,y-0.5], color='red', linewidth=5, solid_capstyle='butt')

        #seqlogo.ax.plot([self.utr_start, self.utr_end],[y-0.5, y-0.5], color='mediumslateblue', linewidth=10, solid_capstyle='butt')
        sign_symbol = lambda x : '+' if (x > 0) else '-'
        for i, state in enumerate(self.hmm_model.viterbi_path):
            if state in [0]:
                seqlogo.highlight_position_range(pmin=i, pmax=i, color='rosybrown')
        
        for i, state in enumerate(self.master_hmm_model.viterbi_path):
            if state in [0]:
                seqlogo.ax.plot([i-0.5, i+0.5], [y-1.1,y-1.1], color='purple', linewidth=8, solid_capstyle='butt')
                #seqlogo.highlight_position_range(pmin=i, pmax=i, color='rosybrown')
        
        for j, pairwise_hmm in enumerate(self.hmm_model_list):
            seqlogo.ax.text(-text_offset,y-1.55-0.4*(j+1),self.species_name_dict[self.species_names[j]])
            for i, state in enumerate(pairwise_hmm.viterbi_path):
                if state in [0]:
                    seqlogo.ax.plot([i-0.5, i+0.5], [y-1.5-0.4*(j+1),y-1.5-0.4*(j+1)], color='slategrey', linewidth=8, solid_capstyle='butt')
            for k in self.insertion_locations[self.species_names[j]]:
                seqlogo.ax.plot([k-0.5, k+0.5], [y-1.5-0.4*(j+1),y-1.5-0.4*(j+1)], color='deeppink', linewidth=3, solid_capstyle='butt')
                
        seqlogo.ax.text(-text_offset,y,self.locus_tag + ' ('+sign_symbol(self.locus_strand)+')')
        seqlogo.ax.text(-text_offset,1.2*y,int(self.start), verticalalignment='top', horizontalalignment='left')
        seqlogo.ax.text(self.alignment.modified_sequence_length, y,self.locus_tag_2+ ' ('+sign_symbol(self.locus_strand_2)+')', horizontalalignment='left')#,fontsize=12)
        seqlogo.ax.text(self.alignment.modified_sequence_length, 1.2*y,int(self.end), verticalalignment='top', horizontalalignment='left')
        seqlogo;
        
        sign_symbol = lambda x : '+' if (x > 0) else '-'