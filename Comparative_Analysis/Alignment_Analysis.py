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
from . import Master_Alignment_HMM as master_alignment_hmm
from . import Multi_Species_Master_Alignment_HMM as multi_species_master_alignment_hmm
from . import Arneson_Ernst_HMM as ae_hmm
import copy

class Alignment_Analysis:
    
    def __init__(self, analysis_type, alignment, seq_data, non_cds_offset, group_id, individual_model_num_states, individual_model_parameters, overall_model, 
                 overall_model_num_states, overall_model_parameters, non_cds_output_dir, tb_species, genome_ids, pairwise_observation_probabilities, alignment_hmm_model, model):
        self.analysis_type = analysis_type
        self.group_id = group_id
        self.alignment = copy.deepcopy(alignment) 
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
       
        # Individual HMMs -one per species
        initial_state_probabilities = [1.0/individual_model_num_states]*individual_model_num_states
        self.species_names = []
        self.individual_model_list = []
        pairwise_state_probabilities = []
        for params in individual_model_parameters:
            transition_probabilities, mutation_probabilities = alignment_hmm_model.alignment_hmm_model_inputs(params[1])
            observation_probabilities = alignment_hmm_model.calculate_observation_probs(mutation_probabilities, self.alignment.modified_sequence_list, self.alignment, 
                                                                                        all_species=False, comparison_species = params[0])
            self.individual_model_list.append(hmm.HMM(initial_state_probabilities, transition_probabilities, observation_probabilities))
            self.individual_model_list[-1].calculate_probabilities()
            self.species_names.append(params[0])
            pairwise_state_probabilities.append(self.individual_model_list[-1].state_probabilities)
        
        # Overall HMM
        initial_state_probabilities = [1.0/overall_model_num_states]*overall_model_num_states
        if overall_model == 'Simple':
            transition_probabilities, mutation_probabilities = model.alignment_hmm_model_inputs(overall_model_parameters)
            observation_probabilities = model.calculate_observation_probs(mutation_probabilities, pairwise_state_probabilities)
        elif overall_model == 'Multi_Species':
            transition_probabilities, mutation_probabilities = model.alignment_hmm_model_inputs(overall_model_parameters)
            observation_probabilities = model.calculate_observation_probs(mutation_probabilities, pairwise_state_probabilities)
        elif overall_model == 'AE':
            transition_probabilities, mutation_probabilities = model.alignment_hmm_model_inputs(overall_model_parameters)
            observation_probabilities = model.calculate_observation_probs(mutation_probabilities, self.alignment.modified_sequence_list, self.alignment)
        else:
            pass
        self.overall_model = hmm.HMM(initial_state_probabilities, transition_probabilities, observation_probabilities)
        self.overall_model.calculate_probabilities()
        
        
        #Sequence information for display
        self.buffer_end = non_cds_offset - 1
        self.target_end = self.alignment.modified_sequence_length - non_cds_offset
        if analysis_type == 'Downstream':
            self.start = seq_data.master_species_info(self.group_id, 'non_cds_offset_start')
            self.end = seq_data.master_species_info(self.group_id, 'non_cds_offset_end')
            self.locus_tag = seq_data.master_species_info(self.group_id, 'locus_tag')
            self.locus_strand = seq_data.master_species_info(self.group_id, 'strand')
            if self.locus_strand == 1:
                self.organism_start_co_ordinates = self.start
                self.organism_end_co_ordinates = self.end
                self.locus_tag_2 = seq_data.master_species_info(self.group_id, 'next_locus_tag')
                self.locus_strand_2 = seq_data.master_species_info(self.group_id, 'next_strand')
            else:
                self.organism_start_co_ordinates = self.end
                self.organism_end_co_ordinates = self.start
                self.locus_tag_2 = seq_data.master_species_info(self.group_id, 'previous_locus_tag')
                self.locus_strand_2 = seq_data.master_species_info(self.group_id, 'prev_strand')
        else:
            self.start = seq_data.master_species_info(self.group_id, 'upstream_non_cds_offset_start')
            self.end = seq_data.master_species_info(self.group_id, 'upstream_non_cds_offset_end')
            self.locus_tag_2 = seq_data.master_species_info(self.group_id, 'locus_tag')
            self.locus_strand_2 = seq_data.master_species_info(self.group_id, 'strand')
            if self.locus_strand_2 == 1:
                self.organism_start_co_ordinates = self.start
                self.organism_end_co_ordinates = self.end
                self.locus_tag = seq_data.master_species_info(self.group_id, 'previous_locus_tag')
                self.locus_strand = seq_data.master_species_info(self.group_id, 'prev_strand')
            else:
                self.organism_start_co_ordinates = self.end
                self.organism_end_co_ordinates = self.start
                self.locus_tag = seq_data.master_species_info(self.group_id, 'next_locus_tag')
                self.locus_strand = seq_data.master_species_info(self.group_id, 'next_strand')
            
        self.counts_df = lm.alignment_to_matrix(sequences = self.alignment.modified_sequence_list, to_type = 'counts', characters_to_ignore = '-', pseudocount=0)
        self.background_probs = [0.25, 0.25, 0.25, 0.25]
        for i, r in self.counts_df.iterrows():
            temp_relent = []
            num_gaps = self.alignment.num_sequences
            for k in range(4):
                num_gaps = num_gaps - r.iloc[k]
            for k in range(4):
                ct = r.iloc[k] + num_gaps*self.background_probs[k]
                if ct == 0:
                    temp_relent.append(0)
                else:
                    temp_relent.append((ct /self.alignment.num_sequences) * math.log((ct /self.alignment.num_sequences)/self.background_probs[k],2))
            for k in range(4):
                r.iloc[k] = temp_relent[k]
        
    def display_analysis(self, co_ordinate_start = -999, co_ordinate_end = -999):
        
        if co_ordinate_end < 0:
            plot_start = -0.5
            plot_end = abs(self.end - self.start) - 0.5
            print_coordinates_start = int(self.organism_start_co_ordinates)
            print_coordinates_end = int(self.organism_end_co_ordinates)
        else:
            if self.organism_start_co_ordinates < self.organism_end_co_ordinates:
                plot_start = co_ordinate_start - self.organism_start_co_ordinates - 0.5
                plot_end = co_ordinate_end - self.organism_start_co_ordinates-0.5
                print_coordinates_start = co_ordinate_start
                print_coordinates_end = co_ordinate_end
            else:
                plot_start = self.organism_start_co_ordinates - max(co_ordinate_start, co_ordinate_end) -0.5
                plot_end = self.organism_start_co_ordinates - min(co_ordinate_start, co_ordinate_end) -0.5
                print_coordinates_start = max(co_ordinate_end, co_ordinate_start)
                print_coordinates_end = min(co_ordinate_end, co_ordinate_start)     
        plot_length = (plot_end - plot_start)

        y = -1        
        text_offset = 0.06 * plot_length

        seqlogo = lm.Logo(self.counts_df, figsize = [30,6])
        seqlogo.style_spines(visible=False)
        seqlogo.style_spines(spines=['left'], visible=True, bounds=[0, 2])
        seqlogo.ax.set_xticks([])
        seqlogo.ax.set_yticks([0,2])
        seqlogo.ax.set_ylim([-10.5, 2])
        seqlogo.ax.set_xlim([plot_start, plot_end])
        seqlogo.ax.axhline(y, color = 'k', linewidth = 1)

        #Title
        if self.analysis_type == 'Upstream':
            seqlogo.ax.set_title(self.analysis_type + ' of ' + self.locus_tag_2)
        else:
            seqlogo.ax.set_title(self.analysis_type + ' of ' + self.locus_tag)

        #Text labels 
        sign_symbol = lambda x : '+' if (x > 0) else '-'
        seqlogo.ax.text(plot_start-text_offset,y,str(self.locus_tag) + ' ('+sign_symbol(self.locus_strand)+')')
        seqlogo.ax.text(plot_start-text_offset,1.2*y,print_coordinates_start, verticalalignment='top', horizontalalignment='left')
        seqlogo.ax.text(plot_end + 1, y,str(self.locus_tag_2)+ ' ('+sign_symbol(self.locus_strand_2)+')', horizontalalignment='left')
        seqlogo.ax.text(plot_end + 1, 1.2*y,print_coordinates_end, verticalalignment='top', horizontalalignment='left')


        # Start and end regions
        seqlogo.ax.plot([-0.5, self.buffer_end+0.5], [y,y], color='skyblue', linewidth=10, solid_capstyle='butt')
        seqlogo.ax.plot([self.target_end-0.5, self.alignment.modified_sequence_length +0.5], [y,y], color='skyblue', linewidth=10, solid_capstyle='butt')



      # Stop, Start codons in frame
        tolerance = 3
        f_sense_1 = False
        if self.locus_strand == self.locus_strand_2:
            f_sense_2 = False
            arrow_direction_2 = 1
        else:
            f_sense_2 = True
            arrow_direction_2 = -1
        if self.analysis_type == 'Upstream': 
            f_start_1 = self.target_end
            f_start_2 = self.buffer_end - 2
        else:
            f_start_1 = self.buffer_end - 2
            f_start_2 = self.target_end


        for i in self.alignment.find_pattern(['ATG','GTG','TTG','CTG'],0,self.alignment.modified_sequence_length,1,tolerance,in_frame = True, frame_start = f_start_1, method = 'count', rev_complement = f_sense_1):
            seqlogo.ax.arrow(i-0.5, y, 3, 0, color='green', head_length = 1, head_width = 0.3, width = 0.1, linestyle ='solid', length_includes_head = True, zorder = 3)
        for i in self.alignment.find_pattern(['TAG','TGA','TAA'],0,self.alignment.modified_sequence_length,1,tolerance,in_frame = True, frame_start = f_start_1, method = 'count', rev_complement = f_sense_1):
            seqlogo.ax.arrow(i-0.5, y, 3, 0, color='red', head_length = 1, head_width = 0.3, width = 0.1, linestyle ='solid', length_includes_head = True, zorder = 3)
        for i in self.alignment.find_pattern(['GGAG','GAGG','AGGA'],0,self.alignment.modified_sequence_length,1,tolerance, method = 'count', rev_complement = f_sense_1):
            seqlogo.ax.arrow(i-0.5, y, 4, 0, color='pink', head_length = 1, head_width = 0.3, width = 0.1, linestyle ='solid', length_includes_head = True, zorder = 3)
        for i in self.alignment.find_pattern(['TANNNT'] , 0 , self.alignment.modified_sequence_length, 1.3,0, method = 'entropy', rev_complement = f_sense_1):
            seqlogo.ax.arrow(i-0.5, y, 6, 0, color='orange', head_length = 1, head_width = 0.3, width = 0.1, linestyle ='solid', length_includes_head = True, zorder = 3)

        for i in self.alignment.find_pattern(['ATG','GTG','TTG','CTG'],0,self.alignment.modified_sequence_length,1,tolerance,in_frame = True, frame_start = f_start_2, method = 'count', rev_complement = f_sense_2):
            seqlogo.ax.arrow(i+1-1.5*arrow_direction_2, y-0.5, 3*arrow_direction_2, 0, color='green', head_length = 1, head_width = 0.3, width = 0.1, linestyle ='solid', length_includes_head = True)
        for i in self.alignment.find_pattern(['TAG','TGA','TAA'],0,self.alignment.modified_sequence_length,1,tolerance,in_frame = True, frame_start = f_start_2, method = 'count', rev_complement = f_sense_2):
            seqlogo.ax.arrow(i+1-1.5*arrow_direction_2, y-0.5, 3*arrow_direction_2, 0, color='red', head_length = 1, head_width = 0.3, width = 0.1, linestyle ='solid', length_includes_head = True)
        for i in self.alignment.find_pattern(['GGAG','GAGG','AGGA'],0,self.alignment.modified_sequence_length,1,tolerance,method = 'count', rev_complement = f_sense_2):
            seqlogo.ax.arrow(i+1.5-2*arrow_direction_2, y-0.5, 4*arrow_direction_2, 0, color='pink', head_length = 1, head_width = 0.3, width = 0.1, linestyle ='solid', length_includes_head = True)
        for i in self.alignment.find_pattern(['TANNNT'] , 0 , self.alignment.modified_sequence_length, 1.3,0, method = 'entropy', rev_complement = f_sense_1):
            seqlogo.ax.arrow(i+2.75-3.25*arrow_direction_2, y-0.5, 6*arrow_direction_2, 0, color='orange', head_length = 1, head_width = 0.3, width = 0.1, linestyle ='solid', length_includes_head = True)



        for i, state in enumerate(self.overall_model.viterbi_path):
            if state in [0]:
                seqlogo.highlight_position_range(pmin=i-0.5, pmax=i+0.5, color='rosybrown')

        last_pos = 0
        for j, pairwise_hmm in enumerate(self.individual_model_list):
            seqlogo.ax.text(plot_start-text_offset,y-1.55-0.4*(j+1),self.species_name_dict[self.species_names[j]])
            for i, state in enumerate(pairwise_hmm.viterbi_path):
                if state in [0]:
                    seqlogo.ax.plot([i-0.5, i+0.5], [y-1.5-0.4*(j+1),y-1.5-0.4*(j+1)], color='slategrey', linewidth=8, solid_capstyle='butt')
            for k in self.insertion_locations[self.species_names[j]]:
                seqlogo.ax.plot([k-0.5, k+0.5], [y-1.5-0.4*(j+1),y-1.5-0.4*(j+1)], color='deeppink', linewidth=3, solid_capstyle='butt')
        last_pos = y-1.5-0.4*(j+1)
        last_pos = last_pos - 0.4

          # All six reading frames at foot
        rf = 0
        for reverse_complement in [False,True]:
            if reverse_complement == False:
                dx = 3
                start = -0.5
            else:
                dx = -3
                start = 2.5
            for reading_frame in range(3):
                rf += 1
                last_pos = last_pos - 0.4
                seqlogo.ax.text(plot_start-text_offset,last_pos-0.05,'Reading Frame '+str(rf))
                for i in self.alignment.find_pattern(['ATG','GTG','TTG','CTG'],0,self.alignment.modified_sequence_length,1,tolerance,in_frame = True, 
                                                     frame_start = self.target_end + reading_frame, method = 'count', rev_complement = reverse_complement):
                     seqlogo.ax.arrow(i+start, last_pos, dx, 0, color='green', head_length = 1, head_width = 0.3, width = 0.1, linestyle ='solid', length_includes_head = True)
                for i in self.alignment.find_pattern(['TAG','TGA','TAA'],0,self.alignment.modified_sequence_length,1,tolerance,in_frame = True,
                                                     frame_start = self.target_end + reading_frame, method = 'count', rev_complement = reverse_complement):
                    seqlogo.ax.arrow(i+start, last_pos, dx, 0, color='red', head_length = 1, head_width = 0.3, width = 0.1, linestyle ='solid', length_includes_head = True)


        seqlogo;
        