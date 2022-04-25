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
    
    def __init__(self, analysis_type, alignment, seq_data, group_id, individual_model_num_states, individual_model_parameters, overall_model, 
                 overall_model_num_states, overall_model_parameters, non_cds_output_dir, tb_species, genome_ids, pairwise_observation_probabilities, 
                 alignment_hmm_model, model, literature_annotations_df_list):
        self.analysis_type = analysis_type
        self.group_id = group_id
        self.alignment = copy.deepcopy(alignment) 
        self.alignment.modify_sequence(1,False,False)
        self.alignment.calculate_entropies(mvave_len = 10)
        self.literature_annotations = [[],[]]
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
        individual_model_list = []
        self.individual_model_viterbi_path = []
        pairwise_state_probabilities = []
        for params in individual_model_parameters:
            transition_probabilities, mutation_probabilities = alignment_hmm_model.alignment_hmm_model_inputs(params[1])
            observation_probabilities = alignment_hmm_model.calculate_observation_probs(mutation_probabilities, self.alignment.modified_sequence_list, self.alignment, 
                                                                                        all_species=False, comparison_species = params[0])
            individual_model_list.append(hmm.HMM(initial_state_probabilities, transition_probabilities, observation_probabilities))
            individual_model_list[-1].calculate_probabilities()
            self.individual_model_viterbi_path.append(individual_model_list[-1].viterbi_path)
            self.species_names.append(params[0])
            pairwise_state_probabilities.append(individual_model_list[-1].state_probabilities)
        
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
        overall_model = hmm.HMM(initial_state_probabilities, transition_probabilities, observation_probabilities)
        overall_model.calculate_probabilities()
        self.overall_model_viterbi_path = overall_model.viterbi_path
        
        #Sequence information for display
       
        if seq_data.master_species_info(self.group_id, 'strand') == 1:
            self.buffer_end = seq_data.master_species_info(self.group_id, 'start') - seq_data.master_species_info(self.group_id, 'cds_extended_region_start')
            self.target_end = seq_data.master_species_info(self.group_id, 'end') - seq_data.master_species_info(self.group_id, 'cds_extended_region_start')
        else:
            self.buffer_end = seq_data.master_species_info(self.group_id, 'cds_extended_region_end') - seq_data.master_species_info(self.group_id, 'end')
            self.target_end = seq_data.master_species_info(self.group_id, 'cds_extended_region_end') - seq_data.master_species_info(self.group_id, 'start')
        
        self.start = seq_data.master_species_info(self.group_id, 'cds_extended_region_start')
        self.end = seq_data.master_species_info(self.group_id, 'cds_extended_region_end')
        self.locus_tag = seq_data.master_species_info(self.group_id, 'locus_tag')
        self.locus_strand = seq_data.master_species_info(self.group_id, 'strand')
        
        if self.locus_strand == 1:
            self.organism_start_co_ordinates = self.start
            self.organism_end_co_ordinates = self.end
            self.locus_tag_1 = seq_data.master_species_info(self.group_id, 'previous_locus_tag')
            self.locus_strand_1 = seq_data.master_species_info(self.group_id, 'prev_strand')
            self.locus_tag_2 = seq_data.master_species_info(self.group_id, 'next_locus_tag')
            self.locus_strand_2 = seq_data.master_species_info(self.group_id, 'next_strand')
        else:
            self.organism_start_co_ordinates = self.end
            self.organism_end_co_ordinates = self.start
            self.locus_tag_1 = seq_data.master_species_info(self.group_id, 'next_locus_tag')
            self.locus_strand_1 = seq_data.master_species_info(self.group_id, 'next_strand')
            self.locus_tag_2 = seq_data.master_species_info(self.group_id, 'previous_locus_tag')
            self.locus_strand_2 = seq_data.master_species_info(self.group_id, 'prev_strand')
        
        
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
        
        for i, r in literature_annotations_df_list[0].iterrows():
            if (r['Start'] <= self.end) and (r['Stop'] >= self.start):
                self.literature_annotations[0].append([r['Feature'], max(self.start, r['Start']), min(self.end, r['Stop'])])
        for i, r in literature_annotations_df_list[1].iterrows():
            if (r['Revised CDS Start'] <= self.end) and (r['Revised CDS Start'] >= self.start):
                self.literature_annotations[1].append(['RASS', r['Revised CDS Start'], r['Revised CDS Start']])
    
    def plot_annotation(self, seqlogo, coordinates_start, coordinates_end, print_coordinates_start, print_coordinates_end, coord_1, coord_2, label, colour, last_pos):
        if coordinates_start < coordinates_end:
            if coord_1 <= print_coordinates_end and coord_2 >= print_coordinates_start:
                    seqlogo.ax.plot([max(coord_1, coordinates_start) - coordinates_start - 1.5, min(coord_2, coordinates_end) - coordinates_start-0.5], 
                                    [last_pos,last_pos], color=colour, linewidth=3, solid_capstyle='butt')
                    seqlogo.ax.text(max(coord_1, coordinates_start, print_coordinates_start) - coordinates_start - 1.5,last_pos - 0.5, label)
                    
        if coordinates_start > coordinates_end:
            if coord_1 <= print_coordinates_start and coord_2 >= print_coordinates_end:
                    seqlogo.ax.plot([coordinates_start - max(coord_1, coordinates_end) - 1.5, coordinates_start - min(coord_2, coordinates_start)-0.5], 
                                    [last_pos,last_pos], color=colour, linewidth=3, solid_capstyle='butt')
                    seqlogo.ax.text(coordinates_start - min(coord_2, print_coordinates_start)-0.5,last_pos - 0.5, label)    
    
    def display_analysis(self, mutation_count_dict, co_ordinate_start = -999, co_ordinate_end = -999):
        
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
        seqlogo.ax.set_ylim([-12, 2])
        seqlogo.ax.set_xlim([plot_start, plot_end])
        seqlogo.ax.axhline(y, color = 'k', linewidth = 1)

        #Title
        sign_symbol = lambda x : '+' if (x > 0) else '-'
        seqlogo.ax.set_title(str(self.locus_tag) + ' ('+sign_symbol(self.locus_strand)+')' + ' and inter CDS regions')

        #Text labels 
        seqlogo.ax.text(plot_start-text_offset,y,str(self.locus_tag_1) + ' ('+sign_symbol(self.locus_strand_1)+')')
        seqlogo.ax.text(plot_start-text_offset,1.2*y,print_coordinates_start, verticalalignment='top', horizontalalignment='left')
        seqlogo.ax.text(plot_end + 1, y,str(self.locus_tag_2)+ ' ('+sign_symbol(self.locus_strand_2)+')', horizontalalignment='left')
        seqlogo.ax.text(plot_end + 1, 1.2*y,print_coordinates_end, verticalalignment='top', horizontalalignment='left')


        # Start and end regions
        seqlogo.ax.plot([self.buffer_end-0.5, self.target_end +0.5], [y,y], color='skyblue', linewidth=10, solid_capstyle='butt')
            

        # Stop, Start codons in frame
        tolerance = 3
        f_sense_1 = False
        
        if self.locus_strand == self.locus_strand_2:
            f_sense_2 = False
            arrow_direction_2 = 1
        else:
            f_sense_2 = True
            arrow_direction_2 = -1
        
        f_start_1 = self.target_end 
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

        # Starts and stops in master species
        
        for i in self.alignment.master_species_find_pattern(['ATG','GTG','TTG','CTG'],0,self.alignment.modified_sequence_length, in_frame = True, frame_start = f_start_1, rev_complement = f_sense_1):
            seqlogo.ax.arrow(i-0.5, y-1, 3, 0, color='green', head_length = 1, head_width = 0.3, width = 0.1, linestyle ='solid', length_includes_head = True, zorder = 3)
        for i in self.alignment.master_species_find_pattern(['TAG','TGA','TAA'],0,self.alignment.modified_sequence_length, in_frame = True, frame_start = f_start_1, rev_complement = f_sense_1):
            seqlogo.ax.arrow(i-0.5, y-1, 3, 0, color='red', head_length = 1, head_width = 0.3, width = 0.1, linestyle ='solid', length_includes_head = True, zorder = 3)
        
        # Overall Viterbi paths - overall and individual
        for i, state in enumerate(self.overall_model_viterbi_path):
            if state in [0]:
                seqlogo.highlight_position_range(pmin=i-0.5, pmax=i+0.5, color='rosybrown')

        last_pos = 0
        for j, path in enumerate(self.individual_model_viterbi_path):
            seqlogo.ax.text(plot_start-text_offset,y-1.55-0.4*(j+1),self.species_name_dict[self.species_names[j]])
            for i, state in enumerate(path):
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
        
        # Literature annotations
        last_pos = last_pos - 0.8
        seqlogo.ax.text(plot_start-text_offset,last_pos-0.05,'Mycobrowser_R4')
        #To DO - Reference print co-ordinates #######################
        for annotation in self.literature_annotations[0]:
            self.plot_annotation(seqlogo, self.organism_start_co_ordinates, self.organism_end_co_ordinates, print_coordinates_start, print_coordinates_end, annotation[1], annotation[2], annotation[0], 'blue', last_pos)
                  
        last_pos = last_pos - 0.8
        seqlogo.ax.text(plot_start-text_offset,last_pos-0.05,'DeJesus (2013)')
        for annotation in self.literature_annotations[1]:
            self.plot_annotation(seqlogo, self.organism_start_co_ordinates, self.organism_end_co_ordinates, print_coordinates_start, print_coordinates_end, annotation[1], annotation[2], annotation[0], 'red', last_pos)
        
        
        for i in range(min(print_coordinates_start, print_coordinates_end), max(print_coordinates_start, print_coordinates_end)):
            v = mutation_count_dict[i]
            if v > 0:
                if self.organism_start_co_ordinates < self.organism_end_co_ordinates:
                    temp = self.organism_start_co_ordinates
                    seqlogo.ax.plot([i - temp - 1, i - temp - 1], 
                                        [last_pos,last_pos+v/10], color='black', linewidth=3, solid_capstyle='butt')
                
                else:
                    temp = self.organism_start_co_ordinates
                    seqlogo.ax.plot([temp -i-1, temp-i - 1], 
                                        [last_pos,last_pos+v/10], color='black', linewidth=3, solid_capstyle='butt')
            
        
        
        seqlogo;                                               

        