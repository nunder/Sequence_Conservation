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

#  Dataset manipulation

def generate_protein_file(input_fileloc, output_fileloc):
    protein_list = []
    with open(output_fileloc, "w") as f:
        genome_record = next(SeqIO.parse(input_fileloc, "genbank"))
        for feature in genome_record.features:
            a = feature.qualifiers
            if feature.type == 'CDS':
                if (a.get('protein_id') != None) and (a.get('translation') != None) and (not(a.get('protein_id')[0] in protein_list)): 
                    #Don't output duplicates
                    protein_list.append(a.get('protein_id')[0])
                    f.write('>' + a.get('protein_id')[0] + '\n' + a.get('translation')[0] + '\n')

def run_sonic_paranoid(protein_file_location, output_location, run_name):
    subprocess.run('wsl cd ~; source sonicparanoid/bin/activate; sonicparanoid -i ' + wslname(protein_file_location) +' -o ' + wslname(output_location) + ' -p ' + run_name + ' -t 8 -ot' , shell=True)


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

def parse_genbank(input_filename, non_cds_offset = 0):
    offset = non_cds_offset
    temp = list()
    genome_record = next(SeqIO.parse(input_filename, "genbank"))
    organism_name = genome_record.annotations['organism']
    full_sequence = genome_record.seq
    for feature in genome_record.features:
        a = feature.qualifiers
        featureseq = feature.extract(full_sequence)
        #if not (feature.type in ['source','gene','ncRNA']):   # Exclude full source and gene (appears as CDS), exclude ncRNA as not consistently annotated 
        #but include other features eg tRNA as these should be conserved and not form part of the 'non=CDS' regions under consideration
        if feature.type == 'CDS':
            if a.get("locus_tag") != None:
                locus_tag = a.get("locus_tag")[0]
            else:
                locus_tag = ''
            if a.get("protein_id") != None:
                protein_id = a.get("protein_id")[0]
            else:
                protein_id = ''
            temp.append([organism_name, feature.type, feature.location.start, feature.location.end, feature.location.strand, str(feature.location), locus_tag, protein_id, str(featureseq), len(featureseq) ])              
    df  = pd.DataFrame(data = temp, columns = ['name','type','start','end','strand','loc','locus_tag','protein_id', 'cds_seq','cds_length'])
    df[['non_cds_start','non_cds_end','upstream_non_cds_start','upstream_non_cds_end','ss_non_cds_start','ss_non_cds_end']]=int
    df_p = df[df['strand']==1]
    df_n = df[df['strand']==-1]
    df_p = df_p.sort_values(['start'])
    df_n = df_n.sort_values(['start'])
    df_p['ss_next_start'] = df_p['start'].shift(-1)
    df_p['ss_prev_end'] = df_p['end'].shift(1)
    df_n['ss_next_start'] = df_n['start'].shift(-1)
    df_n['ss_prev_end'] = df_n['end'].shift(1)
    df = pd.concat([df_p,df_n],ignore_index = True)
    df = df.sort_values(['start']) 
    df['next_strand']=df['strand'].shift(-1)
    df['next_start']=df['start'].shift(-1)
    df['prev_strand']=df['strand'].shift(1)
    df['prev_end']=df['end'].shift(1)
    df['previous_locus_tag']=df['locus_tag'].shift(1)
    df['next_locus_tag']=df['locus_tag'].shift(-1)
    for i, r in df.iterrows():
        if not(r['next_start'] > 0):
            df.at[i,'next_start'] = len(full_sequence)
            df.at[i,'next_strand'] = r['strand']
        if not(r['ss_next_start'] > 0):
            df.at[i,'ss_next_start'] = len(full_sequence)
        if not(r['prev_end'] >0):
            df.at[i,'prev_end']=0
            df.at[i,'prev_strand'] = r['strand']
        if not(r['ss_prev_end'] >0):
            df.at[i,'ss_prev_end']=0
    
    for i, r in df.iterrows():
        if r['strand']==1:
            if r['next_strand']==1:
                df.at[i,'bp_restrict'] = 0
            else:
                df.at[i,'bp_restrict'] = 1
          
            df.at[i,'non_cds_start'] = int(r['end'])
            df.at[i,'non_cds_end'] = int(r['next_start'])
            df.at[i,'non_cds_seq']=str(genome_record.seq[int(r['end']):int(r['next_start'])])
            df.at[i,'non_cds_offset_seq']=str(genome_record.seq[max(0,int(r['end'])-offset):min(len(full_sequence),int(r['next_start'])+offset)])
            df.at[i,'non_cds_offset_start']=max(0,int(r['end'])-offset)
            df.at[i,'non_cds_offset_end']= min(len(full_sequence),int(r['next_start'])+offset)
            df.at[i,'ss_non_cds_start'] = int(r['end'])
            df.at[i,'upstream_non_cds_start'] = int(r['prev_end'])
            df.at[i,'upstream_non_cds_end'] = int(r['start'])
            df.at[i,'upstream_non_cds_seq']=str(genome_record.seq[int(r['prev_end']):int(r['start'])])
            df.at[i,'upstream_non_cds_offset_seq']=str(genome_record.seq[max(0, int(r['prev_end']) - offset):min(len(full_sequence),int(r['start'])+offset)])
            df.at[i,'upstream_non_cds_offset_start']= max(0, int(r['prev_end']) - offset)
            df.at[i,'upstream_non_cds_offset_end']= min(len(full_sequence),int(r['start'])+offset)
            df.at[i,'ss_non_cds_end'] = int(r['ss_next_start'])
            df.at[i,'ss_non_cds_seq']=str(genome_record.seq[int(r['end']):int(r['ss_next_start'])])
            df.at[i,'cds_extended_region_seq']= str(genome_record.seq[int(r['prev_end']):int(r['start'])]) + r['cds_seq'] + str(genome_record.seq[int(r['end']):int(r['next_start'])])
            df.at[i,'cds_extended_region_start']= int(r['prev_end'])
            df.at[i,'cds_extended_region_end']= int(r['next_start'])
        else:
            if r['prev_strand']== -1:
                df.at[i,'bp_restrict'] = 0
            else:
                df.at[i,'bp_restrict'] = 1
            df.at[i,'non_cds_start'] = int(r['prev_end'])
            df.at[i,'non_cds_end'] = int(r['start'])
            df.at[i,'non_cds_seq']=str((genome_record.seq[int(r['prev_end']):int(r['start'])]).reverse_complement())
            df.at[i,'non_cds_offset_seq']=str((genome_record.seq[max(0,int(r['prev_end'])-offset):min(len(full_sequence),int(r['start']) + offset)]).reverse_complement())
            df.at[i,'non_cds_offset_start']=max(0,int(r['prev_end'])-offset)
            df.at[i,'non_cds_offset_end']= min(len(full_sequence),int(r['start']) + offset)
            df.at[i,'upstream_non_cds_start'] = int(r['end'])
            df.at[i,'upstream_non_cds_end'] = int(r['next_start'])
            df.at[i,'upstream_non_cds_seq']=str((genome_record.seq[int(r['end']):int(r['next_start'])]).reverse_complement())
            df.at[i,'upstream_non_cds_offset_seq']=str((genome_record.seq[max(0,int(r['end'])-offset):min(len(full_sequence),int(r['next_start']) + offset)]).reverse_complement())
            df.at[i,'upstream_non_cds_offset_start']=max(0,int(r['end'])-offset)
            df.at[i,'upstream_non_cds_offset_end']=min(len(full_sequence),int(r['next_start']) + offset)
            df.at[i,'ss_non_cds_start'] = int(r['ss_prev_end'])
            df.at[i,'ss_non_cds_end'] = int(r['start'])
            df.at[i,'ss_non_cds_seq']=str((genome_record.seq[int(r['ss_prev_end']):int(r['start'])]).reverse_complement())
            df.at[i,'cds_extended_region_seq']= str((genome_record.seq[int(r['end']):int(r['next_start'])]).reverse_complement()) + r['cds_seq'] + str((genome_record.seq[int(r['prev_end']):int(r['start'])]).reverse_complement())
            df.at[i,'cds_extended_region_start']= int(r['prev_end'])
            df.at[i,'cds_extended_region_end']= int(r['next_start'])
    
    for i, r in df.iterrows():
            if (r['non_cds_start'] < r['non_cds_end']):
                df.at[i,'non_cds_loc'] = str(SeqFeature(FeatureLocation(r['non_cds_start'], r['non_cds_end']), type="gene", strand=r['strand']).location)
                df.at[i,'non_cds_length'] = r['non_cds_end'] - r['non_cds_start']
            else:
                df.at[i,'non_cds_loc'] = ''
                df.at[i,'non_cds_length'] = 0
            df.at[i,'non_cds_offset_length'] = len(r['non_cds_offset_seq'])  
            df.at[i,'upstream_non_cds_offset_length'] = len(r['upstream_non_cds_offset_seq'])  
            df.at[i,'cds_extended_region_length'] = len(r['cds_extended_region_seq'])  
            if (r['upstream_non_cds_start'] < r['upstream_non_cds_end']):
                df.at[i,'upstream_non_cds_loc'] = str(SeqFeature(FeatureLocation(r['upstream_non_cds_start'], r['upstream_non_cds_end']), type="gene", strand=r['strand']).location)
                df.at[i,'upstream_non_cds_length'] = r['upstream_non_cds_end'] - r['upstream_non_cds_start']
            else:
                df.at[i,'upstream_non_cds_loc'] = ''
                df.at[i,'upstream_non_cds_length'] = 0
                
            if (r['ss_non_cds_start'] < r['ss_non_cds_end']):
                df.at[i,'ss_non_cds_loc'] = str(SeqFeature(FeatureLocation(r['ss_non_cds_start'], r['ss_non_cds_end']), type="gene", strand=r['strand']).location)
                df.at[i,'ss_non_cds_length'] = r['ss_non_cds_end'] - r['ss_non_cds_start']
            else:
                df.at[i,'ss_non_cds_loc'] = '' 
                df.at[i,'ss_non_cds_length'] = 0
                   
    df = df[['name','type','locus_tag','previous_locus_tag','next_locus_tag','protein_id','bp_restrict','loc','non_cds_loc','upstream_non_cds_loc','ss_non_cds_loc','strand','prev_strand',
             'next_strand', 'start','end', 'cds_seq','non_cds_seq', 'non_cds_offset_seq','upstream_non_cds_seq','upstream_non_cds_offset_seq', 'cds_extended_region_seq', 'non_cds_start','non_cds_end','upstream_non_cds_start',
             'upstream_non_cds_end','cds_extended_region_start','cds_extended_region_end','ss_non_cds_seq','cds_length', 
             'non_cds_length','upstream_non_cds_length','cds_extended_region_length', 'ss_non_cds_length', 'non_cds_offset_length',
             'non_cds_offset_start','non_cds_offset_end','upstream_non_cds_offset_length','upstream_non_cds_offset_start','upstream_non_cds_offset_end']]                                  
    return df

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


class Ortholog_Grouping:
    def __init__(self, file_loc):
        orthologs_df = pd.read_csv(file_loc, sep="\t", header=0)
        file_list = list(orthologs_df)
        file_list.remove('group_id')
        self.num_ids = len(file_list)
        orthologs_df = orthologs_df.melt(id_vars = ['group_id'], value_vars = file_list)
        orthologs_df = orthologs_df.rename(columns={'variable':'species', 'value':'protein_id'})
        for i, r in tqdm(orthologs_df.iterrows(), total=orthologs_df.shape[0]):
            orthologs_df.at[i,'species'] = r['species'][:-4]
            if r['protein_id'][0] == '*':
                orthologs_df.at[i,'num_protein_ids'] = 0
            else:
                orthologs_df.at[i,'protein_id'] = r['protein_id'].split(',')
                orthologs_df.at[i,'num_protein_ids'] = len(r['protein_id'].split(','))
                
        self.all_copy_orthologs_df = orthologs_df.explode('protein_id', ignore_index=True)        
        temp_df = self.all_copy_orthologs_df.groupby('group_id',as_index=False)['protein_id'].count()
        self.all_copy_ortholog_groups = temp_df['group_id'].tolist()

        self.single_copy_orthologs_df = orthologs_df[orthologs_df['num_protein_ids']==1]   #  Remove species not in ortholog group or where more than one protein 
        self.single_copy_orthologs_df = self.single_copy_orthologs_df.explode('protein_id', ignore_index=True)       
        temp_df = self.single_copy_orthologs_df.groupby('group_id',as_index=False)['protein_id'].count()
        self.single_copy_ortholog_groups = temp_df['group_id'].tolist()
        self.full_single_copy_ortholog_groups = temp_df[temp_df['protein_id'] == self.num_ids]['group_id'].tolist()



class Ortholog_Sequence_Dataset:
    def __init__(self, ortholog_grouping, genome_datasets_dir, genome_ids, non_cds_offset, master_species, single_copy = True, num_cores = 16):
        core_numbers = list(range(1, num_cores+1))
        parallel_output = Parallel(n_jobs=-1)(delayed(self.parallel_populate)(num_cores, core_number, ortholog_grouping, genome_datasets_dir, genome_ids, 
                                                                              non_cds_offset, master_species, single_copy) for core_number in tqdm(core_numbers))
        df_list = [item for sublist in parallel_output for item in sublist]
        self.sequence_data = pd.concat(df_list)  
        self.master_species = master_species
        #self.sequence_data.to_csv(output_dir + '/Ortholog_Master_Data/'+'sequence_master_data.csv')
    
    def master_species_info(self, group_id, fieldname):
        temp_df = self.sequence_data[self.sequence_data['group_id'] == group_id]
        return temp_df[temp_df['species'] == self.master_species].iloc[0][fieldname]

    def species_info(self):
        return self.sequence_data.drop_duplicates(['name','species'])[['name','species']]
    
    def parallel_populate(self, num_subsets, subset_num, ortholog_grouping, genome_datasets_dir, genome_ids, non_cds_offset, master_species, single_copy): 
        genomes = util.chunk_list(genome_ids, num_subsets, subset_num)
        df_list = []
        for id in genomes:
            cds_data = parse_genbank(genome_datasets_dir + '/' + id +'/genomic.gbff',non_cds_offset)
            
            if single_copy == True:
                orthologs_for_id = ortholog_grouping.single_copy_orthologs_df[ortholog_grouping.single_copy_orthologs_df['species'] == id]
                orthologs_and_cds_info = orthologs_for_id.merge(cds_data, how = 'left', left_on = 'protein_id', right_on = 'protein_id')
                orthologs_and_cds_info.drop_duplicates(subset = ['protein_id'], keep=False, inplace=True)   # Remove instances where same protein is coded in multiple locuses
            else:
                orthologs_for_id = ortholog_grouping.single_copy_orthologs_df[ortholog_grouping.all_copy_orthologs_df['species'] == id]
                orthologs_and_cds_info = orthologs_for_id.merge(cds_data, how = 'left', left_on = 'protein_id', right_on = 'protein_id')
            orthologs_and_cds_info = orthologs_and_cds_info[orthologs_and_cds_info['name']==orthologs_and_cds_info['name']]    #Remove proteins which can't be matched back
            df_list.append(orthologs_and_cds_info)
        return df_list
    
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
