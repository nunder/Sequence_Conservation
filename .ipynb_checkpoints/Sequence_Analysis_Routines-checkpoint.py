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

def chunk_list(id_list, num_subsets, subset_num):
    len_ids = len(id_list)
    subset_size = int(len_ids / num_subsets)
    if subset_num == num_subsets:
        ids = id_list[(subset_num - 1) * subset_size:]
    else:
        ids = id_list[(subset_num - 1) * subset_size: (subset_num ) * subset_size]
    return ids

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
    
def align_and_build(id_list, num_subsets, subset_num, source_data, length_field, seq_field, out_loc, min_species): 
    muscle_exe = 'C:/Users/nicho/Muscle/muscle3.8.31_i86win32.exe'
    ids = chunk_list(id_list, num_subsets, subset_num)
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
            delete_if_exists(out_loc +'temp'+str(j)+'.fasta')

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
            df.at[i,'non_cds_offset_stop']= min(len(full_sequence),int(r['next_start'])+offset)
            df.at[i,'ss_non_cds_start'] = int(r['end'])
            df.at[i,'upstream_non_cds_start'] = int(r['prev_end'])
            df.at[i,'upstream_non_cds_end'] = int(r['start'])
            df.at[i,'upstream_non_cds_seq']=str(genome_record.seq[int(r['prev_end']):int(r['start'])])
            df.at[i,'upstream_non_cds_offset_seq']=str(genome_record.seq[max(0, int(r['prev_end']) - offset):min(len(full_sequence),int(r['start'])+offset)])
            df.at[i,'upstream_non_cds_offset_start']= max(0, int(r['prev_end']) - offset)
            df.at[i,'upstream_non_cds_offset_end']= min(len(full_sequence),int(r['start'])+offset)
            df.at[i,'ss_non_cds_end'] = int(r['ss_next_start'])
            df.at[i,'ss_non_cds_seq']=str(genome_record.seq[int(r['end']):int(r['ss_next_start'])])
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
            df.at[i,'non_cds_offset_stop']= min(len(full_sequence),int(r['start']) + offset)
            df.at[i,'upstream_non_cds_start'] = int(r['end'])
            df.at[i,'upstream_non_cds_end'] = int(r['next_start'])
            df.at[i,'upstream_non_cds_seq']=str((genome_record.seq[int(r['end']):int(r['next_start'])]).reverse_complement())
            df.at[i,'upstream_non_cds_offset_seq']=str((genome_record.seq[max(0,int(r['end'])-offset):min(len(full_sequence),int(r['next_start']) + offset)]).reverse_complement())
            df.at[i,'upstream_non_cds_offset_start']=max(0,int(r['end'])-offset)
            df.at[i,'upstream_non_cds_offset_end']=min(len(full_sequence),int(r['next_start']) + offset)
            df.at[i,'ss_non_cds_start'] = int(r['ss_prev_end'])
            df.at[i,'ss_non_cds_end'] = int(r['start'])
            df.at[i,'ss_non_cds_seq']=str((genome_record.seq[int(r['ss_prev_end']):int(r['start'])]).reverse_complement())
    
    for i, r in df.iterrows():
            if (r['non_cds_start'] < r['non_cds_end']):
                df.at[i,'non_cds_loc'] = str(SeqFeature(FeatureLocation(r['non_cds_start'], r['non_cds_end']), type="gene", strand=r['strand']).location)
                df.at[i,'non_cds_length'] = r['non_cds_end'] - r['non_cds_start']
            else:
                df.at[i,'non_cds_loc'] = ''
                df.at[i,'non_cds_length'] = 0
            df.at[i,'non_cds_offset_length'] = len(r['non_cds_offset_seq'])  
            df.at[i,'upstream_non_cds_offset_length'] = len(r['upstream_non_cds_offset_seq'])  
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
             'next_strand', 'start','end', 'cds_seq','non_cds_seq', 'non_cds_offset_seq','upstream_non_cds_seq','upstream_non_cds_offset_seq', 'non_cds_start','non_cds_end','upstream_non_cds_start',
             'upstream_non_cds_end','ss_non_cds_seq','cds_length', 'non_cds_length','upstream_non_cds_length','ss_non_cds_length', 'non_cds_offset_length',
             'non_cds_offset_start','non_cds_offset_stop','upstream_non_cds_offset_length','upstream_non_cds_offset_start','upstream_non_cds_offset_end']]                                  
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
        alphabet_list =['A','C','T','G']
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
                    pass
                else:
                    relent = relent + (ct/num_sequences) * math.log((ct/num_sequences)/background_probs[j],2)
            cumulative_relent = cumulative_relent + relent  
            relent_list.append(relent)
    if element_wise == True:
        return relent_list
    else:
        return cumulative_relent


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
    def __init__(self, ortholog_grouping, genome_datasets_dir, genome_ids, offset, master_species, single_copy = True):
        df_list = list()
        match_stats = list()
        for id in tqdm(genome_ids):
            cds_data = parse_genbank(genome_datasets_dir + '/' + id +'/genomic.gbff',offset)
            
            if single_copy == True:
                orthologs_for_id = ortholog_grouping.single_copy_orthologs_df[ortholog_grouping.single_copy_orthologs_df['species'] == id]
                orthologs_and_cds_info = orthologs_for_id.merge(cds_data, how = 'left', left_on = 'protein_id', right_on = 'protein_id')
                orthologs_and_cds_info.drop_duplicates(subset = ['protein_id'], keep=False, inplace=True)   # Remove instances where same protein is coded in multiple locuses
            else:
                orthologs_for_id = ortholog_grouping.single_copy_orthologs_df[ortholog_grouping.all_copy_orthologs_df['species'] == id]
                orthologs_and_cds_info = orthologs_for_id.merge(cds_data, how = 'left', left_on = 'protein_id', right_on = 'protein_id')
                
            temp_1 = len(orthologs_and_cds_info[orthologs_and_cds_info['name']==orthologs_and_cds_info['name']])
            temp_2 = orthologs_and_cds_info.protein_id.nunique()
            match_stats.append([id, temp_1, temp_2, round(temp_1/temp_2*100,1)]) 
            orthologs_and_cds_info = orthologs_and_cds_info[orthologs_and_cds_info['name']==orthologs_and_cds_info['name']]    #Remove proteins which can't be matched back
            df_list.append(orthologs_and_cds_info)
        self.sequence_data = pd.concat(df_list)  
        self.match_info = pd.DataFrame(match_stats, columns = ['id','num_orthologs','num_matches','match_pct'])
        self.master_species = master_species
        #dna_data.to_csv(output_dir + '/Ortholog_Master_Data/'+'dna_data.csv')
        #match_info.to_csv(output_dir + '/Ortholog_Master_Data/'+'match_info.csv')
    
    def master_species_info(self, group_id, fieldname):
        temp_df = self.sequence_data[self.sequence_data['group_id'] == group_id]
        return temp_df[temp_df['species'] == self.master_species].iloc[0][fieldname]

    def species_info(self):
        return self.sequence_data.drop_duplicates(['name','species'])[['name','species']]
    
    
class Alignment:
    def __init__(self, fileloc, master_species, alphabet_name, insert_symbol = '-'): #  group_id, mvave_len, remove_insertions = 'Y', consensus = 1):
        temp = read_fasta_to_arrays(fileloc)
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
            self.relative_entropy = relative_entropy(self.modified_sequence_list,alphabet_name = self.alphabet_name)
        else:
            self.relative_entropy = relative_entropy(self.sequence_list,alphabet_name = self.alphabet_name)
        self.mave_relative_entropy = []
        for k in range(len(self.relative_entropy)):
            mv_temp = int(mvave_len/2)
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

    
class HMM:
    def __init__(self, initial_state_probabilities, transition_probabilities, observation_probabilities, termination = False):
        self.initial_state_probabilities = initial_state_probabilities
        self.transition_probabilities = transition_probabilities
        self.observation_probabilities = observation_probabilities
        self.num_states = observation_probabilities.shape[0]
        self.observation_length = observation_probabilities.shape[1]
        self.viterbi_path = np.zeros(self.observation_length, dtype='int16')
        self.viterbi_probability = 0
        self.forward_probabilities = []
        self.backward_probabilities = []
        self.forward_ll = 0
        self.backward_ll = 0
    
    def viterbi(self):
        max_probs = np.zeros((self.num_states, self.observation_length))
        pointers = np.zeros((self.num_states, self.observation_length), dtype='int16')
        for s in range(self.num_states):
            max_probs[s, 0] = math.log(self.initial_state_probabilities[s]) + math.log(self.observation_probabilities[s, 0])
        for i in range(1, self.observation_length):
            for t in range(self.num_states):
                max_state = 0
                max_val = -np.inf
                for s in range(self.num_states):
                    temp = max_probs[s, i-1] + math.log(self.transition_probabilities[s, t]) + math.log(self.observation_probabilities[t, i])
                    if temp > max_val:
                        max_state = s
                        max_val = temp
                max_probs[t, i] = max_val
                pointers[t, i] = max_state
        max_state = 0
        max_val = -np.inf
        for t in range(self.num_states):
            if max_probs[t, self.observation_length - 1] > max_val:
                max_state = t
                max_val = max_probs[t, self.observation_length - 1]
        self.viterbi_log_probability = max_val

        #  Traceback
        for i in reversed(range(self.observation_length)):
            self.viterbi_path[i] = max_state
            max_state = pointers[max_state, i]
    
    def sum_logs(self, p, q):
        if p>9999 and q>99999:
            ans = math.log(math.exp(p) + math.exp(q))
        else:
            if p > q:
                ans =  p + math.log(1 + math.exp(q - p))
            else:
                ans =  q + math.log(1 + math.exp(p - q))
        return ans
    
    def forward(self):
        self.forward_probabilities = np.zeros((self.num_states, self.observation_length))
        for s in range(self.num_states):
            self.forward_probabilities[s, 0] = math.log(self.initial_state_probabilities[s]) + math.log(self.observation_probabilities[s, 0])
        for i in range(1, self.observation_length):
            for t in range(self.num_states):
                temp = 0
                for s in range(self.num_states):
                    if s == 0:
                        temp = math.log(self.transition_probabilities[s, t]) + self.forward_probabilities[s, i-1]
                    else:
                        temp = self.sum_logs(temp, math.log(self.transition_probabilities[s, t]) + self.forward_probabilities[s, i-1])
                self.forward_probabilities[t, i] = temp + math.log(self.observation_probabilities[t, i])
        temp = 0
        for t in range(self.num_states):
            if t == 0:
                temp = self.forward_probabilities[t, self.observation_length -1]
            else:
                temp = self.sum_logs(temp, self.forward_probabilities[t, self.observation_length -1])
        self.forward_ll = temp
        
    def backward(self):
        self.backward_probabilities = np.zeros((self.num_states, self.observation_length))
        for s in range(self.num_states):
            self.backward_probabilities[s, self.observation_length - 1] = 0 #math.log(self.observation_probabilities[s, self.observation_length - 1])
        for i in reversed(range(0, self.observation_length - 1)):
            for s in range(self.num_states):
                temp = 0
                for t in range(self.num_states):
                    if t == 0:
                        temp = self.backward_probabilities[t, i+1] + math.log(self.transition_probabilities[s, t]) + math.log(self.observation_probabilities[t, i+1])
                    else:
                        temp = self.sum_logs(temp, self.backward_probabilities[t, i+1] + math.log(self.transition_probabilities[s, t]) + math.log(self.observation_probabilities[t, i+1]))
                self.backward_probabilities[s, i] = temp
        temp = 0
        for t in range(self.num_states):
            if t == 0:
                temp = math.log(self.initial_state_probabilities[t]) + self.backward_probabilities[t, 0] + math.log(self.observation_probabilities[t,0])
            else:
                temp = self.sum_logs(temp, math.log(self.initial_state_probabilities[t]) + self.backward_probabilities[t, 0] + math.log(self.observation_probabilities[t,0]))
        self.backward_ll = temp

        
def cons_mutation_probs(params, alignment_list, alignment_names, num_symbols, sequence_name_dict, master_species_index):    
    num_states = 3
    align_list =  alignment_list
    len_align_list = len(alignment_list[0])
    num_sequences = len(alignment_list)
    observation_probs =  np.zeros((num_states, len_align_list))
    for i in range(len_align_list):
        for a_name in alignment_names:
            j = sequence_name_dict[a_name]
            if j == master_species_index:
                master_species_symbol = alignment_list[j][i]
        for s in range(num_states):
            ans = 1
            for a_name in alignment_names:
                j = sequence_name_dict[a_name]
                if j == master_species_index:
                    continue
                else:
                    aligned_symbol = alignment_list[j][i]
                    if aligned_symbol == master_species_symbol:
                        ans = ans * (params[s])
                    else:
                        ans = ans * (1-params[s])
            observation_probs[s, i] = ans
    return observation_probs

def mutation_probs(rates, alignment_list, alignment_names, master_tree, num_symbols):
    num_states = len(rates)
    align_list =  alignment_list
    len_align_list = len(alignment_list[0])
    num_sequences = len(alignment_list)
    observation_probs =  np.zeros((num_states, len_align_list))
    for i in range(len_align_list):
        temp = []
        temp.append([x for x in alignment_names])
        temp.append([x[i] for x in alignment_list])
        for j in range(num_states):
            observation_probs[j, i] = felsenstein_probability (temp, num_symbols, master_tree, rates[j]) 
    return observation_probs

def felsenstein_probability (state_list, num_symbols, master_tree, length_scalar):
    if num_symbols == 4:
        alphabet = ['A','C','G','T']
    else:
        alphabet = ['A','C','G','T','-']
    initial_states = {}
    prior_probabilities = [1/num_symbols] * num_symbols
    for i in range(len(state_list[0])):
        initial_states[state_list[0][i]] = alphabet.index(state_list[1][i])
    nodes_under_consideration = []
    info_dict = {}
    num_nodes = 0
    for node in master_tree.traverse():
        num_nodes+=1
        if node.is_leaf():
            nodes_under_consideration.append(node)
            temp_probs = []
            for s in range(num_symbols):
                if initial_states[node.name] == s:
                    temp_probs.append(1)
                else:
                    temp_probs.append(0)
            info_dict[node] = temp_probs
    while(len(nodes_under_consideration) < num_nodes):
        for n in nodes_under_consideration:
            if n.up in info_dict:
                continue
            sibling_group = [n]
            for p in n.get_sisters():
                sibling_group.append(p)
            num_not_in_dict = 0
            for x in sibling_group:
                if not(x in info_dict):
                    num_not_in_dict +=1
            if num_not_in_dict == 0:
                new_probs = []
                for s in range(num_symbols):
                    temp_prob = 1
                    for x in sibling_group:
                        branch_length = x.dist
                        probs = info_dict[x]
                        temp_prob_2 = 0
                        for t in range(num_symbols):
                            jc_prob = math.exp(-1.0*num_symbols*branch_length*length_scalar/(num_symbols -1))
                            if s == t:
                                transition_probability = 1.0/num_symbols + (num_symbols-1)/num_symbols * jc_prob
                            else:
                                transition_probability = 1.0/num_symbols - 1.0/num_symbols * jc_prob
                            temp_prob_2 += transition_probability * probs[t]
                        temp_prob = temp_prob * temp_prob_2
                    new_probs.append(temp_prob)
                info_dict[n.up] = new_probs
        nodes_under_consideration = list(info_dict.keys()) 
    for node in master_tree.traverse():
        if node.is_root():
            ans = 0
            probs = info_dict[node]
            for s in range(num_symbols):
                ans = ans + prior_probabilities[s] * probs[s] 
    return ans

def fit_phylo_hmm(tree, num_symbols, num_states, params, group_ids, align_dict, num_subsets, subset_num, offset, min_length):
    initial_state_probabilities = [1.0/num_states]*num_states
    total_probability = 0
    a = params[0]
    b = (1-params[0])
    c = 1 - (params[1])
    d = params[1]
    transition_probabilities = np.array([[a,b],[c,d]])
    ids = chunk_list(group_ids, num_subsets, subset_num)
    for group_id in ids:
        alignment = align_dict[group_id]
        align_list =  alignment.modified_sequence_list
        align_names = alignment.sequence_names
        len_align_list = len(align_list[0])
        non_cds = [x[offset:len_align_list - offset] for x in align_list]
        if len(non_cds[0]) < min_length:
            continue
        #observation_probabilities = mutation_probs(params[2:4], non_cds, align_names, tree, num_symbols)
        observation_probabilities = mutation_probs([params[2],params[3],params[3]], non_cds, align_names, tree, num_symbols)
        trial_hmm = HMM(initial_state_probabilities, transition_probabilities, observation_probabilities)
        #trial_hmm.viterbi()
        #total_probability += trial_hmm.viterbi_log_probability * -1
        trial_hmm.forward()
        total_probability += trial_hmm.forward_ll * -1
    return total_probability

def fit_cons_hmm(num_symbols, num_states, params, group_ids, align_dict, num_subsets, subset_num, offset, min_length, sequence_name_dict, master_species_index):
    initial_state_probabilities = [1.0/num_states]*num_states
    total_probability = 0
    a = params[0]
    b = (1-params[0])*(params[1])
    c = 1-a-b
    e = params[2]
    d = (1-params[2])*(params[3])
    f = 1-e-d
    i = params[4]
    g = (1-params[4])*(params[5])
    h = 1 - i - g
    transition_probabilities = np.array([[a,b,c],[d,e,f],[g,h,i]])
    ids = chunk_list(group_ids, num_subsets, subset_num)
    for group_id in ids:
        alignment = align_dict[group_id]
        align_list =  alignment.modified_sequence_list
        align_names = alignment.sequence_names
        len_align_list = len(align_list[0])
        non_cds = [x[offset:len_align_list - offset] for x in align_list]
        if len(non_cds[0]) < min_length:
            continue
        observation_probabilities = cons_mutation_probs(params[6:], non_cds, align_names, num_symbols, sequence_name_dict, master_species_index)
        trial_hmm = HMM(initial_state_probabilities, transition_probabilities, observation_probabilities)
        trial_hmm.forward()
        total_probability += trial_hmm.forward_ll * -1
    return total_probability