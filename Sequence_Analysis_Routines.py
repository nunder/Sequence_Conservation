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
    subprocess.run('wsl cd ~; source sonicparanoid/bin/activate; sonicparanoid -i ' + wslname(protein_file_location) +' -o ' + 
    wslname(output_location) + ' -p ' + run_name + ' -t 8' , shell=True)
    
    
#  Orthologue class

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
                orthologs_df.at[i,'protein_id'] = r['protein_id'].split(',')[0]
                orthologs_df.at[i,'num_protein_ids'] = len(r['protein_id'].split(','))
        self.orthologs_df = orthologs_df[orthologs_df['num_protein_ids']==1]   #  Remove species not in ortholog group or where more than one protein 
        temp_df = self.orthologs_df.groupby('group_id',as_index=False)['protein_id'].count()
        self.single_copy_ortholog_groups = temp_df['group_id'].tolist()
        self.full_ortholog_groups = temp_df[temp_df['protein_id'] == self.num_ids]['group_id'].tolist()

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
                   
    df = df[['name','type','locus_tag','previous_locus_tag','next_locus_tag','protein_id','bp_restrict','loc','non_cds_loc','upstream_non_cds_loc','ss_non_cds_loc','strand','start','end', 'cds_seq','non_cds_seq',
              'non_cds_offset_seq', 'upstream_non_cds_seq', 'non_cds_start','non_cds_end','upstream_non_cds_start','upstream_non_cds_end','ss_non_cds_seq','cds_length','non_cds_length','upstream_non_cds_length','ss_non_cds_length',
              'non_cds_offset_length','non_cds_offset_start','non_cds_offset_stop']]                                  
    return df
