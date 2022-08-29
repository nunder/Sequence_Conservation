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
import copy
import sys
import pickle
from . import Utilities as util

def build_blast_db(seq_dir, seq_filename, blast_db_name, blast_dir, db_type = 'prot'):
    w_d = os.getcwd()
    os.chdir("D:/")
    subprocess.run('cd '+ seq_dir + ' &  makeblastdb -in ' + seq_filename + ' -dbtype '+ db_type + ' -out ' + blast_db_name, shell=True, capture_output = True)
    os.chdir(w_d)
    files_to_move = [x for x in util.list_files(seq_dir) if x[:-4] == blast_db_name]
    for file in files_to_move:
        source = seq_dir + '/' + file
        destination = blast_dir + '/' + file
        shutil.move(source, destination)
        
def run_blastp(blast_dir, query_file, blast_db_name, e_value = 1e-10): 
    w_d = os.getcwd()
    os.chdir("D:/")
    subprocess.run('cd '+blast_dir+' & blastp -query '+blast_dir+'/'+query_file+' -db '+blast_db_name +' -out hits.csv -evalue '+str(e_value) +' -seg no -outfmt  "10 qaccver saccver qlen slen pident length mismatch gapopen qstart qend sstart send evalue bitscore" -num_threads 16', shell=True, capture_output = True)
    os.chdir(w_d)

def process_blast_output(infile_loc, outfile_loc, names_dict, top_hit_only = False):
    blast_results = pd.read_csv(infile_loc, header = None)
    blast_results.columns = ['query_ref', 'target_ref', 'query_length', 'subject_length', 'percent_identical_matches','alignment_length', 'number_mismatches', 'number_of_gap_openings', 'query_start_alignment', 'query_end_alignment', 'target_start_alignment', 'target_end_alignment', 'e_value', 'bit_score']
    for i, r in blast_results.iterrows():
        blast_results.at[i, 'query_species'] = r.query_ref.split('@')[0]
        blast_results.at[i, 'target_species'] = r.target_ref.split('@')[0]
       # blast_results.at[i, 'target_species'] = '_'.join(r.target_ref.split('_')[:-2])
    blast_results['query_species_name'] = blast_results['query_species'].map(names_dict)
    blast_results['target_species_name'] = blast_results['target_species'].map(names_dict)
    if top_hit_only == True:
        blast_results = blast_results.loc[blast_results.groupby(['query_ref','target_species'])['bit_score'].idxmax()]
    blast_results['species_count'] = blast_results.groupby('query_ref')['query_ref'].transform('size')
    with open(outfile_loc, 'wb') as f:
        pickle.dump(blast_results, f)
    return blast_results

def keep_reciprocal_best_hits(query_df, reverse_query_df, outfile_loc):
    temp_1_dict = {}
    temp_2_dict = {}
    for i, r in query_df.iterrows():
        temp_1_dict[r['query_ref']] = r['target_ref']
    for i, r in reverse_query_df.iterrows():
        temp_2_dict[r['query_ref']] = r['target_ref']
    for i, r in query_df.iterrows():
        if temp_1_dict[r['query_ref']] in temp_2_dict and temp_2_dict[temp_1_dict[r['query_ref']]] == r['query_ref']:
            query_df.at[i, 'reciprocal_best_hit'] = 'Y'
        else:
            query_df.at[i, 'reciprocal_best_hit'] = 'N'
    output = query_df[query_df.reciprocal_best_hit == 'Y'] 
    with open(outfile_loc, 'wb') as f:
        pickle.dump(output, f)
    return output

def align_keep_top_hit_per_species(files_dir, hit_file, alignment_file, output_alignment_file, package, evalue = 0.01): 
    wsl_files_loc = util.wslname(files_dir)
    dict = {}
    if package == 'INFERNAL':
        with open(files_dir + '/' + hit_file, 'r') as f:
            for l in f:
                if not(l[0] == '#'): 
                    a = l.split()
                    if a[16] == '!':
                        if a[0] in dict:
                            if float(a[15]) < dict[a[0]][1]:
                                dict[a[0]] = (a[7]+'-'+a[8],float(a[15]))
                        else:
                            dict[a[0]] = (a[7]+'-'+a[8],float(a[15]))
    elif package == 'HMMER':
        with open(files_dir + '/' + hit_file, 'r') as f:
            for l in f:
                if not(l[0] == '#'): 
                    a = l.split()
                    if float(a[12]) < evalue:
                        if a[0] in dict:
                            if float(a[12]) < dict[a[0]][1]:
                                dict[a[0]] = (a[6]+'-'+a[7],float(a[12]))
                        else:
                            dict[a[0]] = (a[6]+'-'+a[7],float(a[12]))
    else:
        pass
    with open(files_dir + '/keep_list.txt', 'w') as f:
        lines = []
        for k, v in dict.items():
            lines.append(k + '/' + v[0] + "\n")
        f.write(''.join(lines))
    subprocess.run('wsl cd ' + wsl_files_loc + ' ; esl-alimanip -o '+output_alignment_file + ' --seq-k keep_list.txt '+ alignment_file, shell=True)
    
    
def nhmmer_search_sequence(query_dir, query_file, target_dir, target_file, align_name, output_name, summary_name, e_value):
    wsl_query_dir = util.wslname(query_dir)
    wsl_target_dir = util.wslname(target_dir)
    subprocess.run('wsl cd ' + wsl_query_dir + ' ; nhmmer -A ' + align_name + ' -o ' + output_name + ' --tblout ' + summary_name + ' --notextw --cpu 16 --incE ' + str(e_value) +' '+ query_file + ' ' + wsl_target_dir+'/' + target_file, shell=True)
 
def hmmer_build(alignment_dir, alignment_file, model_name):
    wsl_alignment_dir = util.wslname(alignment_dir)
    subprocess.run('wsl cd ' + wsl_alignment_dir + ' ; hmmbuild --cpu 16 '+model_name+' '+alignment_file, shell=True)

def nhmmer_search_model(model_dir, model_file, target_dir, target_file, align_name, output_name, summary_name, e_value):
    wsl_model_dir = util.wslname(model_dir)
    wsl_target_dir = util.wslname(target_dir)
    subprocess.run('wsl cd ' + wsl_model_dir + ' ; nhmmer -A ' + align_name + ' -o ' + output_name + ' --tblout ' + summary_name + ' --notextw --cpu 16 --incE ' + str(e_value) +' '+ model_file + ' ' + wsl_target_dir+'/' + target_file, shell=True)
 
def run_rscape(alignment_dir, alignment_file, cacofold_suffix):
    wsl_alignment_dir = util.wslname(alignment_dir)
    subprocess.run('wsl cd ' + wsl_alignment_dir + ' ; ~/rscape_v2.0.0.g/bin/R-scape  --cacofold --outname ' + cacofold_suffix +' '+ alignment_file, shell=True)
    