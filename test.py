import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from Bio import Entrez, SeqIO, AlignIO, pairwise2, Align, Seq, motifs
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, FeatureLocation
from scipy.stats import binom
import math
from tqdm.auto import tqdm
from Comparative_Analysis import Sequence_Analysis_Routines as sar
from Comparative_Analysis import Utilities as util
from Comparative_Analysis import Alignment as align
import random
import copy
from joblib import Parallel, delayed
import os
import shutil
import subprocess
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.Align.Applications import MuscleCommandline
import re
import shutil

full_run = True
project_dir = 'D:/Project_Data/Project_8'
datasets_dir = project_dir + '/Datasets'
output_dir = project_dir + '/Output'
wsl_output_loc = util.wslname(output_dir)
seq_dir = 'D:/Actinobacteria_Ref_Rep_Lev_Complete'
blast_dir = 'D:/BLAST/actinobacteria_ref_rep_comp'
blast_db_name = 'actinobacteria_ref_rep_comp'
num_cores = 16
core_numbers = list(range(1, num_cores+1))
reference_species_filename = 'GCF_000195955.2_ASM19595v2_genomic.gbff'
species_list = util.list_files(seq_dir)
hmmer_evalue = 1e-10
infernal_evalue = 1e-10

output = []
features = []
genome_record = next(SeqIO.parse(seq_dir + '/' + reference_species_filename, "genbank"))
full_sequence = str(genome_record.seq)
mycobrowser_df = pd.read_excel(datasets_dir+'/Mycobrowser_Release_4.xlsx')
for i, r in mycobrowser_df.iterrows():
    if r['Strand'] == '+':
        strand = 1
    else:
        strand = -1
    features.append([r['Locus'],r['Start']-1, r['Stop'], strand])
features.sort(key=lambda x: x[1])
feature_info = []
for i, feature in enumerate(features):
    feature_sequence = full_sequence[feature[1]: feature[2]]
    #feature_info.append([feature[0], feature[1], feature[2], feature_sequence, len(feature_sequence)])
    if feature[1] < feature[2]:  
        if (i + 1)< len(features) and feature[2] < features[i+1][1]:
            utr_coords = (feature[2], features[i+1][1])
            utr_sequence = full_sequence[feature[2]: features[i+1][1]]
            utr_length = len(utr_sequence)
        else:
            utr_coords = (0,0)
            utr_sequence = ''
            utr_length = 0
        if utr_length > 50:
            feature_info.append([feature[0] + '_IG', utr_coords[0], utr_coords[1], utr_sequence, utr_length])
intergenic_df = pd.DataFrame(feature_info, columns = ['Locus', 'Start' , 'End', 'Sequence', 'Length'])

def align_keep_top_hit_per_species(output_dir, hit_file, alignment_file, output_alignment_file, package, evalue = 0.01): 
    wsl_output_loc = util.wslname(output_dir)
    dict = {}
    if package == 'INFERNAL':
        with open(output_dir + '/' + hit_file, 'r') as f:
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
        with open(output_dir + '/' + hit_file, 'r') as f:
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
    with open(output_dir + '/keep_list.txt', 'w') as f:
        lines = []
        for k, v in dict.items():
            lines.append(k + '/' + v[0] + "\n")
        f.write(''.join(lines))
    subprocess.run('wsl cd ' + wsl_output_loc + ' ; esl-alimanip -o '+output_alignment_file + ' --seq-k keep_list.txt '+ alignment_file, shell=True)

for i, r in intergenic_df.iterrows():
    sequence_list = [[r['Locus'], r['Sequence']]]
    print(r['Locus'])
    util.produce_fasta_file(sequence_list, output_dir + '/intergenic_region.faa')
    subprocess.run('wsl cd ' + wsl_output_loc + ' ; nhmmer -A align_'+ r['Locus'] +'.sto -o hmmer_' + r['Locus']+'.txt --tblout summary_'+r['Locus']+ '.txt --notextw --cpu 16 --incE 1e-10 intergenic_region.faa /mnt/d/Actinobacteria_Ref_Rep_Lev_Complete/all_actinobacteria_ref_rep_comp.faa', shell=True)
    for i in range(1, 5):
        align_keep_top_hit_per_species(output_dir, 'summary_'+r['Locus']+ '.txt', 'align_'+ r['Locus'] +'.sto', 'align_'+ r['Locus'] +'.sto', 'HMMER', hmmer_evalue)
        subprocess.run('wsl cd ' + wsl_output_loc + ' ; hmmbuild --cpu 16 hmm.hmm align_'+ r['Locus'] +'.sto', shell=True)
        subprocess.run('wsl cd ' + wsl_output_loc + ' ; nhmmer -A align_'+ r['Locus'] +'.sto -o hmmer.txt --tblout summary_'+r['Locus']+ '.txt --notextw --cpu 16 --incE ' + str(hmmer_evalue) +' hmm.hmm /mnt/d/Actinobacteria_Ref_Rep_Lev_Complete/all_actinobacteria_ref_rep_comp.faa', shell=True)

    align_keep_top_hit_per_species(output_dir, 'summary_'+r['Locus']+ '.txt', 'align_'+ r['Locus'] +'.sto', 'align_'+ r['Locus'] +'.sto', 'HMMER', hmmer_evalue)
    subprocess.run('wsl cd ' + wsl_output_loc + ' ; ~/rscape_v2.0.0.g/bin/R-scape  --cacofold --outname ' + r['Locus'] +' align_'+ r['Locus'] +'.sto ', shell=True)

    