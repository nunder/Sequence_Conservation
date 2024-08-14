import subprocess
import os
import re
import pickle
import pandas as pd
import math
import copy
from tqdm import tqdm
from collections import defaultdict
import ete3

from Bio import Entrez, SeqIO, AlignIO, pairwise2, Align, Seq, motifs
from joblib import Parallel, delayed
project_dir = '/d/projects/u/un001/Cryptic_Tree'
datasets_dir = project_dir + '/Datasets'
dictionary_dir = project_dir + '/Dictionaries'
mutation_count_dir = project_dir + '/Mutation_Counts'
chunk_variant_dict_dir = project_dir + '/Chunk_Variant_Dictionaries'
tb_genome_filename = 'GCF_000195955.2_ASM19595v2_genomic.gbff'
full_run = False
tb_tree_filename = 'tb.nwk'

num_cores = 32
core_numbers = list(range(0, num_cores))
timeout=99999

num_iterations = 30

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

def generate_variant_sequence_dict(chunk):
    if os.path.exists(chunk_variant_dict_dir + '/chunk_variant_dictionary_'+str(chunk) + '.pkl'):    #  Number of chunk dictionaries generated can be lower given rounding of chunk size        
        with open(chunk_variant_dict_dir + '/chunk_variant_dictionary_'+str(chunk) + '.pkl', 'rb') as f:
            chunk_variant_dict = pickle.load(f) 
        with open(project_dir + '/ids.pkl', 'rb') as f:
            distinct_sequence_names = pickle.load(f) 

        full_tb_variant_sequence = ''.join([full_sequence[pos] for pos in sorted_variant_position_list])
        start_pos = chunk * chunk_size
        end_pos = min(tb_variants_sequence_length, start_pos + chunk_size)
        variant_sequence_dict = {}
        seq_chunk = [{x} for x in full_tb_variant_sequence[start_pos:end_pos]]
        for seq_id in distinct_sequence_names:
            temp = copy.copy(seq_chunk)
            if seq_id in chunk_variant_dict:
                for (pos, snp) in chunk_variant_dict[seq_id]:
                    temp[pos] = {snp}
            variant_sequence_dict['seq_'+str(seq_id)] = temp
        with open(dictionary_dir + '/variant_dictionary_'+str(start_pos)+'_'+str(end_pos)+'_' + '.pkl', 'wb') as f:
            pickle.dump(variant_sequence_dict, f)

def fitch_1(list_1, list_2):
    res =[]
    for i, j in zip(list_1, list_2):
        a = i.intersection(j)
        if len(a) == 0:
            a = i.union(j)
        res.append(a)
    return res

def fitch_2(parent_list, child_list):
    res = []
    mutations = []
    for i, j in zip(parent_list, child_list):
        mutation = 0
        a = i.intersection(j)
        if len(a) == 0:
            a = set(list(j)[0])
            mutation = 1
        res.append(a)
        if mutation == 1:
            mutations.append(1)
        else:
            mutations.append(0)
    return (res, mutations)

def generate_mutation_counts(filename, core_number):    
    seq_length = 83    #25 * 83 is length of chunks output in previous step
    a = filename.split('_')
    start = a[-3]
    stop = a[-2]
    with open(filename, 'rb') as f:
        sequence_to_score_dict = pickle.load(f)
    master_tree2= ete3.Tree(project_dir + '/' + tb_tree_filename)
    for node in master_tree2.traverse("postorder"):
        if node.is_leaf():
            node.add_features(seq = sequence_to_score_dict[node.name][core_number * seq_length: (core_number+1) * seq_length])
        else:
            children = node.children
            node.add_features(seq = fitch_1(children[0].seq, children[1].seq))

    
    mutation_counts = [0 for i in range(seq_length)]
    for node in master_tree2.traverse("preorder"):
        if node.is_leaf():
            continue
        if node.is_root():
            node.seq = [{list(x)[0]} for x in node.seq]
        children = node.children
        mutations = []
        child_sequences = []
        for child in children:
            (temp_1, temp_2) = fitch_2(node.seq ,child.seq)
            child_sequences.append(temp_1)
            child.seq = temp_1
            mutations.append(temp_2)
        temp = []
        for n, (h, i, j) in enumerate(zip(mutation_counts, mutations[0], mutations[1])):
            if i + j == 0:
                temp.append(h+0)
            elif i + j == 1:
                temp.append(h+1)
            else:
                if child_sequences[0][i] == child_sequences[1][i]:
                    temp.append(h+1)
                else:
                    temp.append(h+2)
            
        mutation_counts = temp     
    return (start, stop, mutation_counts)

#  IMPORT REFERENCE SEQUENCE

for record in SeqIO.parse(datasets_dir + '/' + tb_genome_filename, "genbank"):
    full_sequence = str(record.seq)
tb_length = len(full_sequence)

#  LOAD VARIANT DICTIONARY AND IDS IN TREE (FILES PRODUCED IN TREE CREATION STEP - SEE F1_TREE_CREATION_SCRIPT)

if full_run == True:
    with open(project_dir + '/variant_dict.pkl', 'rb') as f:
        variant_dict = pickle.load(f) 
    with open(project_dir + '/ids.pkl', 'rb') as f:
        distinct_sequence_names = pickle.load(f) 

    variant_positions = []
    for k, v in variant_dict.items():
        if k in distinct_sequence_names:
            for (pos, snp) in v:
                variant_positions.append(pos-1)     #Cryptic is 1 indexed
    sorted_variant_position_list = list(set(variant_positions))
    sorted_variant_position_list.sort()
    pos_id_dict = dict(zip(sorted_variant_position_list, range(len(set(variant_positions)))))
    id_pos_dict = dict(zip(range(len(set(variant_positions))), sorted_variant_position_list))
    tb_variants_sequence_length = len(pos_id_dict)
    num_chunks = num_cores * num_iterations   
    chunk_size = math.ceil(tb_variants_sequence_length/num_chunks)
      
if full_run == True:
    chunk_variant_dict = defaultdict(lambda: defaultdict(list))
    for k, v in variant_dict.items():
        if k in distinct_sequence_names:
            for (pos, snp) in v:
                chunk = int(pos_id_dict[pos-1]/chunk_size)
                position_in_chunk = pos_id_dict[pos-1] % chunk_size
                chunk_variant_dict[chunk][k].append((position_in_chunk,snp.upper()))
    for chunk in tqdm(list(chunk_variant_dict.keys())):
        with open(chunk_variant_dict_dir + '/chunk_variant_dictionary_'+str(chunk) + '.pkl', 'wb') as f:
            pickle.dump(chunk_variant_dict[chunk], f)

    print("Built dictionaries")

if full_run == True:
    for core_1 in tqdm(range(num_iterations)):
        parallel_output = Parallel(n_jobs=-1, timeout = timeout)(delayed(generate_variant_sequence_dict)(core_1 * num_cores + core_2) for (core_2) in core_numbers)
        
if full_run==True:  
    res = []
    filename_list = list_files(dictionary_dir)
    for filename in tqdm(filename_list):
        temp_2 = filename.split('_')
        start_pos = int(temp_2[2])
        end_pos = int(temp_2[3])
        parallel_output = Parallel(n_jobs=-1)(delayed(generate_mutation_counts)(dictionary_dir+'/' + filename, core_number) for core_number in range(25))
        temp = []
        for x in parallel_output:
            temp+=x[2]
        res.append((int(parallel_output[0][0]), int(parallel_output[0][1]), temp))
        with open(mutation_count_dir + '/mutation_counts_'+str(start_pos)+'_'+str(end_pos)+'_' + '.pkl', 'wb') as f:
                   pickle.dump((int(parallel_output[0][0]), int(parallel_output[0][1]), temp), f) 
    res.sort(key = lambda x: x[0])    
    with open(mutation_count_dir + '/all_mutation_counts.pkl', 'wb') as f:
        pickle.dump(res, f) 

if full_run==True:  
    with open(mutation_count_dir + '/all_mutation_counts.pkl', 'rb') as f:
        res = pickle.load(f) 
    non_zero_mutation_counts = []
    for x in res:
        non_zero_mutation_counts += x[2]
    zero_and_non_zero_mutation_counts = []
    for i in range(len(full_sequence)):
        if i in pos_id_dict:
            zero_and_non_zero_mutation_counts.append(non_zero_mutation_counts[pos_id_dict[i]])
        else:
            zero_and_non_zero_mutation_counts.append(0)
    with open(mutation_count_dir + '/zero_and_non_zero_mutation_counts.pkl', 'wb') as f:
        pickle.dump(zero_and_non_zero_mutation_counts, f)  
    mutation_df = pd.DataFrame(zero_and_non_zero_mutation_counts, columns = ['Num_Mutations'])
    mutation_df.to_csv(project_dir + '/mutation_df.csv')   