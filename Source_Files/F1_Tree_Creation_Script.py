import subprocess
import os
import re
import pickle
import pandas as pd
import math
from tqdm import tqdm

from Bio import Entrez, SeqIO, AlignIO, pairwise2, Align, Seq, motifs
from joblib import Parallel, delayed
project_dir = '/d/projects/u/un001/Cryptic_Tree_GPI'
datasets_dir = project_dir + '/Datasets'
tb_genome_filename = 'GCF_000195955.2_ASM19595v2_genomic.gbff'
full_run = False

num_cores = 32
core_numbers = list(range(1, num_cores+1))
timeout=99999

#  IMPORT REFERENCE SEQUENCE

for record in SeqIO.parse(datasets_dir + '/' + tb_genome_filename, "genbank"):
    full_sequence = str(record.seq)
tb_length = len(full_sequence)

#  CREATE VARIANT DATASETS FROM CRYPTIC RAW DATA

if full_run == True:
    variant_df = pd.read_csv(datasets_dir + "/VARIANTS.csv") 
    with open(project_dir + '/variant_df.pkl', 'wb') as f:
        pickle.dump(variant_df[['UNIQUEID', 'VARIANT', 'MUTATION_TYPE', 'IS_NULL', 'IS_HET', 'IS_FILTER_PASS', 'IS_SNP', 'REF', 'ALT', 'GENOME_INDEX']], f)    

if full_run == True:
    with open(project_dir + '/variant_df.pkl', 'rb') as f:
        full_variant_df = pickle.load(f) 
    print(len(full_variant_df))
    genomes_df = pd.read_csv(datasets_dir + '/GENOMES.csv')
    gpi_genomes_df = genomes_df[genomes_df['BELONGS_GPI']==True][['UNIQUEID']] 
    print(len(gpi_genomes_df))
    gpi_variant_df = pd.merge(full_variant_df, gpi_genomes_df, how='inner', on = 'UNIQUEID')
    print(len(gpi_variant_df))
    with open(project_dir + '/gpi_variant_df.pkl', 'wb') as f:
        pickle.dump(gpi_variant_df, f)    

if full_run == True:
    position_dict = {}
    variant_dict = {}
    id_dict = {}
    with open(project_dir + '/gpi_variant_df.pkl', 'rb') as f:
        variant_df = pickle.load(f) 
        unique_ids = variant_df.UNIQUEID.unique()
        for i, unique_id in enumerate(unique_ids):
            id_dict[unique_id] = i

        for i, r in variant_df.iterrows():
            if r['IS_NULL'] == False and r['IS_FILTER_PASS'] == True and r['IS_HET'] == False and r['IS_SNP'] == True :
                
                if id_dict[r['UNIQUEID']] in variant_dict:
                    variant_dict[id_dict[r['UNIQUEID']]].append((r['GENOME_INDEX'], r['ALT']))
                else:
                    variant_dict[id_dict[r['UNIQUEID']]] = [(r['GENOME_INDEX'], r['ALT'])]

                if r['GENOME_INDEX'] in position_dict:
                    position_dict[r['GENOME_INDEX']].append((id_dict[r['UNIQUEID']], r['ALT']))
                else:
                    position_dict[r['GENOME_INDEX']] = [r['REF'], (id_dict[r['UNIQUEID']], r['ALT'])]    # If first entry also include reference value for info

    with open(project_dir + '/id_dict.pkl', 'wb') as f:
        pickle.dump(id_dict, f)
    with open(project_dir + '/variant_dict.pkl', 'wb') as f:
        pickle.dump(variant_dict, f) 
    with open(project_dir + '/position_dict.pkl', 'wb') as f:
        pickle.dump(position_dict, f) 

#  ALTERNATIVE VERSION - CORRECTING ASSUMING COMPLEMENTS WHEN REF IS COMPLEMENT OF ACTUAL REFERENCE SEQUENCE ENTRY

if full_run == True:
    complement_dict = {'a':'t', 'c':'g', 'g':'c', 't':'a'}
    position_dict = {}
    variant_dict = {}
    id_dict = {}
    with open(project_dir + '/gpi_variant_df.pkl', 'rb') as f:
        variant_df = pickle.load(f) 
        unique_ids = variant_df.UNIQUEID.unique()
        for i, unique_id in enumerate(unique_ids):
            id_dict[unique_id] = i

        for i, r in variant_df.iterrows():
            if r['IS_NULL'] == False and r['IS_FILTER_PASS'] == True and r['IS_HET'] == False and r['IS_SNP'] == True :
                genome_index = r['GENOME_INDEX']
                unique_id = r['UNIQUEID']
                ref_sequence_entry = full_sequence[genome_index - 1].lower()   # Cryptic is 1 indexed and lower case
                cryptic_ref_sequence_entry = r['REF']
                cryptic_alt_sequence_entry = r['ALT']
                if cryptic_ref_sequence_entry == ref_sequence_entry:
                    alt = cryptic_alt_sequence_entry
                elif complement_dict[cryptic_ref_sequence_entry] == ref_sequence_entry:
                    alt = complement_dict[cryptic_alt_sequence_entry]
                else:
                    print("Something strange at position ", genome_index, cryptic_ref_sequence_entry, ref_sequence_entry)
                    alt = cryptic_alt_sequence_entry
                
                id_ref = id_dict[unique_id]
                if id_ref in variant_dict:
                    variant_dict[id_ref].append((genome_index, alt))
                else:
                    variant_dict[id_ref] = [(genome_index, alt)]

                if genome_index in position_dict:
                    position_dict[genome_index].append((id_ref, alt))
                else:
                    position_dict[genome_index] = [ref_sequence_entry, (id_ref, alt)]    # If first entry also include reference value for info

    with open(project_dir + '/id_dict.pkl', 'wb') as f:
        pickle.dump(id_dict, f)
    with open(project_dir + '/variant_dict.pkl', 'wb') as f:
        pickle.dump(variant_dict, f) 
    with open(project_dir + '/position_dict.pkl', 'wb') as f:
        pickle.dump(position_dict, f) 



# CONSTRUCT DISTANCE MATRIX

def generate_distances(ind_1, ind_2):
    with open(project_dir + '/snp_pos_dict_'+str(ind_1)+'.pkl', 'rb') as f:
        snp_pos_dict_1 = pickle.load(f)
    with open(project_dir + '/snp_pos_dict_'+str(ind_2)+'.pkl', 'rb') as f:
        snp_pos_dict_2 = pickle.load(f)
    distance_dict = {}
    for k1, v1 in snp_pos_dict_1.items():
        for k2, v2 in snp_pos_dict_2.items():
            if k2 >= k1:
                continue
            sd = v1.symmetric_difference(v2)
            temp = {x[:-1] for x in sd}   # Only count variants with different nt in SNP as one mutation
            # Apply Jukes Cantor transformation if required  #
            #f = len(temp)/tb_length
            #d = -3/4 * math.log(1 - 4*f /3)
            #distance_dict[(k1, k2)] = d
            distance_dict[(k1, k2)] = len(temp)
    return(distance_dict)

if full_run == True:
    with open(project_dir + '/variant_dict.pkl', 'rb') as f:
        temp_variant_dict = pickle.load(f) 
    print(len(temp_variant_dict))
    temp_dict = {}
    for (k, v) in temp_variant_dict.items():
        v.sort(key = lambda x: x[0])
        keystring = ''.join([str(pos) + snp for (pos, snp) in v])
        temp_dict[keystring] = k
    variant_dict = {}
    for (k, v) in temp_dict.items():
    	variant_dict[v] = temp_variant_dict[v] 
    print(len(variant_dict))
    temp_variant_dict = {} 
    for core in tqdm(core_numbers):
        snp_pos_dict = {}
        for n, (k, v) in enumerate(variant_dict.items()):
            if k%num_cores + 1 == core:
                snp_pos_dict[k] = set([str(pos) + snp for (pos, snp) in v])
        with open(project_dir + '/snp_pos_dict_'+str(core)+'.pkl', 'wb') as f:
            pickle.dump(snp_pos_dict, f) 
    variant_dict = {}


if full_run == True:
    for core_1 in tqdm(core_numbers):
        master_dict = {}
        parallel_output = Parallel(n_jobs=-1, timeout = timeout)(delayed(generate_distances)(core_1, core_2) for (core_2) in core_numbers)
        for output_dict in parallel_output:
            for (k, v) in output_dict.items():
                master_dict[k] = v
        with open(project_dir + '/master_distance_dict_'+str(core_1)+'.pkl', 'wb') as f:
            pickle.dump(master_dict, f) 

# COMBINE DICTIONARIES GENERATED INTO SINGLE DICTIONARY

if full_run == True:
#if 1==1:
    master_dict = {}
    for core_1 in tqdm(core_numbers):
        with open(project_dir + '/master_distance_dict_'+str(core_1)+'.pkl', 'rb') as f:
            temp_master_dict = pickle.load(f) 
        k1_vals = []
        k2_vals = []
        for (k1, k2), v in temp_master_dict.items():
            k1_vals.append(k1)
            k2_vals.append(k2)
        k1_vals = list(set(k1_vals))
        k2_vals = list(set(k2_vals))
        k1_vals.sort()
        k2_vals.sort()

        for k1 in k1_vals:
            for k2 in k2_vals:
                if k2 >= k1:
                    continue
                if k1 in master_dict:
                    master_dict[k1].append(str(abs(temp_master_dict[(k1, k2)])))
                else:
                    master_dict[k1] = [str(abs(temp_master_dict[(k1, k2)]))]
    ids = []
    for k, v in master_dict.items():     # Note that seq_0 does not appear on LHS of distance matrix and therefore needs to be included separately in final output
        ids.append(k)
    ids.append(0)
    ids = list(set(ids))   
    ids.sort()


    #FINAL OUTPUT FOR INPUT TO QUICKTREE
    print("Writing data")
    #with open(project_dir + '/master_distance_dict.pkl', 'wb') as f:
    #    pickle.dump(master_dict, f) 
    with open(project_dir + '/ids.pkl', 'wb') as f:
        pickle.dump(ids, f)
    print("Writing Phylip file") 
    with open(project_dir+'/tb_seq_distances.phy', 'w') as f:
        f.write('%d\n' % len(ids))
        for n,idref in enumerate(ids):
            f.write('seq_'+str(idref))
            if idref in master_dict:
                for location in master_dict[idref]:
                    f.write('\t%s' % location)
            f.write('\n')
           
