import os
import pandas as pd
import subprocess
import seaborn as sns
import shutil
from tqdm import tqdm
import numpy as np
import seaborn as sns
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

class Ortholog_Grouping:
    def __init__(self, file_loc):
        orthologs_df = pd.read_csv(file_loc +'/flat.ortholog_groups.tsv', sep="\t", header=0)
        self.unassigned_genes_dict = {}
        with open(file_loc +'/not_assigned_genes.ortholog_groups.tsv','r') as input_file:
            first_seq = 0
            for l in input_file:
                m = l.strip('\n')
                if len(m) == 0:
                    continue
                if m[0] == '#':
                    if first_seq == 1:
                        self.unassigned_genes_dict[species_name] = temp 
                    species_name = m[1:-4]
                    temp = []
                    first_seq = 1
                else:
                    temp.append(m)
            self.unassigned_genes_dict[species_name] = temp 
        pd.read_csv(file_loc +'/not_assigned_genes.ortholog_groups.tsv', sep="\t", header=0)
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
        self.unassigned_genes_dict = ortholog_grouping.unassigned_genes_dict
        organism_names = self.sequence_data[['species','name']].drop_duplicates().reset_index(drop=True)
        self.organism_dict = {}
        for i, r in organism_names.iterrows():
            self.organism_dict[r['species']] = r['name'].split(' ')[0][0] + '.'+r['name'].split(' ')[1]
        
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
                orthologs_for_id = ortholog_grouping.all_copy_orthologs_df[ortholog_grouping.all_copy_orthologs_df['species'] == id]
                orthologs_and_cds_info = orthologs_for_id.merge(cds_data, how = 'left', left_on = 'protein_id', right_on = 'protein_id')
                
            orthologs_and_cds_info = orthologs_and_cds_info[orthologs_and_cds_info['name']==orthologs_and_cds_info['name']]    #Remove proteins which can't be matched back
            df_list.append(orthologs_and_cds_info)
        return df_list

    def generate_synteny_plot(self):
        a = self.sequence_data
        b = self.sequence_data[self.sequence_data['species'] == self.master_species]
        c = a.merge(b, how = 'inner', left_on = 'group_id', right_on = 'group_id')[['name_x','name_y','strand_x','strand_y','group_id','start_x','start_y']]
        g = sns.FacetGrid(c, col='name_x', col_wrap=4)
        g.map_dataframe(sns.scatterplot, x='start_x',y='start_y',s=2, color=".2",legend=True)  #marker='+',
        g.set_titles(row_template = '{row_name}', col_template = '{col_name}')
        #g.savefig(output_loc + "Ortholog_Syteny_Graph.pdf", dpi=300)
        
    def generate_ortholog_count_plot(self):
        a = self.sequence_data
        master_groups = a[a['species'] == self.master_species]['group_id'].unique().tolist()
        summary = a[a['group_id'].isin(master_groups)].groupby('group_id',as_index=False)['species'].count()
        sns.histplot(data=summary, x='species', bins=80)
    
    def generate_master_count_plot(self):
        a = self.sequence_data
        master_groups = a[a['species'] == self.master_species]['group_id'].unique().tolist()
        temp = a[a['group_id'].isin(master_groups)]
        summary = temp[temp['species'] == self.master_species].groupby('group_id',as_index=False)['species'].count()
        sns.histplot(data=summary, x='species', bins=80)
    
    def generate_unassigned_gene_count_plot(self):
        temp = []
        for k, v in self.unassigned_genes_dict.items():
            temp.append([self.organism_dict[k], len(v)])
            unassigned_gene_counts = pd.DataFrame(temp, columns=['species','gene_count'])
            sns.barplot(x='species', y='gene_count', data=unassigned_gene_count)