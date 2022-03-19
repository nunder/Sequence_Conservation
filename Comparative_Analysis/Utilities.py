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
