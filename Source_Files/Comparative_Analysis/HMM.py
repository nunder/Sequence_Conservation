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
import copy
    
class HMM:
    def __init__(self, initial_state_probabilities, transition_probabilities, observation_probabilities, termination = False):
        self.initial_state_probabilities = copy.deepcopy(initial_state_probabilities)
        self.transition_probabilities = copy.deepcopy(transition_probabilities)
        self.observation_probabilities = copy.deepcopy(observation_probabilities)
        self.num_states = len(initial_state_probabilities)
        self.observation_length = self.observation_probabilities.shape[1]
        self.viterbi_path = np.zeros(self.observation_length, dtype='int16')
        self.viterbi_probability = 0
        self.forward_probabilities = []
        self.backward_probabilities = []
        self.state_probabilities = []
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
            self.backward_probabilities[s, self.observation_length - 1] = 0 
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
    
    def calculate_state_probabilities(self):
        for s in range(self.num_states):
            c = self.forward_probabilities[s] + self.backward_probabilities[s]
            self.state_probabilities.append([math.exp(x - self.forward_ll) for x in c])
    
                  
    def calculate_probabilities(self):
        self.viterbi(); self.forward(); self.backward(); self.calculate_state_probabilities(); 


