from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


import matplotlib.pyplot as plt


fs = 20
plt.rc('font', size=fs) #controls default text size
plt.rc('axes', titlesize=fs) #fontsize of the title
plt.rc('axes', labelsize=fs) #fontsize of the x and y labels
plt.rc('xtick', labelsize=fs) #fontsize of the x tick labels
plt.rc('ytick', labelsize=fs) #fontsize of the y tick labels
plt.rc('legend', fontsize=fs) #fontsize of the legend

import numpy as np
import math
import sys
import scipy.io
from copy import deepcopy
import re


if __name__ == "__main__":
    
    filename = sys.argv[1]

    items = []

    with open(filename, 'r') as f:
        lines = f.readlines()
    
    for line in lines[16:]:
        if len(line)>10 and line[10]=='[':
            it = line.split('[')
            epoch = int(it[0])
            nums = it[1].replace(']','').split(',')
            nums = [float(item) for item in nums]
            loss = sum(nums)/len(nums)
            items.append([epoch, loss])
            # print(items)
    
    history = np.array(items)
    plt.figure(figsize=(8,5))
    plt.semilogy(history[:,0], history[:,1])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.tight_layout()
    plt.show()


