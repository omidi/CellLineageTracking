import os
import re
import numpy as np
from scipy.spatial import distance
from scipy.stats import norm
import pandas as pd

data_dir = '/Users/omidi/image_analysis'
output_files = [f for f in os.listdir(data_dir) if re.search('^output_\d+.csv$', f)]

traces = {}
for time in xrange(1, len(output_files)):
    fname1 = os.path.join(data_dir, "output_%d.csv" % time)
    fname2 = os.path.join(data_dir, "output_%d.csv" % (time+1))
    data1 = pd.read_csv(fname1, index_col=0)
    data2 = pd.read_csv(fname2, index_col=0)
    pos1 = data1[['x', 'y']]
    pos2 = data2[['x', 'y']]
    names1 = np.array(data1.index)
    names2 = np.array(data2.index)
    for n in names1:
        traces.setdefault(n, [])
    dst = np.transpose(distance.cdist(pos1, pos2))
    likelihood = norm.pdf(dst, scale=3)
    posterior = likelihood / np.sum(likelihood, axis=1)[:, None]
    indices = np.argmax(likelihood, axis=1)
    temp_traces = {}
    for order, ind in enumerate(indices):
        if posterior[order, ind] > .9:
            temp_traces.setdefault(ind, [])
            temp_traces[ind].append(order)
    for cell1, cell2 in temp_traces.items():
        if len(cell2) > 1:
            print cell2

    print temp_traces
    exit()
    


