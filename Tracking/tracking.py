import os
import re
import numpy as np
from scipy.spatial import distance
from scipy.stats import norm
import pandas as pd

def remove_old_cells_from_df(df, t, diff=10):
    # by default if a cell is not revisited for 10 time points
    # it will be treated as noisy prediction and hence will be
    # removed from the data frame for further considerations.
    cell_names = np.unique(np.array(df.index))
    indices = [i for i, n in enumerate(cell_names) if int(n.split("-")[0]) > (t - diff)]
    return df.iloc[indices]

data_dir = '/Users/omidi/image_analysis'
output_files = [f for f in os.listdir(data_dir) if re.search('^output_\d+.csv$', f)]

traces = {}
data = pd.DataFrame({'l':[], 'x':[], 'y':[], 't':[], 'intensity':[], 'area':[],
                     'orientation':[], 'solidity':[], 'major_axis_len':[],
                     'minor_axis_len':[], 'diameter':[], 'eccentricity':[]},
                      index=[])


sigma_in_likelihood = 5
# for time in xrange(1, len(output_files)):
for time in xrange(1, 16):
    fname1 = os.path.join(data_dir, "output_%d.csv" % time)
    fname2 = os.path.join(data_dir, "output_%d.csv" % (time+1))
    data1 = pd.concat([data, pd.read_csv(fname1, index_col=0)])
    data2 = pd.read_csv(fname2, index_col=0)
    pos1 = data1[['x', 'y']]
    pos2 = data2[['x', 'y']]
    names1 = np.array(data1.index)
    names2 = np.array(data2.index)
    dst = np.transpose(distance.cdist(pos1, pos2))
    likelihood = norm.pdf(dst, scale=sigma_in_likelihood)
    posterior = likelihood / np.sum(likelihood, axis=1)[:, None]
    posterior_t = np.transpose(likelihood) / np.sum(np.transpose(likelihood), axis=1)[:, None]
    indices = np.argmax(likelihood, axis=1)
    traces.setdefault(time, dict([(n, None) for n in names1]))
    cell_is_added = dict([(n, False) for n in names1])
    for order, ind in enumerate(indices):
        if not traces[time][names1[ind]]:
            traces[time][names1[ind]] = (names2[order], posterior_t[ind, order])
            cell_is_added[names1[ind]] = True
        elif posterior_t[ind, order] > traces[time][names1[ind]][1]:
            traces[time][names1[ind]] = (names2[order], posterior_t[ind, order])
            cell_is_added[names1[ind]] = True

    data = remove_old_cells_from_df(data, time, diff=10)
    data = data.append(data1.loc[[n for n, v in cell_is_added.items() if not v]])

print traces[17]['12-187']

exit()


visited_nodes = {}
allowed_num_gaps = 30
for time in xrange(1,25):
    for root in traces[time].keys():
        if root in visited_nodes:
            continue
        visited_nodes.setdefault(root, None)
        curr_cell = root
        vec = [(root, 1.)]
        n = 0  # counts the number of consequent None
        for t in xrange(1, 80):
            if (curr_cell in traces[t]) and traces[t][curr_cell]:
                vec.append(traces[t][curr_cell])
                curr_cell = traces[t][curr_cell][0]
                visited_nodes.setdefault(curr_cell, None)
                n = 0
            else:
                vec.append((None, 0))
                n += 1
            if n > allowed_num_gaps:   # if for more than 10 times the node wasn't updated, then ...
                if len(vec) > allowed_num_gaps + 10 or True:
                    n = 1
                break
        if n <= allowed_num_gaps:
            for x in xrange(len(vec)):
                if vec[x][0]:
                    root = vec[x][0].replace('-', '_')
                    break
            # fname = os.path.join('~/image_analysis', 'traces', root)
            fname = root + '.trace'
            outf = open(fname, 'wa')
            for cell in vec:
                if cell[0]:
                    time_point = cell[0].split('-')[0]
                    outf.write('\t'.join([
                                time_point,
                                cell[0],
                                '%0.6f\n' % cell[1],
                                ]))
            outf.flush()
            cmd = 'Rscript plot_traces.R %s' % fname
            # os.system(cmd)