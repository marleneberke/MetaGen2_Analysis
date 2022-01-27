"""
This file is for taking the outputs of MetaGen as .npzs and a ground_truth
V matrix and calculating the L1 difference and printing it to a .csv file.
"""

import numpy as np

NN = "retinanet"

#num_minibatches = 5

num_categories = 5

gt_v_matrix = np.loadtxt("gt_V.txt", dtype = float)

diffs = np.zeros(num_minibatches)
for m in range(num_minibatches):
    data = np.load(f'{path_to_inferences}/something{m}.npz')
    v_matrix = data["v_matrix"]
    diff = abs(gt_v_matrix - v_matrix)
    diffs[m] = np.mean(diff)


################################################################################

l = [minibatch, shuffle, diff]
data = zip(*l)

################################################################################

#shuffle_number is 0,1,2,3
header = ["minibatch_number_order", "shuffle_number", "diff"]
#header = ["video", "model", "Jaccard_3D", "dist_3D"]
with open(f"{NN}_V_dist.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer. writerows(data)
