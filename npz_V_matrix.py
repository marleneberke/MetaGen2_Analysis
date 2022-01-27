"""
This file is for taking the outputs of MetaGen as .npzs and a ground_truth
V matrix and calculating the L1 difference and printing it to a .csv file.
"""

import numpy as np

NN = "retinanet"
path_to_inferences = "learning_v/learning_v/"

num_minibatches = 5

num_categories = 5

gt_v_matrix = np.loadtxt(f"{NN}_results/gt_V_{NN}.txt", dtype = float)
#gt_v_matrix = np.concatenate((np.zeros((6,1), dtype = float), gt_v_matrix), axis = 1)

mses = np.zeros(num_minibatches+1)
for m in range(num_minibatches+1):
    if m == 0: #use initialization from prior
        v_matrix = np.zeros(shape = (6,6), dtype = float)
        v_matrix[0, 1:6] = 1/5 #hallucination rates
        v_matrix[1,1:6] = 1/14
        v_matrix[2,1:6] = 1/14
        v_matrix[3,1:6] = 1/14
        v_matrix[4,1:6] = 1/14
        v_matrix[5,1:6] = 1/14
        v_matrix[1,1] = 10/14
        v_matrix[2,2] = 10/14
        v_matrix[3,3] = 10/14
        v_matrix[4,4] = 10/14
        v_matrix[5,5] = 10/14
    else:
        data = np.load(f'{path_to_inferences}/batch{m}.npz')
        v_matrix = data["v"]
    #alphas = data["alphas"]
    np.set_printoptions(precision=5, suppress = True)
    #print(alphas)
    #remove v_matrix's column of zeros
    v_matrix = np.delete(v_matrix,0,1)
    print(f"gt_v_matrix {gt_v_matrix}")
    print(f"v_matrix  batch{m} {v_matrix}")
    se = (gt_v_matrix - v_matrix)*(gt_v_matrix - v_matrix) #Squared error
    mses[m-1] = np.mean(se)

np.set_printoptions(precision=6, suppress = True)
print(mses)
################################################################################

l = [minibatch, shuffle, mses]
data = zip(*l)

################################################################################

#shuffle_number is 0,1,2,3
header = ["minibatch_number_order", "shuffle_number", "mse"]
#header = ["video", "model", "Jaccard_3D", "dist_3D"]
with open(f"{NN}_V_dist.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer. writerows(data)
