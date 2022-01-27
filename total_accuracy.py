import json
import numpy as np
from accuracy_metric import *

# dict = json.load(open(f"vectorized_data_labelled_retinanet_1.json.json"))
# #parse for ground_truth
# for batch in range(2, 9):
#     f = open(f"vectorized_data_labelled_retinanet_{batch}.json.json")
#     dict_to_add = json.load(f)
#     dict.extend(dict_to_add)

path_to_inferences = "baseline_retinanet/baseline_retinanet/"
dict = json.load(open(f"{path_to_inferences}/with_inferences.json"))

p = 0.6 #p in geometric distibution

num_videos = 300

#label is a string, a semantic label. want to return the coco index number.
def label_to_num(label):
    #print(f"label {label}")
    dictionary = {"chair" : 62, "plant" : 64, "tv" : 72, "umbrella" : 28, "bowl" : 51} #going by detector outputs from cluster
    label2 = dictionary[label]
    dictionary2 = {62 : 1, 64 : 2, 72 : 3, 28 : 4, 51 : 5}
    return dictionary2[label2]

sims = np.zeros(num_videos)
dists = np.zeros(num_videos)
for v in range(num_videos): #find best solution for each scene
    best_nll = np.inf
    best_solution_labels = []
    best_solution_locations = []

    gt_labels = []
    gt_poses = []
    for i in range(len(dict[v]["labels"])): #got all the ground_truth
        gt_labels.append(label_to_num(dict[v]["labels"][i]["category_name"]))
        gt_poses.append(np.array(dict[v]["labels"][i]["position"]))

    #k = len(gt_labels) #oracle
    for k in range(1, 5): #should be 5
        prob_density = ((1 - p)**k) * p
        neg_log_prior = -np.log(prob_density)
        data = np.load(f'{path_to_inferences}/sigma1_scene{v}_objects{k}.npz')
        neg_log_post = neg_log_prior + data["nll"]
        if neg_log_post < best_nll:
            best_nll = neg_log_post
            best_solution_labels = np.delete(data["object_categories"], 0)#removing first label because it's always 0
            best_solution_locations = np.delete(data["object_locations"], 0, 0)#removing first row

    #print(best_nll)
    print(f"scene {v}")
    print(f"inferred labels{best_solution_labels}")
    print(f"inferred poses{best_solution_locations}")
    print(f"gt_labels{gt_labels}")
    print(f"gt_poses{gt_poses}")
    result = Jaccard_similarity(gt_labels, gt_poses, best_solution_labels, best_solution_locations, True)
    sims[v] = result[0]
    dists[v] = result[1]

#print(sims)
print(f"Jaccard_similarity {np.mean(sims)}")
print(f"Distances {np.nanmean(dists)}")
