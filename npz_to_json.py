"""
This file is for taking the outputs of MetaGen as .npzs and writing it to
a json file called with_inferences.json
"""
import json
import numpy as np
from accuracy_metric import *

dict = json.load(open(f"vectorized_data_labelled_retinanet_1.json.json"))
#parse for ground_truth
for batch in range(2, 9):
    f = open(f"vectorized_data_labelled_retinanet_{batch}.json.json")
    dict_to_add = json.load(f)
    dict.extend(dict_to_add)

path_to_inferences = "baseline_retinanet/baseline_retinanet/"

p = 0.6 #p in geometric distibution

num_videos = 300

#label is a string, a semantic label. want to return the coco index number.
def label_to_num(label):
    #print(f"label {label}")
    dictionary = {"chair" : 62, "plant" : 64, "tv" : 72, "umbrella" : 28, "bowl" : 51} #going by detector outputs from cluster
    label2 = dictionary[label]
    dictionary2 = {62 : 1, 64 : 2, 72 : 3, 28 : 4, 51 : 5}
    return dictionary2[label2]

for v in range(num_videos): #find best solution for each scene
    best_nll = np.inf
    best_solution_labels = []
    best_solution_locations = []

    for k in range(1, 5): #should be 5
        prob_density = ((1 - p)**k) * p
        neg_log_prior = -np.log(prob_density)
        data = np.load(f'{path_to_inferences}/sigma1_scene{v}_objects{k}.npz')
        neg_log_post = neg_log_prior + data["nll"]
        if neg_log_post < best_nll:
            best_nll = neg_log_post
            best_solution_labels = np.delete(data["object_categories"], 0)#removing first label because it's always 0
            best_solution_locations = np.delete(data["object_locations"], 0, 0)#removing first row

    #write the best solution to the json
    intermediate = {"centers" : best_solution_locations.tolist(), "labels" : best_solution_labels.tolist()}
    dict[v]["metagen_inferences"] = intermediate


with open(f"{path_to_inferences}/with_inferences.json", "w") as f:
    #print(f"{type(dict)}")
    json.dump(dict, f)
