"""
This script is for taking a json file with 2D detections in pixel space
the ground-truth in 2D determining the confidence threshold that results
in the highest similarity.
"""

import json
import numpy as np
from accuracy_metric import *

num_videos = 300
num_frames = 300

dict = json.load(open(f"vectorized_data_labelled_retinanet_1.json_with_gt.json"))
#parse for ground_truth
for batch in range(2, 7):
    f = open(f"vectorized_data_labelled_retinanet_{batch}.json_with_gt.json")
    dict_to_add = json.load(f)
    dict.extend(dict_to_add)

d = {62 : 1, 64 : 2, 72 : 3, 28 : 4, 51 : 5}

def fit_threshold_NN(dict, threshold):
    print(threshold)
    sims = np.zeros(num_videos)
    dists = np.zeros(num_videos)
    for v in range(num_videos):
        for f in range(num_frames):
            gt_labels = dict[v]["views"][f]["ground_truth"]["labels"]
            gt_labels = [d[label] for label in gt_labels]
            gt_pos = dict[v]["views"][f]["ground_truth"]["centers"]

            det_scores = dict[v]["views"][f]["detections"]["scores"]
            index = [i for i,val in enumerate(det_scores) if val > threshold]
            det_labels = [dict[v]["views"][f]["detections"]["labels"][i] for i in index]
            det_labels = [d[label] for label in det_labels]
            det_pos = [dict[v]["views"][f]["detections"]["center"][i] for i in index]

            result = Jaccard_similarity(gt_labels, gt_pos, det_labels, det_pos, True)
            sims[v] = sims[v] + result[0]
            dists[v] = dists[v] + result[1]
        sims[v] = sims[v]/num_frames
        dists[v] = dists[v]/num_frames
    return [np.mean(sims), np.nanmean(dists)]

# thresholds = np.linspace(0, 1, 101)
# results = list(map(fit_threshold_NN, [dict]*100, thresholds))
# how_good = results[0]
# #dist = results[1]
# print(how_good)
# best = max(how_good)
# max_index = how_good.index(best)
# print(thresholds[max_index])
# print(how_good[max_index])
# #print(dist[max_index])

results = fit_threshold_NN(dict, 0.0)
print(results)
