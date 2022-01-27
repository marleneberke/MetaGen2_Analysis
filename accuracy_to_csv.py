"""
This script is for taking a json file with 2D detections in pixel space
and inferences in 3D space and the
camera position and the ground-truth objects in 2D and 3D space and calculating
accuracy for each video and printing it to a .csv file
"""
import csv
import json
import numpy as np
from accuracy_metric import *

################################################################################
num_videos = 4
num_frames = 20

#load files
path = "baseline_retinanet/baseline_retinanet/"
file = "with_inferences.json"
f = open(f"{path}{file}")
baseline_dict = json.load(f)

#label is a string, a semantic label. want to return the coco index number.
def label_to_num(label):
    dictionary = {"chair" : 62, "plant" : 64, "tv" : 72, "umbrella" : 28, "bowl" : 51} #going by detector outputs from cluster
    temp = dictionary[label]
    dictionary2 = {62 : 1, 64 : 2, 72 : 3, 28 : 4, 51 : 5}
    return dictionary2[temp]

def label_to_num2(label):
    dictionary2 = {62 : 1, 64 : 2, 72 : 3, 28 : 4, 51 : 5}
    return dictionary2[label]

################################################################################
#MetaGen model. will work for baseline and others
def process_MetaGen_3D(dict):
    sims = np.zeros(num_videos)
    dists = np.zeros(num_videos)
    for v in range(num_videos):
        gt_labels = []
        gt_poses = []
        for i in range(len(dict[v]["labels"])): #got all the ground_truth
            gt_labels.append(label_to_num(dict[v]["labels"][i]["category_name"]))
            gt_poses.append(np.array(dict[v]["labels"][i]["position"]))

        mg_labels = []
        mg_poses = []
        for i in range(len(dict[v]["metagen_inferences"]["labels"])):
            mg_labels.append(dict[v]["metagen_inferences"]["labels"][i])
            mg_poses.append(np.array(dict[v]["metagen_inferences"]["centers"][i]))

        result = Jaccard_similarity(gt_labels, gt_poses, mg_labels, mg_poses, False)
        sims[v] = result[0]
        dists[v] = result[1]
    return [sims, dists]

#MetaGen model. will work for baseline and others
def process_MetaGen_2D(dict):
    sims = np.zeros(num_videos)
    dists = np.zeros(num_videos)
    for v in range(num_videos):
        sims_frame = np.zeros(num_frames)
        dists_frame = np.zeros(num_frames)
        for f in range(num_frames):
            gt_labels = []
            gt_poses = []
            for i in range(len(dict[v]["views"][f]["ground_truth"]["labels"])): #got all the ground_truth
                gt_labels.append(label_to_num2(dict[v]["views"][f]["ground_truth"]["labels"][i]))
                gt_poses.append(np.array(dict[v]["views"][f]["ground_truth"]["centers"][i]))

            mg_labels = []
            mg_poses = []
            for i in range(len(dict[v]["views"][f]["metagen_inferences"]["labels"])):
                mg_labels.append(label_to_num2(dict[v]["views"][f]["metagen_inferences"]["labels"][i]))
                mg_poses.append(np.array(dict[v]["views"][f]["metagen_inferences"]["centers"][i]))

            result = Jaccard_similarity(gt_labels, gt_poses, mg_labels, mg_poses, False)
            sims_frame[f] = result[0]
            dists_frame[f] = result[1]
        sims[v] = np.mean(sims_frame)
        dists[v] = np.nanmean(dists_frame)
    return [sims, dists]

#NN
def process_NN_2D(dict, threshold = 0.):
    sims = np.zeros(num_videos)
    dists = np.zeros(num_videos)
    for v in range(num_videos):
        sims_frame = np.zeros(num_frames)
        dists_frame = np.zeros(num_frames)
        for f in range(num_frames):
            gt_labels = []
            gt_poses = []
            for i in range(len(dict[v]["views"][f]["ground_truth"]["labels"])): #got all the ground_truth
                gt_labels.append(label_to_num2(dict[v]["views"][f]["ground_truth"]["labels"][i]))
                gt_poses.append(np.array(dict[v]["views"][f]["ground_truth"]["centers"][i]))

            det_labels = []
            det_poses = []
            for i in range(len(dict[v]["views"][f]["detections"]["labels"])):
                det_labels.append(label_to_num2(dict[v]["views"][f]["detections"]["labels"][i]))
                det_poses.append(np.array(dict[v]["views"][f]["detections"]["center"][i]))

            result = Jaccard_similarity(gt_labels, gt_poses, det_labels, det_poses, False)
            sims_frame[f] = result[0]
            dists_frame[f] = result[1]
        sims[v] = np.mean(sims_frame)
        dists[v] = np.nanmean(dists_frame)
    return [sims, dists]




################################################################################
video = np.linspace(0, 1, num_videos+1)
model = ["baseline_metagen"]*num_videos

result = process_MetaGen_3D(baseline_dict)
Jac_3D = result[0]
dist_3D = result[1]

result = process_MetaGen_2D(baseline_dict)
Jac_2D = result[0]
dist_2D = result[1]

l = [video, model, Jac_3D, dist_3D, Jac_2D, dist_2D]
baseline_data = zip(*l)

video = np.linspace(0, 1, num_videos+1)
model = ["NN"]*num_videos
result = process_NN_2D(baseline_dict)
Jac_3D = [float("NaN")]*num_videos
dist_3D = [float("NaN")]*num_videos
Jac_2D = result[0]
dist_2D = result[1]

l = [video, model, Jac_3D, dist_3D, Jac_2D, dist_2D]
NN_data = zip(*l)

threshold = 0. #put fitted threshold here
video = np.linspace(0, 1, num_videos+1)
model = ["fitted_NN"]*num_videos
result = process_NN_2D(baseline_dict, threshold)
Jac_3D = [float("NaN")]*num_videos
dist_3D = [float("NaN")]*num_videos
Jac_2D = result[0]
dist_2D = result[1]

l = [video, model, Jac_3D, dist_3D, Jac_2D, dist_2D]
fitted_NN_data = zip(*l)

################################################################################


header = ["video", "model", "Jaccard_3D", "dist_3D", "Jaccard_2D", "dist_2D"]
#header = ["video", "model", "Jaccard_3D", "dist_3D"]
with open('accuracy.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    writer. writerows(baseline_data)
    writer. writerows(NN_data)
    writer. writerows(fitted_NN_data)
