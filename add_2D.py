"""
This script is for taking a json file with 2D detections in pixel space and the
camera position and the ground-truth objects and 3D locations and calculating
the ground-truth labels in 2D and adding it to the json.

It will write the new file in the same place as the old one.
"""
import json
import numpy as np
import math
import geometry as geometry
import sys

num_videos = 50
num_frames = 300

path = "baseline_retinanet/baseline_retinanet/"
file = "with_inferences.json" #will print to this same file location
f = open(f"{path}{file}")
dict = json.load(f)

#option for taking inferred objects in 3D space and projecting them to 2D and writing it to the json file
inferred = int(sys.argv[1])

#label is a string, a semantic label. want to return the coco index number.
def label_to_num(label):
    #print(f"label {label}")
    dictionary = {"chair" : 62, "plant" : 64, "tv" : 72, "umbrella" : 28, "bowl" : 51} #going by detector outputs from cluster
    return dictionary[label]

def label_to_num2(label):
    dictionary2 = {1 : 62, 2 : 64, 3 : 72, 4 : 28, 5 : 51}
    return dictionary2[label]

for v in range(num_videos):
    #print(f"v {v}")
    for f in range(num_frames):
        #camera postion
        c_x = dict[v]["views"][f]["camera"]["x"]
        c_y = dict[v]["views"][f]["camera"]["y"]
        c_z = dict[v]["views"][f]["camera"]["z"]
        camera_pos = np.array([c_x, c_y, c_z])
        #focus
        f_x = dict[v]["views"][f]["lookat"]["x"]
        f_y = dict[v]["views"][f]["lookat"]["y"]
        f_z = dict[v]["views"][f]["lookat"]["z"]
        camera_focus = np.array([f_x, f_y, f_z])

        center = []
        labels = []
        if inferred == 0: #doing this for ground_truth
            for i in range(len(dict[v]["labels"])):
                label = label_to_num(dict[v]["labels"][i]["category_name"])
                obj_pos = np.array(dict[v]["labels"][i]["position"])
                obj_2d = geometry.get_image_xy(camera_pos, camera_focus, obj_pos)
                if geometry.within_frame(obj_2d):
                    center.append(obj_2d.tolist()) #to allow writing to json
                    labels.append(label)
        else: #for inferences
            for i in range(len(dict[v]["metagen_inferences"]["labels"])):
                label = label_to_num2(dict[v]["metagen_inferences"]["labels"][i])
                obj_pos = np.array(dict[v]["metagen_inferences"]["centers"][i])
                obj_2d = geometry.get_image_xy(camera_pos, camera_focus, obj_pos)
                if geometry.within_frame(obj_2d):
                    center.append(obj_2d.tolist()) #to allow writing to json
                    labels.append(label)

        intermediate = {"centers" : center, "labels" : labels}

        if inferred == 0:
            dict[v]["views"][f]["ground_truth"] = intermediate
        else:
            dict[v]["views"][f]["metagen_inferences"] = intermediate

with open(f"{path}{file}", "w") as f:
    #print(f"{type(dict)}")
    json.dump(dict, f)
