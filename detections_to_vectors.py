"""
This script is for taking a json file with 2D detections in pixel space and the
camera postion and turning those detections into into direction vectors.
"""
import json
import numpy as np
import math
import geometry as geometry
import sys

num_videos = 50
num_frames = 300

path = ""
#file = "data_labelled_retinanet_1" #change to making it a commandline argment
file = sys.argv[1]
f = open(f"{path}{file}")
dict = json.load(f)

for v in range(num_videos):
    print(f"v {v}")
    for f in range(num_frames):
        centers = np.array(dict[v]["views"][f]["detections"]["center"])

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

        vectors = []
        for i in range(len(centers)): #for each detection
            #do the math
            center = centers[i]
            vec = geometry.get_vector(camera_pos, camera_focus, center)
            #print(f"{vec}")
            vectors.append(vec.tolist()) #to list necessary in order to write to json

        dict[v]["views"][f]["detections"]["vector"] = vectors


with open(f"vectorized_{path}{file}.json", "w") as f:
    #print(f"{type(dict)}")
    json.dump(dict, f)

# camera_pos = np.array([0.01, 2, 0.001])
# camera_focus = np.array([1.0001, 0.1, 0.0001])
# center = np.array([1., 1.])
# vec = geometry.get_vector(camera_pos, camera_focus, center)
# np.set_printoptions(precision=2, suppress = True)
# print(f"{vec}")
