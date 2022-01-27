"""
This script is for taking a json file with vecotor detections and the
camera postion and the ground-truth objects in 3D and calculating the
ground-truth confusion + hallucination V matrix. This version
considers maps each detection vector to the nearest ground_truth object withing
a 3D radius.
"""
import json
import numpy as np
import math
#from add_2D import label_to_num
from geometry import on_right_side

NN = "retinanet"

num_videos = 100
num_frames = 300

n_categories = 5

radius = 1.

def label_to_num(label):
    dictionary = {"chair" : 62, "plant" : 64, "tv" : 72, "umbrella" : 28, "bowl" : 51} #going by detector outputs from cluster
    temp = dictionary[label]
    #maps COCO id number to row of V_matrix
    dictionary2 = {62 : 1, 64 : 2, 72 : 3, 28 : 4, 51 : 5}
    return dictionary2[temp]

dictionary = {62 : 1, 64 : 2, 72 : 3, 28 : 4, 51 : 5}
list = [62, 64, 72, 28, 51]

dict = json.load(open(f"vectorized_data_labelled_{NN}_1.json.json"))
#parse for ground_truth
for batch in range(2, 7):
    f = open(f"vectorized_data_labelled_{NN}_{batch}.json.json")
    dict_to_add = json.load(f)
    dict.extend(dict_to_add)

V_matrix_numerator = np.zeros(shape=(n_categories + 1, n_categories)) #extra row is for hallucination rates
V_matrix_denominator = np.zeros(shape=(n_categories + 1, n_categories)) #extra row is for hallucination rates

pH_numerator = 0 #pH is the probablility that a detection is a hallucination
pH_denominator = 0

def dist(gt_pos, camera_pos, det_vec):
    #print(f"sin {(np.linalg.norm(np.cross(gt_pos - camera_pos, det_vec)) / np.linalg.norm(det_vec)) / np.linalg.norm(gt_pos - camera_pos)}")
    return np.linalg.norm(np.cross(gt_pos - camera_pos, det_vec)) / np.linalg.norm(det_vec) #norm should be 1

for v in range(num_videos):
    print(f"v {v}")
    #get all the gts
    gt_labels = []
    gt_poses = []
    for i in range(len(dict[v]["labels"])): #got all the ground_truth
        gt_labels.append(label_to_num(dict[v]["labels"][i]["category_name"]))
        gt_poses.append(np.array(dict[v]["labels"][i]["position"]))

    for f in range(num_frames):
        #print(f"f {f}")
        c_x = dict[v]["views"][f]["camera"]["x"]
        c_y = dict[v]["views"][f]["camera"]["y"]
        c_z = dict[v]["views"][f]["camera"]["z"]
        camera_pos = np.array([c_x, c_y, c_z])

        f_x = dict[v]["views"][f]["lookat"]["x"]
        f_y = dict[v]["views"][f]["lookat"]["y"]
        f_z = dict[v]["views"][f]["lookat"]["z"]
        camera_focus = np.array([f_x, f_y, f_z])

        det_labels = dict[v]["views"][f]["detections"]["labels"]
        det_labels = [label for label in det_labels if label in list]
        #may need to add something for taking top n detections or thresholding
        for d in range(len(det_labels)):
            det_label = dictionary[det_labels[d]]
            center = dict[v]["views"][f]["detections"]["center"][d]
            det_vector = np.array(dict[v]["views"][f]["detections"]["vector"][d])
            #for each detection, find nearest gt
            nearest_gt_index = 0
            smallest_distance = radius + 1
            for g in range(len(gt_labels)):
                if on_right_side(camera_pos, camera_focus, gt_poses[g]):
                    distance = dist(gt_poses[g], camera_pos, det_vector)
                else:
                    distance = radius + 1 #something above the radius
                if distance < smallest_distance:
                    smallest_distance = distance
                    nearest_gt_index = g

            #update V_matrix
            j = det_label-1 #colums are different because of extra row
            #pairings
            if smallest_distance < radius: #match this detection to this ground_truth
                i = gt_labels[nearest_gt_index]
                V_matrix_numerator[i,j] = V_matrix_numerator[i,j] + 1
                V_matrix_denominator[i] = V_matrix_denominator[i] + 1 #each time a gt is in a pair, up that whole row since a sample was drawn from its Gaussian
            else: #hallucinations
                V_matrix_numerator[0,j] = V_matrix_numerator[0,j] + 1
                V_matrix_denominator[0] = V_matrix_denominator[0] + 1
                pH_numerator = pH_numerator + 1
                #print("hallucination")
            pH_denominator = pH_denominator + 1

        #np.set_printoptions(precision=2, suppress = True)
        #print(V_matrix_numerator / V_matrix_denominator)
#V_matrix_denominator[0] = sum(V_matrix_numerator[0]) #now 0th row V_matrix would be P(j observed | detection is hallucination)

np.set_printoptions(precision=2, suppress = True)
V_matrix = V_matrix_numerator / V_matrix_denominator
pH = pH_numerator/pH_denominator
V_matrix[0] = V_matrix[0] * pH
print(V_matrix)
#multiply top row by P(hallucination) so it is P(j observed | detection is hallucination)*P(hallucination)
np.savetxt(f"{NN}_results/gt_V_{NN}.txt", V_matrix)
