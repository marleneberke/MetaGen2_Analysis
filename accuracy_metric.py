"""
This file provides functions for calculating the accuracy metric for MetaGen's inferences.
Set twoD to true to do this in 2D image space, false to do in 3D space.
Labels need to be 1,2,3,4,5
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
import math

num_categories = 5

image_side_length = 256

#########################################
#twoD is a boolean. If it's true, than adjust the cost by the square of the diagonal of the image
def calculate_matrix(ground_truth_centers, detection_centers, twoD):
    matrix = np.zeros(shape=(len(ground_truth_centers), len(detection_centers)))
    for i in range(len(ground_truth_centers)):
        for j in range(len(detection_centers)): #each detection is the job. gt is worker explianing detecion
            #cost is Euclidean distance. could square it if desired
            if twoD:
                # norm = np.linalg.norm(np.array(ground_truth_centers[i]) - np.array(detection_centers[j]))
                # cost = norm/math.sqrt(image_side_length**2 + image_side_length**2)
                # matrix[i,j] = min(1., cost)
                matrix[i,j] = min(1., ((np.linalg.norm(np.array(ground_truth_centers[i]) - np.array(detection_centers[j])))**2)/100) #don't want a cost >1

            else:
                matrix[i,j] = min(1., ((np.linalg.norm(np.array(ground_truth_centers[i]) - np.array(detection_centers[j])))**2)/100) #don't want a cost >1
                #matrix[i,j] = 0 #when 0, same as Jaccard
                #matrix[i,j] = np.linalg.norm(np.array(ground_truth_centers[i]) - np.array(detection_centers[j]))
    return matrix

#this one if for just doing the Euclidean distance.
def calculate_distance_matrix(ground_truth_centers, detection_centers, twoD):
    matrix = np.zeros(shape=(len(ground_truth_centers), len(detection_centers)))
    for i in range(len(ground_truth_centers)):
        for j in range(len(detection_centers)): #each detection is the job. gt is worker explianing detecion
            #cost is Euclidean distance. could square it if desired
            if twoD:
                #norm = np.linalg.norm(np.array(ground_truth_centers[i]) - np.array(detection_centers[j]))
                #cost = norm/math.sqrt(image_side_length**2 + image_side_length**2)
                #matrix[i,j] = min(1., cost)
                matrix[i,j] = np.linalg.norm(np.array(ground_truth_centers[i]) - np.array(detection_centers[j]))
            else:
                #matrix[i,j] = min(1., ((np.linalg.norm(np.array(ground_truth_centers[i]) - np.array(detection_centers[j])))**2)/100) #don't want a cost >1
                #matrix[i,j] = 0 #when 0, same as Jaccard
                matrix[i,j] = np.linalg.norm(np.array(ground_truth_centers[i]) - np.array(detection_centers[j]))
    return matrix

#########################################
#twoD is a boolean. If it's true, than adjust the cost by the square of the diagonal of the image
#returns two values: the first is the similarity metric, and the second is the average distance between paired points
def custom_similarity(gt_labels, gt_positions, metagen_labels, metagen_positions, twoD):
    union_val = 0
    weighted_intersection = 0
    dist_num = 0 #for calculating average distance between paired points. numerator and denominator
    dist_denum = 0
    for category in range(1, num_categories+1):
        gt_index = [i for i,val in enumerate(gt_labels) if val==category]
        metagen_index = [i for i,val in enumerate(metagen_labels) if val==category]
        n_matches = 0
        if len(gt_index)!=0 and len(metagen_index)!=0: #if both contain obj of this category
            cost_matrix = calculate_matrix([gt_positions[i] for i in gt_index], [metagen_positions[i] for i in metagen_index], twoD)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            cost = cost_matrix[row_ind, col_ind].sum()
            n_matches = len(row_ind) #row_ind and col_ind should be the same
            weight = n_matches - cost
            weighted_intersection = weighted_intersection + weight

            dist_matrix = calculate_distance_matrix([gt_positions[i] for i in gt_index], [metagen_positions[i] for i in metagen_index], twoD)
            dist_num = dist_num + dist_matrix[row_ind, col_ind].sum()
            dist_denum = dist_denum + 1

        union_val = union_val + len(gt_index) + len(metagen_index) - n_matches #subtract n_matches to avoid double counting

    if union_val == 0 and weighted_intersection == 0: #if nothing in ground_truth or metagen's inferences
        sim = 1.
    else:
        sim = weighted_intersection/union_val

    #if there were no pairings, dist is NaN.
    if dist_denum  == 0:
        dist = float("NaN")
    else:
        dist = dist_num/dist_denum #max(1, dist_denum) is just to avoid dividing by 0

    return [sim, dist]

#########################################
#twoD is a boolean. If it's true, than adjust the cost by the square of the diagonal of the image
#returns two values: the first is the similarity metric, and the second is the average distance between paired points
def Jaccard_similarity(gt_labels, gt_positions, metagen_labels, metagen_positions, twoD):
    union_val = 0
    intersection = 0
    dist_num = 0 #for calculating average distance between paired points. numerator and denominator
    dist_denum = 0
    for category in range(1, num_categories+1):
        gt_index = [i for i,val in enumerate(gt_labels) if val==category]
        metagen_index = [i for i,val in enumerate(metagen_labels) if val==category]
        n_matches = 0
        if len(gt_index)!=0 and len(metagen_index)!=0: #if both contain obj of this category
            cost_matrix = calculate_matrix([gt_positions[i] for i in gt_index], [metagen_positions[i] for i in metagen_index], twoD)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            cost = cost_matrix[row_ind, col_ind].sum()
            n_matches = len(row_ind) #row_ind and col_ind should be the same
            intersection = intersection + n_matches

            dist_matrix = calculate_distance_matrix([gt_positions[i] for i in gt_index], [metagen_positions[i] for i in metagen_index], twoD)
            dist_num = dist_num + dist_matrix[row_ind, col_ind].sum()
            dist_denum = dist_denum + 1

        union_val = union_val + len(gt_index) + len(metagen_index) - n_matches #subtract n_matches to avoid double counting

    if union_val == 0 and intersection == 0: #if nothing in ground_truth or metagen's inferences
        sim = 1.
    else:
        sim = intersection/union_val

    #if there were no pairings, dist is NaN.
    if dist_denum  == 0:
        dist = float("NaN")
    else:
        dist = dist_num/dist_denum #max(1, dist_denum) is just to avoid dividing by 0

    return [sim, dist]


# gt_labels = np.array([])
# gt_positions = np.array([])
#
# metagen_labels = np.array([1])
# metagen_positions = np.array([[0, 0.75, 0]])
#
# print(similarity(gt_labels, gt_positions, metagen_labels, metagen_positions))
