#!/bin/bash

python3 npz_to_json.py #now have one big json with MetaGen's inferences
python3 add_2D.py 0 #add 2D for ground_truth
python3 add_2D.py 1 #add 2D for MetaGen
python3 accuracy_to_csv.py #make the csv file
