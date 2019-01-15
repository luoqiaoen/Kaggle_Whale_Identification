import pandas as pd
from shutil import copyfile

dftrain = pd.read_csv("../large_dataset/whale_files/yolo_box/manual_check_train.csv")
dftest  = pd.read_csv("../large_dataset/whale_files/yolo_box/manual_check_test.csv")

for _ , row in dftrain.iterrows():
    unique_id = row['Image']
    copyfile("../large_dataset/whale_files/train/"+ unique_id, "../large_dataset/whale_box/whale_missed_train/"+ unique_id)
for _ , row in dftest.iterrows():
    unique_id = row['Image']
    copyfile("../large_dataset/whale_files/test/"+ unique_id, "../large_dataset/whale_box/whale_missed_test/"+ unique_id)
