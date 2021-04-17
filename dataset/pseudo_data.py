import numpy as np
import pickle as p
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
from tqdm import tqdm
def run_length_decode(rle, H, W, fill_value=255):

    mask = np.zeros((H,W), np.uint8)

    if rle=='' or type(rle)== float: return mask

    mask = mask.reshape(-1)
    rle = np.array([int(s) for s in rle.split(' ')]).reshape(-1, 2)
    for r in rle:
        start = r[0]-1
        end = start + r[1]
        mask[start : end] = fill_value
    mask = mask.reshape(W, H).T   # H, W need to swap as transposing.
    return mask

def get_pseudo(rle,data,output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if data.endswith('.p'):
        f=open(data,'rb')
        sigmoids=p.load(f)
    else:
        sigmoids=np.load(data)
    std=np.zeros([18000])
    for i in range(len(sigmoids)):
        std[i]=(np.std(sigmoids[i]))
    df = pd.read_csv(rle, index_col='id')
    valid_image_ids=[]
    valid_image_index=[]
    for i in range(len(sigmoids)):
        # if np.std(sigmoids[i]) > np.percentile(std, 60):
        valid_image_ids.append(df.index[i])
        valid_image_index.append(i)

    for index,ids in tqdm(zip(valid_image_index,valid_image_ids)):
        mask=np.array(run_length_decode(df.loc[ids]['rle_mask'],101,101),dtype=np.uint8)
        sigmoid=cv2.resize(sigmoids[index],(101,101))
        show=np.concatenate([mask,sigmoid],axis=1)
        # print(ids)
        # if ids=="f40bfdeb45":
        #     print(mask)
        #     print(sigmoid)
        #     real_mask=np.array((sigmoid>=255*0.4),dtype=np.uint8)*255
        #     print(real_mask)
        #     cv2.namedWindow("real_mask")
        #     cv2.imshow("real_mask",real_mask)
        #     cv2.waitKey(0)
        # cv2.namedWindow("mask")
        # cv2.imshow("mask",show)
        # cv2.waitKey(10)
        cv2.imwrite(output_dir+ids+'.png',mask)







get_pseudo("F:/8708665/0875_baseline_leak4++.csv","F:/WorkStation2/tgs_pytorch/download/oc_net_256/test/final_weighted_merge.prob.uint8.npy","F:/WorkStation2/tgs_pytorch/data/pseudo/masks/")