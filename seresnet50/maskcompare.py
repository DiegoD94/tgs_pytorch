import cv2
import numpy as np
from data_util import *
import pandas as pd
def compare(csv1,csv2,percent):
    count=0
    avg_diff_pixel=0
    df1=pd.read_csv(csv1,index_col='id')
    df2=pd.read_csv(csv2,index_col='id')
    #df1.index=df2.index
    for ids in df1.index:
        rle_mask1=df1.loc[ids]['rle_mask']
        if type(rle_mask1)==float:
            mask1=np.zeros([101,101])
        else:
            mask1=run_length_decode(rle_mask1,101,101)*1.0/255
        rle_mask2=df2.loc[ids]['rle_mask']
        if type(rle_mask2)==float:
            mask2=np.zeros([101,101])
        else:
            try:
                mask2=run_length_decode(rle_mask2,101,101)*1.0/255
            except:
                print(ids)
                print(rle_mask2)
                print(rle_mask2.split(' '))
        # print(mask1,mask2)
        if np.sum(np.abs(mask1-mask2))>=percent*101*101:

            count+=1
    print(count)
    for ids in df1.index:
        rle_mask1=df1.loc[ids]['rle_mask']
        if type(rle_mask1)==float:
            mask1=np.zeros([101,101])
        else:
            mask1=run_length_decode(rle_mask1,101,101)*1.0/255
        rle_mask2=df2.loc[ids]['rle_mask']
        if type(rle_mask2)==float:
            mask2=np.zeros([101,101])
        else:
            mask2=run_length_decode(rle_mask2,101,101)*1.0/255
        avg_diff_pixel+=(np.sum(np.abs(mask1-mask2)))
        if np.sum(np.abs(mask1-mask2))>=percent*101*101:
            show=np.concatenate([mask1,mask2],axis=1)
            cv2.namedWindow("show")
            cv2.imshow("show",show)
            cv2.waitKey(20)
    df1.to_csv(csv1,index=True, columns=['rle_mask'], encoding='utf-8')
    avg_diff_pixel/=18000
    print(avg_diff_pixel)

compare("F:/WorkStation2/tgs_pytorch/download/final/sigmoid/new/optimized_100models_weighted_sum_all_sigmoids_4.csv","F:/WorkStation2/tgs_pytorch/download/final/sigmoid/new/ocnet256_resnet256_0450350101_0871empty_leak4++_stage2+++.csv",0.05)