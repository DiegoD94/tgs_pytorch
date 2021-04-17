import os
from data_util import *
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle as p
from matplotlib.pyplot import bar
# ======================= parameters =======================
# calculate KL with or without zero mask from 0.871
dist_with_zero = False
use_train_annotation = True
thres_start = 0.42
thres_end = 0.55
thres_step = 0.0001
bin_num = 20
polyfit_order = 7
# test sigmoid output here
test_sigmoid_output_file = "F:/WorkStation2/tgs_pytorch/output/seresnext50_bn/non_empty_test_10fold/test/final_weighted_merge.prob.uint8.npy"
# test zero out file here
test_zero_out_file = "F:/WorkStation2/tgs_pytorch/8519771/10folds_ne_majvote_vert_corrected.csv"
# train or valid sigmoid output here
train_sigmoid_output_file = "F:/WorkStation2/tgs_pytorch/output/seresnext50_bn/non_empty_test_10fold/test/merged_submit-1-test_18000-flip.prob.uint8.npy"
# train or valid zero out file here
train_zero_out_file = "F:/WorkStation2/tgs_pytorch/8519771/10folds_ne_majvote_vert_corrected.csv"
train_mask_dir = "F:/WorkStation2/tgs_pytorch/data/train/masks/"
# ======================= parameters =======================



threshold = np.arange(thres_start, thres_end, thres_step)

zero_out_index = bin_num + 1 if dist_with_zero else 0




def get_KL(test_edf,test_sigmoid_output):
    if np.max(test_sigmoid_output) > 120:  # [0,255] else [0,1]
        test_sigmoid_output = test_sigmoid_output / 255.0
    # print(test_sigmoid_output)
    test_zero_index = []
    for n, id in enumerate(test_edf.index):
        if type(test_edf.loc[id]['rle_mask']) == float:
            test_zero_index.append(n)
    # print(test_zero_index)
    print('zero file include ' + str(len(test_zero_index)) + ' zero masks')
    test_zero_index = set(test_zero_index)
    train_edf = pd.read_csv(train_zero_out_file, index_col='id')
    train_sigmoid_output = np.load(train_sigmoid_output_file).astype(np.float32)
    if np.max(train_sigmoid_output) > 120:  # [0,255] else [0,1]
        train_sigmoid_output = train_sigmoid_output / 255.0
    # print(train_sigmoid_output)
    train_zero_index = []
    for n, id in enumerate(train_edf.index):
        if type(train_edf.loc[id]['rle_mask']) == float:
            train_zero_index.append(n)
    # print(train_zero_index)
    print('zero file include ' + str(len(train_zero_index)) + ' zero masks')
    train_zero_index = set(train_zero_index)

    train_mask_list = os.listdir(train_mask_dir)
    train_masks = np.zeros([4000,101,101],dtype=np.uint8)
    print(len(train_mask_list))
    for i,l in enumerate(train_mask_list):
        #print((cv2.cvtColor(cv2.imread(train_mask_dir + l), cv2.COLOR_BGR2GRAY) / 255).shape)
        train_masks[i]=(cv2.cvtColor(cv2.imread(train_mask_dir + l), cv2.COLOR_BGR2GRAY) / 255)
    # train_masks = np.array(train_masks)
    train_dist = _get_cover_distribute(train_masks, set())
    print(train_dist)
    for t in tqdm(threshold):
        test_output = test_sigmoid_output > t
        test_dist = _get_cover_distribute(test_output, test_zero_index)
        if not use_train_annotation:
            train_output = train_sigmoid_output > t
            train_dist = _get_cover_distribute(train_output, train_zero_index)
        # bar(np.arange(0,bin_num,1),np.array(test_dist),width=0.4)
        # bar(np.arange(0,bin_num,1)+0.5,np.array(train_dist),width=0.4)
        # plt.show()
        # print(test_dist)
        # print(train_dist)
        KL = 0.0
        for i in range(bin_num-1):
            KL += test_dist[i] * np.log(test_dist[i] / train_dist[i])
        reverse_KL = 0.0
        for i in range(bin_num-1):
            reverse_KL += train_dist[i] * np.log(train_dist[i] / test_dist[i])
        threshes.append(t)
        KLs.append(KL)
        rKLs.append(reverse_KL)
        JS = 0.0
        m_dist=train_dist+test_dist
        m_dist=m_dist/np.sum(m_dist)
        for i in range(bin_num-1):
            JS += 0.5 * test_dist[i] * np.log(test_dist[i] / m_dist[i]) + \
                  0.5 * train_dist[i] * np.log(train_dist[i] / m_dist[i])
        JSs.append(JS)






def _get_cover_distribute(output, zero_index):
    result = [0] * (bin_num + 2)
    for n in range(len(output)):
        if n in zero_index:
            result[zero_out_index] += 1
            continue
        # print(output)
        result[get_coverness(output[n])] += 1
    result = np.array(result[1:bin_num])
    return result * 1.0 / (np.sum(result))


def get_coverness(output):
    # print(output.shape)
    # print(np.sum(output[output==1]))
    coverness = np.round(bin_num * (np.sum(output[output == 1]) / (output.shape[0] * output.shape[1])))
    # print(coverness)
    return int(coverness)

files=os.listdir("F:/WorkStation2/tgs_pytorch/download/final/sigmoid/new/")
# thresholds=[0.5,0.3,0.2]
data=np.zeros([18000,101,101])
best_threshes=[]
for file_index in range(len(files)):
    if not files[file_index].endswith('.p'):
        continue
    # if files[file_index].startswith('100'):
    #     continue
    print("F:/WorkStation2/tgs_pytorch/download/final/sigmoid/new/"+files[file_index])
    f=open("F:/WorkStation2/tgs_pytorch/download/final/sigmoid/new/"+files[file_index],'rb')
    threshes = []
    KLs = []
    rKLs = []
    JSs = []
    data_0=p.load(f)
    print(data_0.shape)
    data_0_r=np.zeros([18000,101,101])
    for i,img in enumerate(data_0):
        img=cv2.resize(img,(101,101))
        data_0_r[i]=img
    # data+=data_0_r*thresholds[file_index]


    # data_0_r=np.load('F:/WorkStation2/tgs_pytorch/download/100models//test/final_weighted_merge.prob.uint8.npy')
    test_edf = pd.read_csv(test_zero_out_file, index_col='id')
# test_sigmoid_output = np.load(test_sigmoid_output_file).astype(np.float32)
    get_KL(test_edf,data_0_r)


# plt.plot(threshes, KLs, '*', color='red')
# params = np.polyfit(threshes, KLs, polyfit_order)
# fitted_threshes = np.arange(thres_start, thres_end, 1e-6)
# fitted_KLs = np.polyval(params, fitted_threshes)
# plt.plot(fitted_threshes, fitted_KLs, '-', color='red')
#
# plt.plot(threshes, rKLs, '*', color='green')
# params = np.polyfit(threshes, rKLs, polyfit_order)
# fitted_threshes = np.arange(thres_start, thres_end, 1e-6)
# fitted_rKLs = np.polyval(params, fitted_threshes)
# plt.plot(fitted_threshes, fitted_rKLs, '-', color='green')

    plt.plot(threshes, JSs, '*', color='blue')
    for i in range(len(JSs)):
        if JSs[i]==min(JSs):
            print("best: ",threshes[i])
            best_threshes.append(threshes[i])
            mask=data_0_r>=threshes[i]
            assert mask.shape==(18000,101,101)
            split_file = 'E:\\DHWorkStation\\Project\\tgs_pytorch\\data/split/test_18000'
            lines = read_list_from_file(split_file)

            id = []
            rle_mask = []
            for n, line in enumerate(lines):
                folder, name = line.split('/')
                id.append(name)

                if (mask[n].sum() <= 0):
                    encoding = ''
                else:
                    encoding = run_length_encode(mask[n])
                assert (encoding != [])

                rle_mask.append(encoding)

            df = pd.DataFrame({'id': id, 'rle_mask': rle_mask}).astype(str)
            df.to_csv("F:/WorkStation2/tgs_pytorch/download/final/sigmoid/new/optimized_"+str(files[file_index]).strip().split('.')[0]+".csv", index=False, columns=['id', 'rle_mask'], encoding='utf-8')
            print('submit done')



    params = np.polyfit(threshes, JSs, polyfit_order)
    fitted_threshes = np.arange(thres_start, thres_end, 1e-6)
    fitted_JSs = np.polyval(params, fitted_threshes)
    plt.plot(fitted_threshes, fitted_JSs, '-', color='blue')





    # plt.show()
    plt.xlim(0.37,0.63)
    # plt.ylim(0,0.005)
    plt.savefig("F:/WorkStation2/tgs_pytorch/download/final/sigmoid/new/threshold_"+str(files[file_index]).strip().split('.')[0]+".png")
    plt.clf()
print(best_threshes)
