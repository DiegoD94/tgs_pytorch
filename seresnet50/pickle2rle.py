import numpy as np
import pickle as p


def run_length_encode(x):
    bs = np.where(x.T.flatten())[0]

    rle = []
    prev = -2
    for b in bs:
        if (b>prev+1): rle.extend((b + 1, 0))
        rle[-1] += 1
        prev = b

    #https://www.kaggle.com/c/data-science-bowl-2018/discussion/48561#
    #if len(rle)!=0 and rle[-1]+rle[-2] == x.size:
    #    rle[-2] = rle[-2] -1

    rle = ' '.join([str(r) for r in rle])
    return rle

def run_length_decode(rle, H, W, fill_value=255):

    mask = np.zeros((H,W), np.uint8)
    if rle=='': return mask

    mask = mask.reshape(-1)
    rle = np.array([int(s) for s in rle.split(' ')]).reshape(-1, 2)
    for r in rle:
        start = r[0]-1
        end = start + r[1]
        mask[start : end] = fill_value
    mask = mask.reshape(W, H).T   # H, W need to swap as transposing.
    return mask

