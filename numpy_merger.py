import numpy as np
import pandas as pd
predict_dir='F:/WorkStation2/tgs_pytorch/download/oc_net_256/test/'
# submit_score=[0.85154,0.86353,0.83402,0.87875,0.87414,0.84647,0.85708,0.86169,0.85569,0.87731]
submit_score=[0.8692,0.8677,0,0.888,0.8852,0.8511,0.8741,0.8746,0.8571,0.8968]

for i in range(10):
    print(i)
    if i==2:
        continue
    for a in ['null','flip']:
        if i==0 and a=='null':
            predict=submit_score[i]*0.5*np.load(predict_dir+'/merged_submit-'+str(i)+'-test_18000-'+a+'.prob.uint8.npy').astype(np.float32) / 255
            print(predict.shape)
            print(np.mean(predict)/(submit_score[i]*0.5))
            print(np.max(predict)/(submit_score[i]*0.5))
            print(np.min(predict)/(submit_score[i]*0.5))
        else:
            predict+=submit_score[i]*0.5*np.load(predict_dir+'/merged_submit-'+str(i)+'-test_18000-'+a+'.prob.uint8.npy').astype(np.float32) / 255
predict/=np.sum(submit_score)
print(predict.shape)
print(np.mean(predict))
print(np.max(predict))
print(np.min(predict))
predict=np.uint8(predict*255)

# other_model=[
#     'E:/DHWorkStation/Project/tgs_pytorch/output/seresnext50_bn/test/merged_submit-test_18000-flip.prob.uint8.npy',
#     'E:/DHWorkStation/Project/tgs_pytorch/output/seresnext50_bn/test/merged_submit-test_18000-null.prob.uint8.npy'
# ]
# this_model_score=0.863
# model_score=[0.861]
# for index,m in enumerate(other_model):
#     predict+= model_score[int(index//2)]*0.5*np.load(m).astype(np.float32) / 255
# predict/=this_model_score+np.sum(model_score)

np.save(predict_dir+'final_weighted_merge.prob.uint8.npy',predict)
test_list=open('E:/DHWorkStation/Project/tgs_pytorch/data/split/test_18000','r')
lines=test_list.readlines()
predict_mask=(predict/255)>0.50
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
id=[]
rle_mask=[]
for n, line in enumerate(lines):
    folder, name = line.strip().split('/')
    id.append(name)
    if (predict_mask[n].sum() <= 0):
        encoding = ''
    else:
        encoding = run_length_encode(predict_mask[n])
    assert (encoding != [])

    rle_mask.append(encoding)
csv_file=predict_dir+'final_weighted_vote.csv'
df = pd.DataFrame({'id': id, 'rle_mask': rle_mask}).astype(str)
df.to_csv(csv_file, index=False, columns=['id', 'rle_mask'], encoding='utf-8')
print('submit done')

