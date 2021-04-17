# #
# # from sklearn.model_selection import KFold
# # kf = KFold(n_splits=5)
# #
# #
# # import cv2
# # import os
# # import numpy as np
# # lines=os.listdir('./data/train/images')
# # non_empty=[]
# # empty=[]
# # for l in lines:
# #     mask=cv2.imread('./data/train/masks/'+l)
# #     # print(np.max(mask))
# #     if np.max(mask)==0:
# #         empty.append(l)
# #     else:
# #         non_empty.append(l)
# # print(len(empty))
# # print(empty[0])
# # print(len(non_empty))
# # print(non_empty[0])
# # X=range(len(non_empty))
# # kf.get_n_splits(X)
# # for index,(tr,te) in enumerate(kf.split(X)):
# #     print(tr)
# #     print(te)
# #     # f=open('./data/split/train_empty_1312','w')
# #     # for l in empty[:1312]:
# #     #     f.write('train/'+l.strip().split('.')[0]+'\n')
# #     # f.close()
# #     # f=open('./data/split/valid_empty_250','w')
# #     # for l in empty[1312:]:
# #     #     f.write('train/'+l.strip().split('.')[0]+'\n')
# #     # f.close()
# #     f=open('./data/split/train_nonempty_fold_'+str(index),'w')
# #     for trdex in tr:
# #         l=non_empty[trdex]
# #         f.write('train/'+l.strip().split('.')[0]+'\n')
# #     f.close()
# #     f=open('./data/split/valid_nonempty_fold_'+str(index),'w')
# #     for valdex in te:
# #         l=non_empty[valdex]
# #         f.write('train/'+l.strip().split('.')[0]+'\n')
# #     f.close()
#
# out_dir = \
#     'F:/WorkStation2/tgs_pytorch/output/seresnext50_bn/non_empty_test_10fold/'
# split, mode  = 'test_18000',  'test'
# output_files=[]
# import numpy as np
# def merge_prob(output_files,augment,fold):
#     for i,f in enumerate(output_files):
#         if i==0:
#             all_prob=np.load(f).astype(np.float32) / 255
#         else:
#             all_prob+=np.load(f).astype(np.float32) / 255
#     all_prob=all_prob/len(output_files)
#     all_prob=(all_prob * 255).astype(np.uint8)
#     np.save(out_dir + '/test/merged_submit-%s-%s-%s.prob.uint8.npy' % (fold,split, augment), all_prob)
# for a in ["null",'flip']:
#     for fold in range(9):
#         output_files.append(out_dir+'merged_submit-'+str(fold)+'-test_18000-'+a+'.prob.uint8')
# merge_prob(output_files)

from sklearn.cluster import KMeans
import numpy as np
X=np.array([[1,1],[1.5,2],[4,4],[5,7],[6,8]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print(kmeans.cluster_centers_)