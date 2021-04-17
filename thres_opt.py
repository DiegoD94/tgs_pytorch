import numpy as np
import cv2
import pickle as p
def get_coverage(imgs):
    '''

    :param imgs: list of binary images
    :return: list of area coverage of imgs
    '''
    def _get_coverage(img):
        assert(len(img.shape))==2
        assert img.shape[0]==101
        # cv2.namedWindow("0")
        # cv2.imshow("0", mat=img)
        # cv2.waitKey(0)
        # print((img))
        return np.sum(img)/(img.shape[0]*img.shape[1])
    ans=[]
    for i in imgs:
        ans.append(_get_coverage(i))
    return ans

f=open("F:/100models/0_ResNet34_res_25600029500_model.p",'rb')

data_0=p.load(f)
# print(data_0[0,:,:])
data_0_r=np.zeros([18000,101,101])


for i,img in enumerate(data_0):
    img=cv2.resize(img,(101,101))

    data_0_r[i]=img
# print(data_0_r[0, :, :])
coverages=(np.round(100*np.array(get_coverage(np.round(data_0_r))))*0.01)
print(np.sum(coverages[coverages==1]))








