import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # '3,2' #'3,2,1,0'

from data_util import *

# from unet_5_scale_more_aug.model import SaltNet as Net
# from resnet34.model_resnet34_bn import SaltNet as Net
from seresnet50.model_se_resnext50_bn import SeResNeXt50Unet  as Net
# from resnet_aug0.model import SaltNet as Net
SIZE = 202
PAD_1 = 27
PAD_2 = 27
Y0, Y1, X0, X1 = PAD_1, PAD_1 + SIZE, PAD_1, PAD_1 + SIZE,

## global setting ############################################################


out_dir = \
    'F:/WorkStation2/tgs_pytorch/output/seresnext50_bn/non_empty_test_10fold_finetune/'


# 3600_1
#0.854
# initial_checkpoints = [
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\checkpoint\\00071200_loss_0.237_model.pth',
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\checkpoint\\00062000_iou_0.882_model.pth',
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\checkpoint\\00072200_iou_0.879_model.pth'
# ]
#0.861
# initial_checkpoints=[
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold0\\checkpoint\\00043200_loss_0.267_model.pth',
#     #'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold0\\checkpoint\\00061800_loss_0.273_model.pth',
#     #'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold2\\checkpoint\\00045600_iou_0.874_model.pth',
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold2\\checkpoint\\00046600_iou_0.877_model.pth',
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold2\\checkpoint\\00046400_iou_0.875_model.pth',
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\checkpoint\\00071200_loss_0.237_model.pth',
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\checkpoint\\00062000_iou_0.882_model.pth',
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\checkpoint\\00072200_iou_0.879_model.pth',
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold4\\checkpoint\\00053600_loss_0.242_model.pth',
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold4\\checkpoint\\00053800_loss_0.242_model.pth',
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold4\\checkpoint\\00052200_iou_0.881_model.pth',
#
# ]
# merge_files=[
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\oc_test\\test\\00056600_loss_0-test_18000-flip.prob.uint8.npy',
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\oc_test\\test\\00056600_loss_0-test_18000-null.prob.uint8.npy',
#
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\oc_test\\test\\00056800_loss_0-test_18000-flip.prob.uint8.npy',
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\oc_test\\test\\00056800_loss_0-test_18000-null.prob.uint8.npy',
#
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\oc_test\\test\\00062200_iou_0-test_18000-flip.prob.uint8.npy',
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\oc_test\\test\\00062200_iou_0-test_18000-null.prob.uint8.npy',
#
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold0\\test\\00043200_loss_0-test_18000-flip.prob.uint8.npy',
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold0\\test\\00043200_loss_0-test_18000-null.prob.uint8.npy',
#
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold0\\test\\00045600_iou_0-test_18000-flip.prob.uint8.npy',
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold0\\test\\00045600_iou_0-test_18000-null.prob.uint8.npy',
#
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold0\\test\\00046400_iou_0-test_18000-flip.prob.uint8.npy',
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold0\\test\\00046400_iou_0-test_18000-null.prob.uint8.npy',
#
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold0\\test\\00046600_iou_0-test_18000-flip.prob.uint8.npy',
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold0\\test\\00046600_iou_0-test_18000-null.prob.uint8.npy',
#
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold0\\test\\00061800_loss_0-test_18000-flip.prob.uint8.npy',
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold0\\test\\00061800_loss_0-test_18000-null.prob.uint8.npy',
# ]

# merge_files_valid=[
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\oc_test\\test\\00056600_loss_0-test_18000-flip.prob.uint8.npy',
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\oc_test\\test\\00056600_loss_0-test_18000-null.prob.uint8.npy',
#
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\oc_test\\test\\00056800_loss_0-test_18000-flip.prob.uint8.npy',
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\oc_test\\test\\00056800_loss_0-test_18000-null.prob.uint8.npy',
#
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\oc_test\\test\\00062200_iou_0-test_18000-flip.prob.uint8.npy',
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\oc_test\\test\\00062200_iou_0-test_18000-null.prob.uint8.npy',
#
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\oc_test\\test\\00043200_loss_0-test_18000-flip.prob.uint8.npy',
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\oc_test\\test\\00043200_loss_0-test_18000-null.prob.uint8.npy',
#
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold0\\test\\00045600_iou_0-test_18000-flip.prob.uint8.npy',
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold0\\test\\00045600_iou_0-test_18000-null.prob.uint8.npy',
#
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold0\\test\\00046400_iou_0-test_18000-flip.prob.uint8.npy',
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold0\\test\\00046400_iou_0-test_18000-null.prob.uint8.npy',
#
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold0\\test\\00046600_iou_0-test_18000-flip.prob.uint8.npy',
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold0\\test\\00046600_iou_0-test_18000-null.prob.uint8.npy',
#
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold0\\test\\00061800_loss_0-test_18000-flip.prob.uint8.npy',
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold0\\test\\00061800_loss_0-test_18000-null.prob.uint8.npy',
# ]
#dilation 3600_1
# initial_checkpoints=[
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\dilation_test\\checkpoint\\00045800_iou_0.880_model.pth',
#    # 'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\dilation_test\\checkpoint\\00064000_loss_0.252_model.pth',
#     'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\dilation_test\\checkpoint\\00042400_loss_0.252_model.pth',
# ]

#swa_ 3600_1
initial_checkpoints_=[
    # 'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\cosanneal_test\\checkpoint\\00032000_model.pth',
    # 'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\cosanneal_test\\checkpoint\\00040000_model.pth',
    # 'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\cosanneal_test\\checkpoint\\00048000_model.pth',
    # 'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\cosanneal_test\\checkpoint\\00056000_model.pth',
    # 'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\cosanneal_test\\checkpoint\\00064000_model.pth',
    # 'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\cosanneal_test\\checkpoint\\00071999_model.pth',

    # 'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold4\\checkpoint\\00054800_loss_0.227_model.pth',
    # 'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold4\\checkpoint\\00055000_iou_0.883_model.pth',
    # 'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold4\\checkpoint\\00064600_loss_0.226_model.pth',
    # 'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold4\\checkpoint\\00066200_iou_0.881_model.pth',
    # 'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold4\\checkpoint\\00067800_loss_0.229_model.pth',
    # 'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold4\\checkpoint\\00070000_iou_0.883_model.pth',


    # 'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\non_empty\\checkpoint\\00064600_loss_0.226_model.pth',
    # 'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\non_empty\\checkpoint\\00054800_loss_0.227_model.pth',
    # 'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\non_empty\\checkpoint\\00056000_loss_0.230_model.pth',

    # 'F:/WorkStation2/tgs_pytorch/download/fold0/ckpt_1004/00018760_iou_0.860_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold0/ckpt_1004/00005360_iou_0.850_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold0/ckpt_1004/00007035_iou_0.849_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold0/ckpt_1004/00010117_iou_0.853_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold0/ckpt_1004/00013065_iou_0.857_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold0/ckpt_1004/00013936_iou_0.853_model.pth',



    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold0_stage2/00009756_iou_0.855_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold0_stage2/00019241_iou_0.859_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold0_stage2/00037669_iou_0.851_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold0_stage2/00047154_iou_0.850_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold0_stage2/00112465_iou_0.856_model.pth",
    
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold1_stage2/00062601_iou_0.853_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold1_stage2/00020596_iou_0.852_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold1_stage2/00029268_loss_0.558_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold1_stage2/00042005_iou_0.856_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold1_stage2/00053658_iou_0.861_model.pth",

    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold3_stage2/.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold3_stage2/.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold3_stage2/.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold3_stage2/.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold3_stage2/.pth",
    
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold3_stage2/00017886_iou_0.875_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold3_stage2/00027371_iou_0.876_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold3_stage2/00039566_iou_0.876_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold3_stage2/00052574_iou_0.880_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold3_stage2/00060704_iou_0.882_model.pth",

    

    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold4_stage2/00018970_iou_0.872_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold4_stage2/00031707_iou_0.883_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold4_stage2/00037940_iou_0.872_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold4_stage2/00047696_iou_0.876_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold4_stage2/00065040_iou_0.871_model.pth",

    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold5_stage2/00020325_iou_0.834_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold5_stage2/00030081_iou_0.841_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold5_stage2/00040379_iou_0.833_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold5_stage2/00050677_iou_0.842_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold5_stage2/00064498_iou_0.833_model.pth",

    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold6_stage2/00008672_iou_0.856_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold6_stage2/00020596_iou_0.853_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold6_stage2/00040379_iou_0.860_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold6_stage2/00052845_iou_0.860_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold6_stage2/00061788_iou_0.858_model.pth",

    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold7_stage2/00021680_iou_0.866_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold7_stage2/00025745_iou_0.862_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold7_stage2/00038482_iou_0.864_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold7_stage2/00050677_iou_0.858_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold7_stage2/00063956_iou_0.863_model.pth",

    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold8_stage2/00017886_iou_0.842_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold8_stage2/00031978_iou_0.849_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold8_stage2/00037127_iou_0.838_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold8_stage2/00054200_iou_0.840_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold8_stage2/00063414_iou_0.851_model.pth",

    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold9_stage2/00021409_iou_0.878_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold9_stage2/00025745_iou_0.880_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold9_stage2/00041734_iou_0.885_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold9_stage2/00051490_iou_0.884_model.pth",
    "F:/WorkStation2/tgs_pytorch/download/oc_net_256/fold9_stage2/00064498_iou_0.878_model.pth",





    ###---10 fold 1
    # 'F:/WorkStation2/tgs_pytorch/download/fold0/ckpt_1005/00009756_iou_0.855_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold0/ckpt_1005/00019241_iou_0.859_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold0/ckpt_1005/00037669_iou_0.851_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold0/ckpt_1005/00047154_iou_0.850_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold0/ckpt_1005/00112465_iou_0.856_model.pth',


    # 'F:/WorkStation2/tgs_pytorch/download/fold1/00009447_loss_0.237_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold1/00010050_iou_0.856_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold1/00010653_iou_0.858_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold1/00011859_iou_0.858_model.pth',
    #
    # 'F:/WorkStation2/tgs_pytorch/download/fold2/00006700_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold2/00007772_iou_0.832_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold2/00009380_model.pth',
    #
    # 'F:/WorkStation2/tgs_pytorch/download/fold3/00008844_iou_0.881_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold3/00008911_iou_0.882_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold3/00008978_iou_0.882_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold3/00008978_iou_0.883_model.pth',
    #
    # 'F:/WorkStation2/tgs_pytorch/download/fold4/00006566_iou_0.864_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold4/00008040_iou_0.865_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold4/00008174_iou_0.866_model.pth',
    #
    # 'F:/WorkStation2/tgs_pytorch/download/fold5/00004154_iou_0.832_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold5/00004288_iou_0.834_model.pth',
    #
    # 'F:/WorkStation2/tgs_pytorch/download/fold6/00006164_iou_0.850_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold6/00006700_iou_0.851_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold6/00006834_iou_0.849_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold6/00006968_iou_0.851_model.pth',
    #
    # 'F:/WorkStation2/tgs_pytorch/download/fold7/00005226_iou_0.851_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold7/00005427_iou_0.853_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold7/00005695_iou_0.856_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold7/00005695_iou_0.858_model.pth',
    #
    # 'F:/WorkStation2/tgs_pytorch/download/fold8/00001206_loss_0.647_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold8/00002211_iou_0.836_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold8/00003484_iou_0.843_model.pth',
    #
    # 'F:/WorkStation2/tgs_pytorch/download/fold9/00001809_loss_0.542_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold9/00003551_iou_0.860_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold9/00005963_iou_0.863_model.pth',
    # 'F:/WorkStation2/tgs_pytorch/download/fold9/00006834_iou_0.865_model.pth',
    ###---10 fold 1

]


prob_preds=[

]

# out_dir + '/resnet34/checkpoint/00018000_model.pth'
# '/root/share/project/kaggle/tgs/results/simple-004-d/checkpoint/00032000_model.pth'
# '/root/share/project/kaggle/tgs/results/simple-004-b/checkpoint/00016000_model.pth'
# '/root/share/project/kaggle/tgs/results/simple-002-02-xx/checkpoint/00014000_model.pth'

# split, mode = 'valid_800_4', 'valid'


split, mode  = 'test_18000',  'test'

# #augment = 'flip'
# augment = 'null'
# #augment = 'intensity'
# #augment = 'intensity-flip'
#


def augment_flip(image, mask, index):
    cache = Struct(image=image.copy(), mask=mask.copy())

    if mask == []:
        image = do_horizontal_flip(image)
        # image = do_center_pad_to_factor(image, factor=32)
        image = cv2.resize(image, dsize=(SIZE, SIZE))
        image = do_center_pad(image, PAD_1, PAD_2)
    else:
        image, mask = do_horizontal_flip2(image, mask)
        image, mask = do_resize2(image, mask, SIZE, SIZE)
        image, mask = do_center_pad2(image, mask, PAD_1, PAD_2)
        # image, mask = do_center_pad_to_factor2(image, mask, factor=32)
    # print(image.shape)
    # print(mask.shape)
    return image, mask, index, cache


def unaugment_flip(prob):
    # dy0, dy1, dx0, dx1 = compute_center_pad(IMAGE_HEIGHT, IMAGE_WIDTH, factor=32)
    # prob = prob[:, dy0:dy0 + IMAGE_HEIGHT, dx0:dx0 + IMAGE_WIDTH]
    res = []
    for p in prob:
        p = p[Y0:Y1, X0:X1]
        p = p[:, ::-1]
        p = cv2.resize(p, (101, 101))
        res.append(p)
    res = np.array(res)
    # prob = prob[:, Y0:Y1, X0:X1]
    # prob = prob[:,:, ::-1]
    return res


# ---------------------
# augment == 'null' :
def augment_null(image, mask, index):
    cache = Struct(image=image.copy(), mask=mask.copy())

    if mask == []:
        image = cv2.resize(image, dsize=(SIZE, SIZE))
        image = do_center_pad(image, PAD_1, PAD_2)
        # image = do_center_pad_to_factor(image, factor=32)
    else:
        image, mask = do_resize2(image, mask, SIZE, SIZE)
        image, mask = do_center_pad2(image, mask, PAD_1, PAD_2)
        # image, mask = do_center_pad_to_factor2(image, mask, factor=32)

    return image, mask, index, cache


def unaugment_null(prob):
    res = []
    for p in prob:
        p = p[Y0:Y1, X0:X1]
        p = cv2.resize(p, (101, 101))
        res.append(p)
    res = np.array(res)
    # dy0, dy1, dx0, dx1 = compute_center_pad(IMAGE_HEIGHT, IMAGE_WIDTH, factor=32)
    # prob = prob[:, dy0:dy0 + IMAGE_HEIGHT, dx0:dx0 + IMAGE_WIDTH]
    return res


def run_predict(augment,ckpt,output_prob):
    if augment == 'null':
        test_augment = augment_null
        test_unaugment = unaugment_null
    if augment == 'flip':
        test_augment = augment_flip
        test_unaugment = unaugment_flip
    # ....................................................

    initial_checkpoint=ckpt
    ## setup  -----------------
    os.makedirs(out_dir + '/test/' + split, exist_ok=True)
    os.makedirs(out_dir + '/backup', exist_ok=True)
    # backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.test.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir + '/log.submit.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    batch_size = 16

    test_dataset = TsgDataset(split, test_augment, mode)
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=null_collate)

    assert (len(test_dataset) >= batch_size)
    log.write('batch_size = %d\n' % (batch_size))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net(dilation=False).cuda()

    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        dict = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        #try:
        loading_dict={}
        for k in dict.keys():
            if 'num_batches_tracked' in k:

                continue
            else:
                loading_dict[k]=dict[k]
        print('f num batched tracked')
        net.load_state_dict(loading_dict)
        # except:
        #     print("wrong dict in load dict")
        #     exit(-1)
        #     # dict=torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        #     # for k in dict.keys():
        #     #
        #     #         print(dict[k])
        #     f=open('dct_1.txt','w')
        #     for k in net.state_dict().keys():
        #         f.write(k+'\n')
        #     f.close()
        #     f = open('dct_2.txt','w')
        #     for k in dict.keys():
        #         f.write(k + '\n')
        #     f.close()

    log.write('%s\n\n' % (type(net)))
    log.write('\n')

    ####### start here ##########################
    all_prob = []
    all_num = 0
    all_loss = np.zeros(2, np.float32)

    net.set_mode('test')
    for input, truth, index, cache in test_loader:
        # print(input.shape)
        #
        print('\r', all_num, end='', flush=True)
        batch_size = len(index)
        all_num += batch_size

        input = input.cuda()
        with torch.no_grad():
            logit = net(input)
            prob = F.sigmoid(logit)

            if 0:  ##for debug
                truth = truth.cuda()
                loss = net.criterion(logit, truth)
                dice = net.metric(logit, truth)
                all_loss += batch_size * np.array((loss.item(), dice.item(),))

        ##-----------------------------
        prob = prob.squeeze().data.cpu().numpy()
        if len(prob.shape)==2:
            prob=np.expand_dims(prob,axis=0)
            print(prob.shape)
        prob = test_unaugment(prob)
        all_prob.append(prob)

        if 0:  ##for debug

            os.makedirs(out_dir + '/test/%s/%s' % (split, augment), exist_ok=True)

            for b in range(batch_size):
                name = test_dataset.ids[index[b]]
                predict = prob[b]
                image = cache[b].image * 255
                truth = cache[b].mask
                image = np.dstack([image, image, image])

                overlay0 = draw_mask_overlay(predict, image, color=[0, 0, 255])
                overlay0 = draw_mask_to_contour_overlay(predict, overlay0, 2, color=[0, 0, 255])

                if truth == []:
                    overlay1 = np.zeros((101, 101, 3), np.float32)
                else:
                    overlay1 = draw_mask_overlay(truth, image, color=[255, 0, 0])

                overlay = np.hstack([image, overlay0, overlay1])
                cv2.imwrite(out_dir + '/test/%s/%s/%s.png' % (split, augment, name), overlay * 255)

                # image_show_norm('overlay',overlay,1,2)
                image_show('overlay', overlay, 2)
                cv2.waitKey(0)

    print('\r', all_num, end='\n', flush=True)
    all_prob = np.concatenate(all_prob)
    # for thres in xrange(0.15,0.85,0.05):
    all_prob = (all_prob * 255).astype(np.uint8)
    print('in run_predict ', out_dir + '/test/%s-%s-%s.prob.uint8.npy' % (output_prob,split, augment))
    np.save(out_dir + '/test/%s-%s-%s.prob.uint8.npy' % (output_prob,split, augment), all_prob)
    print(all_prob.shape)

    print('')
    assert (all_num == len(test_loader.sampler))
    all_loss = all_loss / all_num
    print(all_loss)
    log.write('\n')

def merge_prob(output_files,augment,fold):
    for i,f in enumerate(output_files):
        if i==0:
            all_prob=np.load(f).astype(np.float32) / 255
        else:
            all_prob+=np.load(f).astype(np.float32) / 255
    all_prob=all_prob/len(output_files)
    all_prob=(all_prob * 255).astype(np.uint8)
    np.save(out_dir + '/test/merged_submit-%s-%s-%s.prob.uint8.npy' % (fold,split, augment), all_prob)
    #return all_prob


def run_submit(augment,thres,fold,prob_preds=None):
    print('running submit')
    if augment in ['null', 'flip']:
        augmentation = [
            1, out_dir + '/test/merged_submit-%s-%s-%s.prob.uint8.npy' % (fold,split, augment),
        ]
        csv_file = out_dir + '/test/merged_submit-%s-%s-%s.csv' % (fold,split, augment)

    if augment == 'aug2':
        augmentation = [
            1, out_dir + '/test/merged_submit-%s-%s-%s.prob.uint8.npy' % (fold,split, 'null'),
            1, out_dir + '/test/merged_submit-%s-%s-%s.prob.uint8.npy' % (fold,split, 'flip'),
        ]
        csv_file = out_dir + '/test/merged_submit-%s-%s-%s.csv' % (fold,split, augment)

        ##---------------------------------------

        # augments, csv_file = ['null','flip'], '/submit1_simple-valid0-300-aug.csv.gz'
        # augments, csv_file = ['flip'], '/submit1_simple-xxx-flip.csv.gz'
        # augments, csv_file = ['null'], '/submit1_simple-xxx-null.csv.gz'

        ##---------------------------------------
    # if os.path.exists(csv_file):
    #     print('exists: '+csv_file)
    #     return
        # save
    log_file = csv_file + '.log'
    write_list_to_file(augmentation, log_file)

    augmentation = np.array(augmentation, dtype=object).reshape(-1, 2)
    num_augments = len(augmentation)
    w, augment_file = augmentation[0]
    all_prob = w * np.load(augment_file).astype(np.float32) / 255
    all_w = w
    for i in range(1, num_augments):
        w, augment_file = augmentation[i]
        prob = w * np.load(augment_file).astype(np.float32) / 255
        all_prob += prob
        all_w += w
    all_prob /= all_w
    all_prob = all_prob > thres
    print(all_prob.shape)

    # ----------------------------

    split_file = 'E:\\DHWorkStation\\Project\\tgs_pytorch\\data/split/' + split
    lines = read_list_from_file(split_file)

    id = []
    rle_mask = []
    for n, line in enumerate(lines):
        folder, name = line.split('/')
        id.append(name)

        if (all_prob[n].sum() <= 0):
            encoding = ''
        else:
            encoding = run_length_encode(all_prob[n])
        assert (encoding != [])

        rle_mask.append(encoding)

    df = pd.DataFrame({'id': id, 'rle_mask': rle_mask}).astype(str)
    df.to_csv(csv_file, index=False, columns=['id', 'rle_mask'], encoding='utf-8')
    print('submit done')

    # csv_file = out_dir + '/submit1_iter20k-1.csv'
    # df.to_csv(csv_file, index=False, columns=['id', 'rle_mask'])

    ############################################################################################


def run_local_leaderboard(augment,fold):
    # -----------------------------------------------------------------------
    submit_file = out_dir + '/test/merged_submit-%s-%s-%s.csv' % (fold,split, augment)
    dump_dir = out_dir + '/test/merged_submit-%s-%s-%s-dump' % (fold,split, augment)
    os.makedirs(dump_dir, exist_ok=True)

    log = Logger()
    log.open(out_dir + '/test/log.submit.txt', mode='a')

    split_file = 'E:\\DHWorkStation\\Project\\tgs_pytorch\\data/split/' + split
    lines = read_list_from_file(split_file)
    ids = [line.split('/')[-1] for line in lines]
    sorted(ids)

    df_submit = pd.read_csv(submit_file).set_index('id')
    df_submit = df_submit.fillna('')

    df_truth = pd.read_csv('E:\\DHWorkStation\\Project\\tgs_pytorch\\data/train.csv').set_index('id')
    df_truth = df_truth.loc[ids]
    df_truth = df_truth.fillna('')

    N = len(df_truth)
    predict = np.zeros((N, 101, 101), np.bool)
    truth = np.zeros((N, 101, 101), np.bool)

    for n in range(N):
        id = ids[n]
        p = df_submit.loc[id].rle_mask
        t = df_truth.loc[id].rle_mask
        p = run_length_decode(p, H=101, W=101, fill_value=1).astype(np.bool)
        t = run_length_decode(t, H=101, W=101, fill_value=1).astype(np.bool)

        predict[n] = p
        truth[n] = t

        # if 0:
        #     image_p = predict[n].astype(np.uint8)*255
        #     image_t = truth[n]  .astype(np.uint8)*255
        #     image_show('image_p', image_p,2)
        #     image_show('image_t', image_t,2)
        #     cv2.waitKey(0)

    ##--------------
    ### Threshold Optimizer

    precision, result, threshold = do_kaggle_metric(predict, truth, threshold=0.5)
    precision_mean = precision.mean()

    tp, fp, fn, tn_empty, fp_empty = result.transpose(1, 2, 0).sum(2)
    all = tp + fp + fn + tn_empty + fp_empty
    p = (tp + tn_empty) / (tp + tn_empty + fp + fp_empty + fn)

    log.write('\n')
    log.write('      |        |                                      |           empty          |         \n')
    log.write('th    |  prec  |      tp          fp          fn      |      tn          fp      |         \n')
    log.write('-------------------------------------------------------------------------------------------\n')
    for i, t in enumerate(threshold):
        log.write(
            '%0.2f  |  %0.2f  |  %3d / %0.2f  %3d / %0.2f  %3d / %0.2f  |  %3d / %0.2f  %3d / %0.2f  | %5d\n' % (
                t, p[i],
                tp[i], tp[i] / all[i],
                fp[i], fp[i] / all[i],
                fn[i], fn[i] / all[i],
                tn_empty[i], tn_empty[i] / all[i],
                fp_empty[i], fp_empty[i] / all[i],
                all[i])
        )

    log.write('\n')
    log.write('num images :    %d\n' % N)
    log.write('LB score   : %0.5f\n' % (precision_mean))

    # --------------------------------------
    predict = predict.reshape(N, -1)
    truth = truth.reshape(N, -1)
    p = predict > 0.5
    t = truth > 0.5
    intersection = t & p
    union = t | p
    # iou = intersection.sum(1)/(union.sum(1)+EPS)
    log.write('iou        : %0.5f\n' % (intersection.sum() / (union.sum() + EPS)))

    return
    # exit(0)
    ## show --------------------------

    predicts = predict.reshape(-1, 101, 101).astype(np.float32)
    truths = truth.reshape(-1, 101, 101).astype(np.float32)
    for m, name in enumerate(ids):
        print('%s' % name)
        print('      |        |               |  empty  |      ')
        print('th    |  prec  |  tp  fp  fn   |  tn  fp |      ')
        print('------------------------------------------------')
        for i, t in enumerate(threshold):
            tp, fp, fn, fp_empty, tn_empty = result[m, :, i]
            p = (tp + tn_empty) / (tp + tn_empty + fp + fp_empty + fn)
            print('%0.2f  |  %0.2f  |   %d   %d   %d   |   %d   %d   ' % (
                t, p, tp, fp, fn, fp_empty, tn_empty))
        print(precision[m])
        print('')
        # ----
        image_file = '/root/share/project/kaggle/tgs/data/train/images/' + name + '.png'
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        # mask = mask>0

        predict = predicts[m]
        truth = truths[m]

        # print(predict.sum())

        overlay0 = draw_mask_overlay(predict, image, color=[0, 0, 255])
        overlay0 = draw_mask_to_contour_overlay(predict, overlay0, 1, color=[0, 0, 255])
        overlay1 = draw_mask_overlay(truth, image, color=[0, 255, 0])
        overlay1 = draw_mask_to_contour_overlay(truth, overlay1, 1, color=[0, 255, 0])
        overlay2 = draw_mask_overlay(predict, None, color=[0, 0, 255])
        overlay2 = draw_mask_overlay(truth, overlay2, color=[0, 255, 0])

        draw_shadow_text(image, '%0.2f' % precision[m], (3, 15), 0.5, [255, 255, 255], 1)

        overlay = np.hstack([image, overlay0, overlay1, overlay2])
        cv2.imwrite(dump_dir + '/%s.png' % name, overlay)
        image_show('overlay', overlay, 2)
        cv2.waitKey(1)




models=[(0,5),(5,10),(10,15),(15,20),(20,25),(25,30),(30,35),(35,40),(40,45),(45,50)]
threshes=[0.525,0.515,0.44,0.46,0.465,0.48,0.485,0.465,0.455,0.44]
for i in range(10):
    print(models[i][0],models[i][1])
    initial_checkpoints=initial_checkpoints_[models[i][0]:models[i][1]]
    print(initial_checkpoints)
    if i==2 :
        continue
    # split, mode = 'list_valid'+str(i)+'_400_ne_balanced', 'valid'
    split, mode = 'test_18000', 'test'
    if mode == 'valid':
        for a in ['null', 'flip']:
            #if merge_files is None:
            merge_files=[]
            for ckpt in initial_checkpoints:
                print('a=', a)
                print(ckpt)
                print(ckpt.split('/')[-1].split('m')[0])
                if os.path.exists(out_dir+('/test/%s-%s-%s.prob.uint8.npy' % (ckpt.split('/')[-1].split('m')[0],split, a))):
                    print('exist: ',out_dir+('/test/%s-%s-%s.prob.uint8.npy' % (ckpt.split('/')[-1].split('m')[0],split, a)))
                else:
                    run_predict(a,ckpt,ckpt.split('/')[-1].split('m')[0])
                merge_files.append(out_dir+('/test/%s-%s-%s.prob.uint8.npy' % (ckpt.split('/')[-1].split('m')[0],split, a)))
            merge_prob(merge_files,a,i)

            # run_submit('aug2')

        for t in np.arange(0.4,0.6,0.005):
            print(t)
            for a in [ 'aug2']:
                print('a=', a)
                run_submit(a,t,i)
                run_local_leaderboard(a,i)
                #time.sleep(1)
        # best_t=0.455
        # run_submit(a,t,i)

    if mode == 'test':
        for a in ['null', 'flip']:
            #if merge_files is None:
            merge_files=[]
            for ckpt in initial_checkpoints:
                print('a=', a)
                print(ckpt)
                print(ckpt.split('/')[-1].split('m')[0])
                if os.path.exists(out_dir+('/test/%s-%s-%s.prob.uint8.npy' % (ckpt.split('/')[-1].split('m')[0],split, a))):
                    print('exist: ',out_dir+('/test/%s-%s-%s.prob.uint8.npy' % (ckpt.split('/')[-1].split('m')[0],split, a)))
                else:
                    run_predict(a,ckpt,ckpt.split('/')[-1].split('m')[0])
                merge_files.append(out_dir+('/test/%s-%s-%s.prob.uint8.npy' % (ckpt.split('/')[-1].split('m')[0],split, a)))
            merge_prob(merge_files,a,i)
            # run_submit('aug2')
        run_submit('aug2',threshes[i],i)
    # run_local_leaderboard()
    print('\nsucess!')
