import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # '3,2' #'3,2,1,0'

from data_util import *

# from unet_5_scale_more_aug.model import SaltNet as Net
# from resnet34.model_resnet34_bn import SaltNet as Net
from seresnet50.model_se_resnext50_bn import SeResNeXt50Unet  as Net
# from resnet_aug0.model import SaltNet as Net
SIZE = 101
PAD_1 = 13
PAD_2 = 14
Y0, Y1, X0, X1 = PAD_1, PAD_1 + SIZE, PAD_1, PAD_1 + SIZE,

## global setting ############################################################


out_dir = \
    'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\non_empty_test\\'

initial_checkpoint = \
    'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\non_empty_test\\checkpoint\\00026752_loss_0.232_model.pth'
# out_dir + '/resnet34/checkpoint/00018000_model.pth'
# '/root/share/project/kaggle/tgs/results/simple-004-d/checkpoint/00032000_model.pth'
# '/root/share/project/kaggle/tgs/results/simple-004-b/checkpoint/00016000_model.pth'
# '/root/share/project/kaggle/tgs/results/simple-002-02-xx/checkpoint/00014000_model.pth'

# split, mode = 'valid_nonempty_390', 'valid'


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
        # image = cv2.resize(image, dsize=(SIZE, SIZE))
        image = do_center_pad(image, PAD_1, PAD_2)
    else:
        image, mask = do_horizontal_flip2(image, mask)
        # image, mask = do_resize2(image, mask, SIZE, SIZE)
        image, mask = do_center_pad2(image, mask, PAD_1, PAD_2)
        # image, mask = do_center_pad_to_factor2(image, mask, factor=32)

    return image, mask, index, cache


def unaugment_flip(prob):
    # dy0, dy1, dx0, dx1 = compute_center_pad(IMAGE_HEIGHT, IMAGE_WIDTH, factor=32)
    # prob = prob[:, dy0:dy0 + IMAGE_HEIGHT, dx0:dx0 + IMAGE_WIDTH]
    res = []
    for p in prob:
        p = p[Y0:Y1, X0:X1]
        p = p[:, ::-1]
        # p = cv2.resize(p, (101, 101))
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
        # image = cv2.resize(image, dsize=(SIZE, SIZE))
        image = do_center_pad(image, PAD_1, PAD_2)
        # image = do_center_pad_to_factor(image, factor=32)
    else:
        # image, mask = do_resize2(image, mask, SIZE, SIZE)
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


def run_predict(augment):
    if augment == 'null':
        test_augment = augment_null
        test_unaugment = unaugment_null
    if augment == 'flip':
        test_augment = augment_flip
        test_unaugment = unaugment_flip
    # ....................................................


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
    batch_size = 32

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
    net = Net().cuda()

    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

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
    np.save(out_dir + '/test/%s-%s.prob.uint8.npy' % (split, augment), all_prob)
    print(all_prob.shape)

    print('')
    assert (all_num == len(test_loader.sampler))
    all_loss = all_loss / all_num
    print(all_loss)
    log.write('\n')


def run_submit(augment,thres):
    print('running submit')
    if augment in ['null', 'flip']:
        augmentation = [
            1, out_dir + '/test/%s-%s.prob.uint8.npy' % (split, augment),
        ]
        csv_file = out_dir + '/test/%s-%s.csv' % (split, augment)

    if augment == 'aug2':
        augmentation = [
            1, out_dir + '/test/%s-%s.prob.uint8.npy' % (split, 'null'),
            1, out_dir + '/test/%s-%s.prob.uint8.npy' % (split, 'flip'),
        ]
        csv_file = out_dir + '/test/%s-%s.csv' % (split, augment)

        ##---------------------------------------

        # augments, csv_file = ['null','flip'], '/submit1_simple-valid0-300-aug.csv.gz'
        # augments, csv_file = ['flip'], '/submit1_simple-xxx-flip.csv.gz'
        # augments, csv_file = ['null'], '/submit1_simple-xxx-null.csv.gz'

        ##---------------------------------------

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


def run_local_leaderboard(augment):
    # -----------------------------------------------------------------------
    submit_file = out_dir + '/test/%s-%s.csv' % (split, augment)
    dump_dir = out_dir + '/test/%s-%s-dump' % (split, augment)
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


if mode == 'valid':
    for a in ['null', 'flip']:
        print('a=', a)
        run_predict(a)


        #run_submit('aug2')

    for t in np.arange(0.3,0.6,0.01):
        print(t)
        for a in [ 'aug2']:
            print('a=', a)
            run_submit(a,t)
            run_local_leaderboard(a)

if mode == 'test':
    run_predict('null')
    run_predict('flip')
    run_submit('aug2',0.51)
    # run_local_leaderboard()

print('\nsucess!')
