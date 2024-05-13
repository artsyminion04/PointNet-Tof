"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.S3DISDataLoader import TOF_DATASET, TOF_TRAIN
from data_utils.indoor3d_util import g_label2color
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np
import time

#python test_semseg.py --num_point 1596 --log_dir pointnet2_sem_seg --test_area 5 --visual

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['cylinder','box','floor', 'back', 'ceiling', 'side'] 

class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--visual', action='store_true', default=True, help='visualize result [default: False]')
    parser.add_argument('--test_area', type=int, default=5, help='area for testing, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=5, help='aggregate segmentation scores with voting [default: 5]')
    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    visual_dir = experiment_dir + '/visual/test'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data/tof/'
    NUM_CLASSES = 6
    BATCH_SIZE = 1 
    NUM_POINT = args.num_point

    TEST_DATASET = TOF_TRAIN(split='test', num_point=NUM_POINT, num_classes=NUM_CLASSES, block_size=1.0, sample_rate=1.0, transform=None)
    #TOF_DATASET(root, block_points=NUM_POINT, num_classes=NUM_CLASSES,stride=0.005, block_size=0.01) #0.005, 0.09
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False) #, num_workers=10, pin_memory=True, drop_last=True)

    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    model_name = os.listdir('log/sem_seg/pointnet2_sem_seg/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    ''' Evaluate on whole scene'''
    with torch.no_grad():
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        labelweights = np.zeros(NUM_CLASSES)
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        visual = True
        num_episodes = len(testDataLoader)
        #for points, target in testDataLoader:  #in range(num_episodes):
        for i, (points, target) in tqdm(enumerate(testDataLoader), total=num_episodes, smoothing=0.9):
            # print("Inference [%d/%d] %s ..." % (i + 1, num_episodes, f'Testing on Episode {i}'))

            points = points.data.numpy()
            episode_data = points[0]
            # points = TEST_DATASET.cloud_points_list[ep]
            # episode_data = points[0]
            # target = TEST_DATASET.cloud_labels_list[ep]

            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]

            pred_val = np.argmax(pred_val, 2)
            correct = np.sum((pred_val == batch_label))
            total_correct += correct
            total_seen += BATCH_SIZE * NUM_POINT
            tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
            labelweights += tmp

            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum((batch_label == l))
                total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            # VISUALIZATION CODE #
            if visual:
                pred_val = pred_val[0]
                batch_label = batch_label[0]
            
                fout = open(os.path.join(visual_dir, f'ep{i}_pred.obj'), 'w')
                fout_gt = open(os.path.join(visual_dir, f'ep{i}__gt.obj'), 'w')
                filename = os.path.join(visual_dir, f'ep{i}' + '.txt')

                with open(filename, 'w') as pl_save:
                    for i in pred_val:
                        pl_save.write(str(int(i)) + '\n')
                    pl_save.close()
                for i in range(batch_label.shape[0]):
                    color = g_label2color[pred_val[i]]
                    color_gt = g_label2color[batch_label[i]]
                    if visual:
                        fout.write('v %f %f %f %d %d %d\n' % (
                            episode_data[i, 0], episode_data[i, 1], episode_data[i, 2], color[0], color[1],
                            color[2]))
                        fout_gt.write(
                            'v %f %f %f %d %d %d\n' % (
                                episode_data[i, 0], episode_data[i, 1], episode_data[i, 2], color_gt[0],
                                color_gt[1], color_gt[2]))

                fout.close()
                fout_gt.close()
                
                # iou_map = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6)
                # print(iou_map)
                # arr = np.array(total_seen_class)
                # tmp_iou = np.mean(iou_map[arr != 0])
                # log_string('Mean IoU of %s: %.4f' % (f'ep{ep}', tmp_iou))
                # print('----------------------------')

        IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6)
        log_string('test point avg class IoU: %f' % np.mean(IoU))
        log_string('test whole scene point avg class acc: %f' % (
            np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6))))
        log_string('test whole scene point accuracy: %f' % (
            np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))

        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASSES):
            iou_per_class_str += 'class %s, IoU: %.3f \n' % (
            seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
            total_correct_class[l] / float(total_iou_deno_class[l]))

        log_string(iou_per_class_str)

    print("Done Testing!")


if __name__ == '__main__':
    args = parse_args()
    main(args)
