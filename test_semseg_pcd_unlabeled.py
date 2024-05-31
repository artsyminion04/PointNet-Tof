"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.ToFDataLoader import TOF_TEST_REALTIME
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

# python test_semseg_pcd_unlabeled.py --num_point 1596 --log_dir pointnet2_sem_seg

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
    visual_dir = experiment_dir + '/visual/realtime_trained_on_noisy'
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

    TEST_DATASET = TOF_TEST_REALTIME(num_point=NUM_POINT, num_classes=NUM_CLASSES, block_size=1.0, sample_rate=1.0, transform=None)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False)

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
        for i, points in tqdm(enumerate(testDataLoader), total=num_episodes, smoothing=0.9):

            points = points.data.numpy()
            episode_data = points[0]

            points = torch.Tensor(points)
            points = points.float().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            pred_val = np.argmax(pred_val, 2)
           
            total_seen += BATCH_SIZE * NUM_POINT

            # VISUALIZATION CODE #
            if visual:
                pred_val = pred_val[0]
            
                fout = open(os.path.join(visual_dir, f'move_full.obj'), 'w')

                for i in range(pred_val.shape[0]):
                    color = g_label2color[pred_val[i]]
                    if visual:
                        fout.write('v %f %f %f %d %d %d\n' % (
                            episode_data[i, 0], episode_data[i, 1], episode_data[i, 2], color[0], color[1],
                            color[2]))
                
                fout.close()
                
    print("Done Testing!")


if __name__ == '__main__':
    args = parse_args()
    main(args)
