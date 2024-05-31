import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset

# FOR TESTING ON DATA FROM REAL WORLD 
class TOF_TEST_REALTIME(Dataset):
    def __init__(self, num_point=4096, num_classes=6, block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        self.num_classes = num_classes

        self.file = '/home/shaktis/Documents/Pointnet_Pointnet2_pytorch/data/tof/realtime/move_full.npy'
        
        raw_data = np.load(self.file, allow_pickle=True)
        points = raw_data[:, [0,1,2]]

        print(len(points))
   
        self.cloud_points_list = []
        self.cloud_coord_min, self.cloud_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(self.num_classes)

        
        self.cloud_points_list.append(points)

        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights) 
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        
        print("Totally {} samples in set.".format(len(self.cloud_points_list)))

    def __getitem__(self, idx):

        # get points associated with episode
        points = self.cloud_points_list[idx]
        
        N_points = len(points)  #TODO: FIGURE OUT WHAT THIS NUMBER SHOULD BE

        while (True):
            center = points[np.random.choice(N_points)] # generate random sample 
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        # randomly select batch from episode
        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :] 
        current_points = np.zeros((self.num_point, 3)) 

        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        
        current_points = selected_points

        if self.transform is not None:
            current_points = self.transform(current_points)

        return current_points

    def __len__(self):
        return len(self.cloud_points_list)

# FOR TRAINING AND TESTING ON SIM DATA
class TOF_TRAIN(Dataset):
    def __init__(self, split='train', num_point=4096, num_classes=6, block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        self.split = split
        self.num_classes = num_classes

        # load files with data
        if self.split == 'train':
            self.file = '/home/shaktis/Documents/Pointnet_Pointnet2_pytorch/data/tof/noisy_train.npy' 
        else:
            self.file = '/home/shaktis/Documents/Pointnet_Pointnet2_pytorch/data/tof/noisy_val.npy'
        
        raw_data = np.load(self.file, allow_pickle=True)

        points = raw_data[: ,0]
        labels = raw_data[: ,1]
   
        self.cloud_points_list, self.cloud_labels_list = [], []
        self.cloud_coord_min, self.cloud_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(self.num_classes)

        STEPS = 58
        EPISODES = int(len(raw_data)/STEPS)
        for ep in range(EPISODES):
            ep_points, ep_labels = points[ep*STEPS: (ep+1)*STEPS], labels[ep*STEPS: (ep+1)*STEPS]

            ep_points = np.concatenate(ep_points)
            ep_labels = np.concatenate(ep_labels)

            ep_labels[(ep_labels ==6)] = 5 # ceiling should be class 5
            ep_labels = ep_labels[ep_labels >= 0] # get rid of -1's (did not hit objects or box)

            if len(ep_points) != len(ep_labels):
                print(f'ep {ep} points did not match labels: removed')
                continue   

            self.cloud_points_list.append(ep_points)
            self.cloud_labels_list.append(ep_labels)

            tmp, _ = np.histogram(ep_labels, range(self.num_classes + 1))
            labelweights += tmp

            sum_coords = np.sum(ep_points, axis = 1)

            coord_min_ind, coord_max_ind = np.argmin(sum_coords), np.argmax(sum_coords)
            self.cloud_coord_min.append(ep_points[coord_min_ind]), self.cloud_coord_max.append(ep_points[coord_max_ind])

            num_point_all.append(len(ep_labels))

        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights) 
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        
        print("Totally {} samples in {} set.".format(len(self.cloud_points_list), split))

    def __getitem__(self, idx):

        # get points associated with episode
        points = self.cloud_points_list[idx]  
        labels = self.cloud_labels_list[idx]    
        
        N_points = len(points) 

        while (True):
            center = points[np.random.choice(N_points)] # generate random sample 
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        # randomly select batch from episode
        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :] 
        current_points = np.zeros((self.num_point, 3)) 

        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        
        current_points = selected_points
        current_labels = labels[selected_point_idxs]

        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)

        return current_points, current_labels

    def __len__(self):
        return len(self.cloud_points_list)

# # TEST SIM DATA 
# class TOF_DATASET():
#     # prepare to give prediction on each points
#     def __init__(self, root, block_points=4096, num_classes=6, stride=0.5, block_size=1.0, padding=0.001):
#         self.block_points = block_points
#         self.block_size = block_size
#         self.padding = padding
#         self.root = root
#         self.stride = stride
#         self.scene_points_num = []
#         self.num_classes = num_classes

#         self.file = '/home/shaktis/Documents/Pointnet_Pointnet2_pytorch/data/tof/12-19-25.npy'
        
#         raw_data = np.load(self.file, allow_pickle=True)
#         points = raw_data[: ,0]
#         labels = raw_data[: ,1]

#         self.cloud_points_list, self.cloud_labels_list = [], []
#         self.cloud_coord_min, self.cloud_coord_max = [], []

#         STEPS = 58
#         EPISODES = int(len(raw_data)/STEPS)
#         for ep in range(EPISODES):
#             ep_points, ep_labels = points[ep*STEPS: (ep+1)*STEPS], labels[ep*STEPS: (ep+1)*STEPS]

#             ep_points = np.concatenate(ep_points)
#             ep_labels = np.concatenate(ep_labels)

#             ep_labels[(ep_labels ==6)] = 5 # ceiling should be class 5
#             ep_labels = ep_labels[ep_labels >= 0] # get rid of -1's (did not hit objects or box)

#             if len(ep_points) != len(ep_labels):
#                 print(f'ep {ep} points did not match labels: removed')
#                 continue

#             self.cloud_points_list.append(ep_points)
#             self.cloud_labels_list.append(ep_labels)

#             sum_coords = np.sum(ep_points, axis = 1)

#             coord_min_ind, coord_max_ind = np.argmin(sum_coords), np.argmax(sum_coords)
#             self.cloud_coord_min.append(ep_points[coord_min_ind]), self.cloud_coord_max.append(ep_points[coord_max_ind])
        
#         assert len(self.cloud_points_list) == len(self.cloud_labels_list)

#         labelweights = np.zeros(self.num_classes)
#         for seg in self.cloud_labels_list:
#             tmp, _ = np.histogram(seg, range(self.num_classes + 1))
#             self.scene_points_num.append(seg.shape[0])
#             labelweights += tmp 
#         labelweights = labelweights.astype(np.float32)
#         labelweights = labelweights / np.sum(labelweights)
#         self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

#     def __getitem__(self, idx):

#         points = self.cloud_points_list[idx]
#         labels = self.cloud_labels_list[idx] 

#         sum_coords = np.sum(points, axis = 1)
#         coord_min_ind, coord_max_ind = np.argmin(sum_coords), np.argmax(sum_coords)
#         coord_min = points[coord_min_ind]
#         coord_max = points[coord_max_ind]

#         grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
#         grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
#         data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]),  np.array([])
        
#         print(f'grid_x:{grid_x}  grid_y:{grid_y}')
        
#         for index_y in range(0, grid_y):
#             for index_x in range(0, grid_x):
#                 s_x = coord_min[0] + index_x * self.stride
#                 e_x = min(s_x + self.block_size, coord_max[0])
#                 s_x = e_x - self.block_size
#                 s_y = coord_min[1] + index_y * self.stride
#                 e_y = min(s_y + self.block_size, coord_max[1])
#                 s_y = e_y - self.block_size
#                 point_idxs = np.where(
#                     (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (points[:, 1] >= s_y - self.padding) & (
#                                 points[:, 1] <= e_y + self.padding))[0]
#                 if point_idxs.size == 0:
#                     continue
                
#                 # randomly get batch of data
#                 num_batch = int(np.ceil(point_idxs.size / self.block_points))
#                 point_size = int(num_batch * self.block_points)
#                 replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
#                 point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
#                 point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
#                 np.random.shuffle(point_idxs)
#                 data_batch = points[point_idxs, :]

#                 # normalize points
#                 normlized_xyz = np.zeros((point_size, 3))
#                 normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
#                 normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
#                 normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]

#                 data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
#                 data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)

#                 data_batch = normlized_xyz #np.concatenate((data_batch, normlized_xyz), axis=1) # why concatenate instead of replace
#                 label_batch = labels[point_idxs].astype(int)
#                 batch_weight = self.labelweights[label_batch]

#                 data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
#                 label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
#                 sample_weight = np.hstack([sample_weight, batch_weight]) if label_room.size else batch_weight
#                 index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
        
#         data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))
#         label_room = label_room.reshape((-1, self.block_points))
#         sample_weight = sample_weight.reshape((-1, self.block_points))
#         index_room = index_room.reshape((-1, self.block_points))
        
#         return data_room, label_room, sample_weight, index_room
    
#     def __len__(self):
#         return len(self.cloud_points_list)
