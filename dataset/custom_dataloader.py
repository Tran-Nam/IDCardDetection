import os
import numpy as np 
import pandas as pd 
import cv2 
from albumentations import HorizontalFlip, VerticalFlip, Compose, Transpose, \
                        RandomRotate90, IAAAdditiveGaussianNoise, \
                        RandomGamma, RandomBrightnessContrast, CLAHE, OneOf, \
                        IAASharpen, Blur, MotionBlur, RandomContrast, HueSaturationValue

import sys 
sys.path.append('..')
from data.processing import gen_groundtruth

def augmentation():
    transform = [
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        IAAAdditiveGaussianNoise(p=0.2),
        OneOf([
            RandomGamma(p=1),
            RandomBrightnessContrast(p=1),
            CLAHE(p=1)
        ], p=0.9),
        OneOf([
            RandomRotate90(p=1),
            Transpose(p=1)
        ], p=0.3),
        OneOf([
            IAASharpen(p=1),
            Blur(blur_limit=3, p=1),
            MotionBlur(blur_limit=3, p=1)
        ], p=0.8),
        OneOf([
            RandomContrast(p=1),
            HueSaturationValue(p=1)
        ], p=0.8)
    ]
    return Compose(transform)

class Dataloader():
    def __init__(self, 
        image_dir, 
        label_path,
        batch_size, 
        augmentation=None,
        shuffle=None):
        self._image_dir = image_dir
        self._augmentation = augmentation
        self._shuffle = shuffle
        self._label = pd.read_csv(label_path)
        self._im_paths = np.array(list(set(self._label['filename'])))
        self._num_samples = len(self._im_paths)
        self._batch_size = batch_size
        self._step_per_epoch = self._num_samples // self._batch_size
        self._batch_id = 0

    def _process_image(self, im_path):
        im = cv2.imread(os.path.join(self._image_dir, im_path))
        rows_label = self._label[self._label['filename']==im_path]
        list_pts = rows_label.loc[:, 'tlx': 'bly'].values
        list_pts = list_pts.reshape(-1, 4, 2)
        data = gen_groundtruth(im, list_pts) 
        if self._augmentation:
            data = self._augmentation(**data)
        # out = {
        #     'image': im,
        #     'groundtruth': groundtruth
        # }
        return data

    def generator(self):
        while True:
            start = self._batch_id * self._batch_size
            end = start + self._batch_size
            self._batch_id += 1

            if end + self._batch_size > self._num_samples: # last batch
                end = self._num_samples
            im_paths = self._im_paths[start: end]
            # TODO: process here

            if end == self._num_samples: # end epoch
                self._batch_id = 0
                if self._shuffle:
                    indices = np.arange(self._num_samples)
                    np.random.seed(2020)
                    np.random.shuffle(indices)
                    self._im_paths = self._im_paths[indices]
            batch_im = np.empty((0, 512, 512, 3))
            batch_gt = np.empty((0, 128, 128, 15))
            for im_path in im_paths:
                im = cv2.imread(im)
                data_image_once = self._process_image(im_path)
                batch_im = np.concatenate((batch_im, data_image_once['image']), axis=0)
                batch_gt = np.concatenate((batch_gt, data_image_once['mask']), axis=0)
            print(batch_im.shape, batch_gt.shape)
            yield batch_im, batch_gt
    
    def next_batch(self):
        return next(self.generator())

"""
class TrainDataloader():
    def __init__(self, 
        image_dir, 
        label_path,
        batch_size):
        self._image_dir = image_dir
        self._label = pd.read_csv(label_path)
        self._im_paths = np.array(list(set(self._label['filename'])))
        self._num_samples = len(self._im_paths)
        self._batch_size = batch_size
        self._batch_id = 0

    def _process_image(self, im_path):
        im = cv2.imread(os.path.join(self._image_dir, im_path))
        rows_label = self._label[self._label['filename']==im_path]
        list_pts = rows_label.loc[:, 'tlx': 'bly'].values
        list_pts = list_pts.reshape(-1, 4, 2)
        groundtruth = gen_groundtruth(im, list_pts) 
        out = {
            'image': im,
            'groundtruth': groundtruth
        }
        return out

    def generator(self):
        while True:
            start = self._batch_id * self._batch_size
            end = start + self._batch_size
            self._batch_id += 1

            if end + self._batch_size > self._num_samples: # last batch
                end = self._num_samples
            im_paths = self._im_paths[start: end]
            # TODO: process here

            if end == self._num_samples: # end epoch
                self._batch_id = 0
                indices = np.arange(self._num_samples)
                np.random.seed(2020)
                np.random.shuffle(indices)
                self._im_paths = self._im_paths[indices]
            # data = np.empty((0, 512, 512, 3))
            # label = np.empty((0, 128, 128, 12))
            # for im_path in im_paths:
            #     im = cv2.imread(im)
            #     groundtruth_image_once = process_image(im_path)
            #     data = np.vstack((data, groundtruth_image_once['image']))
            #     label = np.vstack((label, groundtruth_image_once['groundtruth']))
            # print(data.shape, label.shape)
            yield im_paths
    
    def next_batch(self):
        return next(self.generator())

class ValDataloader():
    def __init__(self, 
        image_dir, 
        label_path,
        batch_size):
        self._image_dir = image_dir
        self._label = pd.read_csv(label_path)
        self._im_paths = np.array(list(set(self._label['filename'])))
        self._num_samples = len(self._im_paths)
        self._batch_size = batch_size
        self._steps_per_eval = self._num_samples // self._batch_size
        self._batch_id = 0

    def _process_image(self, im_path):
        im = cv2.imread(os.path.join(self._image_dir, im_path))
        rows_label = self._label[self._label['filename']==im_path]
        list_pts = rows_label.loc[:, 'tlx': 'bly'].values
        list_pts = list_pts.reshape(-1, 4, 2)
        groundtruth = gen_groundtruth(im, list_pts) 
        out = {
            'image': im,
            'groundtruth': groundtruth
        }
        return out

    def generator(self):
        while True:
            start = self._batch_id * self._batch_size
            end = start + self._batch_size
            self._batch_id += 1

            if end + self._batch_size > self._num_samples: # last batch
                end = self._num_samples
            im_paths = self._im_paths[start: end]
            # TODO: process here

            if end == self._num_samples: # end epoch
                self._batch_id = 0
            # data = np.empty((0, 512, 512, 3))
            # label = np.empty((0, 128, 128, 12))
            # for im_path in im_paths:
            #     im = cv2.imread(im)
            #     groundtruth_image_once = process_image(im_path)
            #     data = np.vstack((data, groundtruth_image_once['image']))
            #     label = np.vstack((label, groundtruth_image_once['groundtruth']))
            # print(data.shape, label.shape)
            yield im_paths

    def next_batch(self):
        return next(self.generator())
"""

