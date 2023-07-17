import copy
import glob
import math
import os
import os.path as osp
import random
import time

from collections import OrderedDict, defaultdict

import cv2
import numpy as np
import torch
from utils.image import draw_msra_gaussian, draw_umich_gaussian, gaussian_radius, random_affine, letterbox
from utils.utils import xywh2xyxy, xyxy2xywh


class LoadImages:  # for inference
    def __init__(self, path, img_size=(1088, 608)):
        if os.path.isdir(path):
            image_format = ['.jpg', '.jpeg', '.png', '.tif']
            self.files = sorted(glob.glob('%s/*.*' % path))
            self.files = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_format, self.files))
        elif os.path.isfile(path):
            self.files = [path]

        self.nF = len(self.files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        assert self.nF > 0, 'No images found in ' + path

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nF:
            raise StopIteration
        img_path = self.files[self.count]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        #assert img0 is not None, 'Failed to load ' + img_path
        if img0 is None:
            raise StopIteration
        
        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return img_path, img, img0

    def __getitem__(self, idx):
        idx = idx % self.nF
        img_path = self.files[idx]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img_path, img, img0

    def __len__(self):
        return self.nF  # number of files
    
    

class LoadImagesAndBoxes:  # for inference
    def __init__(self, root, path, valset = "MacaquePose", img_size=(1088, 608)):
        #if os.path.isdir(path):
        #    image_format = ['.jpg', '.jpeg', '.png', '.tif']
        #    self.files = sorted(glob.glob('%s/*.*' % path))
        #   self.files = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_format, self.files))
        #elif os.path.isfile(path):
        #    self.files = [path]
        
        
        with open(path, 'r') as file:
            self.files = file.readlines()
            
            
            
            # for each line of one file: 1. build complete path 2. strip '\n' character 3. put back into list -> at position 'ds' are all images from one of these list files
            self.files = [osp.join(root, x.strip()) for x in self.files]
            # get rid of empty lines
            self.files = list(filter(lambda x: len(x) > 0, self.files))
            self.files = [x.replace("MacaquePose", valset) for x in self.files]

            
            
            
        self.label_files = [
                x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt').replace('.JPG', '.txt')
                for x in self.files]

        self.nF = len(self.files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        assert self.nF > 0, 'No images found in ' + path

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nF:
            raise StopIteration
        img_path = self.files[self.count]
        label_path = self.label_files[self.count]
        
        labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        #assert img0 is not None, 'Failed to load ' + img_path
        if img0 is None:
            raise StopIteration
        
        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return img_path, img, img0, labels0

    def __getitem__(self, idx):
        idx = idx % self.nF
        img_path = self.files[idx]
        label_path = self.label_files[idx]
        
        labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img_path, img, img0, labels0

    def __len__(self):
        return self.nF  # number of files


class LoadVideo:  # for inference
    def __init__(self, path, img_size=(1088, 608)):# FIXME these values are approximate downsize shape of image to input into net
        self.cap = cv2.VideoCapture(path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        # FIXME should be real image size! of input
        # self.w, self.h = 1920, 1080 # before 1920, 1080
        self.w, self.h = 1024, 768
        #self.w, self.h = 608, 1088
        print('Length of the video: {:d} frames'.format(self.vn))

    def get_size(self, vw, vh, dw, dh):
        wa, ha = float(dw) / vw, float(dh) / vh
        a = min(wa, ha)
        return int(vw * a), int(vh * a)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == len(self):
            raise StopIteration
        # Read image
        res, img0 = self.cap.read()  # BGR
        #assert img0 is not None, 'Failed to load frame {:d}'.format(self.count)
        if img0 is None:
            raise StopIteration
        # FIXME resize doesnt make sense for me, just let the image in original size 
        # img0 = cv2.resize(img0, (self.w, self.h))

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return self.count, img, img0

    def __len__(self):
        return self.vn  # number of files


class LoadImagesAndLabels:  # for training

    def get_data(self, img_path, label_path, with_gc = False, aug_hsv = True):

        height = self.height
        width = self.width
        img = cv2.imread(img_path)  # BGR
        if img is None:
            raise ValueError('File corrupt {}'.format(img_path))
        augment_hsv = aug_hsv
        if self.augment and augment_hsv:
            # SV augmentation by 50%
            fraction = 0.50
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)

            a = (random.random() * 2 - 1) * fraction + 1
            S *= a
            if a > 1:
                np.clip(S, a_min=0, a_max=255, out=S)

            a = (random.random() * 2 - 1) * fraction + 1
            V *= a
            if a > 1:
                np.clip(V, a_min=0, a_max=255, out=V)

            img_hsv[:, :, 1] = S.astype(np.uint8)
            img_hsv[:, :, 2] = V.astype(np.uint8)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
            

        h, w, _ = img.shape
        img, ratio, padw, padh = letterbox(img, height=height, width=width)

        # Load labels
        if os.path.isfile(label_path):
            
            if with_gc:
                labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 7)
            else:
                labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)
            # TODO fix
            # labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 7)

            # Normalized xywh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 2] = ratio * w * (labels0[:, 2] - labels0[:, 4] / 2) + padw
            labels[:, 3] = ratio * h * (labels0[:, 3] - labels0[:, 5] / 2) + padh
            labels[:, 4] = ratio * w * (labels0[:, 2] + labels0[:, 4] / 2) + padw
            labels[:, 5] = ratio * h * (labels0[:, 3] + labels0[:, 5] / 2) + padh
        else:
            labels = np.array([])

        # Augment image and labels
        if self.augment:
            img, labels, M = random_affine(img, labels, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.50, 1.20))

        plotFlag = False
        if plotFlag:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure(figsize=(50, 50))
            plt.imshow(img[:, :, ::-1])
            plt.plot(labels[:, [1, 3, 3, 1, 1]].T, labels[:, [2, 2, 4, 4, 2]].T, '.-')
            plt.axis('off')
            plt.savefig('test.jpg')
            time.sleep(10)

        nL = len(labels)
        if nL > 0:
            # convert xyxy to xywh
            labels[:, 2:6] = xyxy2xywh(labels[:, 2:6].copy())  # / height
            labels[:, 2] /= width
            labels[:, 3] /= height
            labels[:, 4] /= width
            labels[:, 5] /= height
        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip & (random.random() > 0.5):
                img = np.fliplr(img)
                if nL > 0:
                    labels[:, 2] = 1 - labels[:, 2]

        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB

        if self.transforms is not None:
            img = self.transforms(img)

        return img, labels, img_path, (h, w)

    def __len__(self):
        return self.nF  # number of batches


class JointDataset(LoadImagesAndLabels):  # for training
    default_resolution = [1088, 608]
    mean = None
    std = None

    def __init__(self, opt, root, paths, img_size=(1088, 608), augment=False, transforms=None):
        self.opt = opt
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()

        if self.opt.use_gc:
            self.gc_labels = OrderedDict()

        self.class_names = opt.reid_cls_names.split(',')
        self.num_classes = len(self.class_names)
        self.aug_hsv = opt.no_aug_hsv

        if self.opt.use_gc:
            self.gc_cls_names = opt.gc_cls_names.split(',')
            self.num_gc_cls = len(self.gc_cls_names)


        if self.opt.cat_spec_wh:
            self.wh_classes = self.num_classes
        else:
            self.wh_classes = 1

        for ds, path in paths.items():
            with open(path, 'r') as file:
                # each 'ds' corresponds to one image/label file
                self.img_files[ds] = file.readlines()
                # for each line of one file: 1. build complete path 2. strip '\n' character 3. put back into list -> at position 'ds' are all images from one of these list files
                self.img_files[ds] = [osp.join(root, x.strip()) for x in self.img_files[ds]]
                # get rid of empty lines
                self.img_files[ds] = list(filter(lambda x: len(x) > 0, self.img_files[ds]))

            self.label_files[ds] = [
                x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt').replace('.JPG', '.txt')
                for x in self.img_files[ds]]


            # added read in of pose labels
            # if self.opt.use_gc:
            #     with open(path.replace('.train','-pose.train').replace('.val','-pose.val'), 'r') as file:
            #         self.gc_labels[ds] = [x.rstrip() for x in file.readlines()]
            #         self.gc_labels[ds] = list(filter(lambda x: len(x) > 0, self.gc_labels[ds]))

        # read in GT labels
        # for each file of directories
        for ds, label_paths in self.label_files.items():
            # max_index = -1
            max_ids_dict = defaultdict(int)
            # for each label file
            for lp in label_paths:
                lb = np.loadtxt(lp)
                if len(lb) < 1:
                    continue
                # if len(lb.shape) < 2:
                #     img_max = lb[1]
                # else:
                #     img_max = np.max(lb[:, 1])
                # if img_max > max_index:
                #     max_index = img_max
                if self.opt.use_gc:
                    lb = lb.reshape(-1, 7)
                else:
                    lb = lb.reshape(-1, 6)

                for item in lb:
                    if item[1] > max_ids_dict[int(item[0])]:  # item[0]: cls_id, item[1]: track id
                        max_ids_dict[int(item[0])] = item[1]
            # track id number
            self.tid_num[ds] = max_ids_dict

        self.tid_start_idx_of_cls_ids = defaultdict(dict)
        last_idx_dict = defaultdict(int)
        for k, v in self.tid_num.items():
            for cls_id, id_num in v.items():
                self.tid_start_idx_of_cls_ids[k][cls_id] = last_idx_dict[cls_id]
                last_idx_dict[cls_id] += id_num

        self.nID_dict = defaultdict(int)
        for k, v in last_idx_dict.items():
            self.nID_dict[k] = int(v)
        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)
        self.width = img_size[0]
        self.height = img_size[1]
        self.max_objs = opt.K
        self.augment = augment
        self.transforms = transforms

        print('=' * 80)
        print('dataset summary')
        print(self.tid_num)
        # print('total # identities:', self.nID)
        print('start index')
        print(self.tid_start_index)
        print('=' * 80)

    def __getitem__(self, files_index):

        for i, c in enumerate(self.cds):
            if files_index >= c:
                ds = list(self.label_files.keys())[i]
                start_index = c

        img_path = self.img_files[ds][files_index - start_index]
        label_path = self.label_files[ds][files_index - start_index]

        # added for gc
        # if self.opt.use_gc:
        #     classify_cls = self.gc_labels[ds][files_index - start_index]
        #     classify_cls = torch.tensor(int(classify_cls))

        imgs, labels, img_path, (input_h, input_w) = self.get_data(img_path, label_path, with_gc = self.opt.use_gc, aug_hsv = self.aug_hsv)
        for i, _ in enumerate(labels):
            if labels[i, 1] > -1:
                cls_id = int(labels[i][0])
                start_idx = self.tid_start_idx_of_cls_ids[ds][cls_id]
                labels[i, 1] += start_idx

        output_h = imgs.shape[1] // self.opt.down_ratio
        output_w = imgs.shape[2] // self.opt.down_ratio
        # num_classes = self.num_classes
        num_objs = labels.shape[0]
        hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        if self.opt.ltrb:
            wh = np.zeros((self.max_objs, 4 * self.wh_classes), dtype=np.float32)
        else:
            wh = np.zeros((self.max_objs, 2 * self.wh_classes), dtype=np.float32)# mcmot uses this one, without if/else
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs, ), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs, ), dtype=np.uint8)
        ids = np.zeros((self.max_objs, ), dtype=np.int64)
        cls_tr_ids = np.zeros((self.num_classes, output_h, output_w), dtype=np.int64)
        cls_id_map = np.full((1, output_h, output_w), -1, dtype=np.int64)
        # bbox_xys = np.zeros((self.max_objs, 4), dtype=np.float32)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian
        for k in range(num_objs):
            label = labels[k]
            # bbox = label[2:]
            # TODO clean up, doesnt need to be in for loop, richards code for if label in same gt file as rest
            bbox = label[2:6]
            if self.opt.use_gc:
                classify_cls = torch.tensor(int(label[6])) #FIXME: don't overwrite but store in vector
            #
            cls_id = int(label[0])
            bbox[[0, 2]] = bbox[[0, 2]] * output_w
            bbox[[1, 3]] = bbox[[1, 3]] * output_h
            bbox_amodal = copy.deepcopy(bbox)
            bbox_amodal[0] = bbox_amodal[0] - bbox_amodal[2] / 2.
            bbox_amodal[1] = bbox_amodal[1] - bbox_amodal[3] / 2.
            bbox_amodal[2] = bbox_amodal[0] + bbox_amodal[2]
            bbox_amodal[3] = bbox_amodal[1] + bbox_amodal[3]
            bbox[0] = np.clip(bbox[0], 0, output_w - 1)
            bbox[1] = np.clip(bbox[1], 0, output_h - 1)
            h = bbox[3]
            w = bbox[2]

            bbox_xy = copy.deepcopy(bbox)
            bbox_xy[0] = bbox_xy[0] - bbox_xy[2] / 2
            bbox_xy[1] = bbox_xy[1] - bbox_xy[3] / 2
            bbox_xy[2] = bbox_xy[0] + bbox_xy[2]
            bbox_xy[3] = bbox_xy[1] + bbox_xy[3]

            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                # radius = 6 if self.opt.mse_loss else radius
                radius = self.opt.hm_gauss if self.opt.mse_loss else radius
                #radius = max(1, int(radius)) if self.opt.mse_loss else radius
                ct = np.array(
                    [bbox[0], bbox[1]], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                if self.opt.ltrb:
                    wh[k, (cls_id*2,cls_id*2+1) if self.wh_classes > 1 else (0,1)] = ct[0] - bbox_amodal[0], ct[1] - bbox_amodal[1], \
                            bbox_amodal[2] - ct[0], bbox_amodal[3] - ct[1]
                else:# only else for mcmot
                    wh[k, (cls_id*2,cls_id*2+1) if self.wh_classes > 1 else (0,1)] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                # ids[k] = label[1]
                # output feature map
                cls_id_map[0][ct_int[1], ct_int[0]] = cls_id
                # track ids
                cls_tr_ids[cls_id][ct_int[1]][ct_int[0]] = label[1] - 1
                # track id -1
                ids[k] = label[1] - 1

                # bbox_xys[k] = bbox_xy

            # gc_labels = self.gc_labels[k][files_index]
            
        #FIXME for DEREK
        # pose = torch.tensor(labels[:,1], dtype=int)

        if self.opt.use_gc:
            ret = {'input': imgs, 'hm': hm, 'reg': reg, 'wh': wh, 'ind': ind, 'reg_mask': reg_mask, 'gc': classify_cls, 'ids': ids, 'cls_id_map': cls_id_map, 'cls_tr_ids': cls_tr_ids}#'bbox': bbox_xys}
        else:
            ret = {'input': imgs, 'hm': hm, 'reg': reg, 'wh': wh, 'ind': ind, 'reg_mask': reg_mask, 'ids': ids, 'cls_id_map': cls_id_map, 'cls_tr_ids': cls_tr_ids}#'bbox': bbox_xys}
        return ret




class DetDataset(LoadImagesAndLabels):  # for training
    def __init__(self, root, paths, img_size=(1088, 608), augment=False, transforms=None):

        dataset_names = paths.keys()
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        for ds, path in paths.items():
            with open(path, 'r') as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [osp.join(root, x.strip()) for x in self.img_files[ds]]
                self.img_files[ds] = list(filter(lambda x: len(x) > 0, self.img_files[ds]))

            self.label_files[ds] = [
                x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                for x in self.img_files[ds]]

        for ds, label_paths in self.label_files.items():
            max_index = -1
            for lp in label_paths:
                lb = np.loadtxt(lp)
                if len(lb) < 1:
                    continue
                if len(lb.shape) < 2:
                    img_max = lb[1]
                else:
                    img_max = np.max(lb[:, 1])
                if img_max > max_index:
                    max_index = img_max
            self.tid_num[ds] = max_index + 1

        last_index = 0
        for i, (k, v) in enumerate(self.tid_num.items()):
            self.tid_start_index[k] = last_index
            last_index += v

        self.nID = int(last_index + 1)
        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)
        self.width = img_size[0]
        self.height = img_size[1]
        self.augment = augment
        self.transforms = transforms

        print('=' * 80)
        print('dataset summary')
        print(self.tid_num)
        print('total # identities:', self.nID)
        print('start index')
        print(self.tid_start_index)
        print('=' * 80)

    def __getitem__(self, files_index):

        for i, c in enumerate(self.cds):
            if files_index >= c:
                ds = list(self.label_files.keys())[i]
                start_index = c

        img_path = self.img_files[ds][files_index - start_index]
        label_path = self.label_files[ds][files_index - start_index]
        if os.path.isfile(label_path):
            # if else needed for with without gc head
            
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)
            # TODO also for richards change
            # labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 7)

        imgs, labels, img_path, (h, w) = self.get_data(img_path, label_path)
        for i, _ in enumerate(labels):
            if labels[i, 1] > -1:
                labels[i, 1] += self.tid_start_index[ds]

        return imgs, labels0, img_path, (h, w)


def collate_fn(batch):
    imgs, labels, paths, sizes = zip(*batch)
    batch_size = len(labels)
    imgs = torch.stack(imgs, 0)
    max_box_len = max([l.shape[0] for l in labels])
    labels = [torch.from_numpy(l) for l in labels]
    filled_labels = torch.zeros(batch_size, max_box_len, 6)
    labels_len = torch.zeros(batch_size)

    for i in range(batch_size):
        isize = labels[i].shape[0]
        if len(labels[i]) > 0:
            filled_labels[i, :isize, :] = labels[i]
        labels_len[i] = isize

    return imgs, filled_labels, paths, sizes, labels_len.unsqueeze(1)