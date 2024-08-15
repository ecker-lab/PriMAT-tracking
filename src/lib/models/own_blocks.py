import numpy as np
import torch
import torch.nn as nn
import random
from models.utils import _tranpose_and_gather_feat


import torchvision
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, emb_dim=3, num_classes=8):
        super().__init__()
        self.conv1 = nn.Conv2d(emb_dim, 32, 3)
        self.conv2 = nn.Conv2d(32, 16, 3)
        #self.conv3 = nn.Conv2d(32, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16, num_classes)
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1,1)) #nn.AvgPool2d(61)
        #self.fc2 = nn.Linear(32 * 30 * 30, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.global_avg_pooling(x)
        #x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        #x = F.relu(self.fc1(x))
        x = self.fc1(x)
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.global_avg_pooling(x)
        #x = torch.flatten(x, 1)
        #x = self.fc2(x)
        return x


class LemurIdentityClassification(nn.Module):
    def __init__(self, num_classes, emb_dim=128):
        super().__init__()

        self.num_classes = num_classes
        #self.cnn = CNN(emb_dim=emb_dim, num_classes=num_classes)
        self.cnn = torchvision.models.resnet18(pretrained=True)

        #num_ftrs = self.cnn.fc.in_features
        #in case you want to extract feature vector from second last layer
        #self.cnn_base = nn.Sequential(*list(self.cnn.children())[:-1])

        #self.classification_layer = nn.Linear(self.cnn.fc.in_features, self.num_classes)
        
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(num_ftrs, self.num_classes)

        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_features, gt_labels=None, bboxes=None, square_bboxes = False, move_px = 100, zoom_min = 0.5, zoom_max = 1.5): 
        """Extract ROI features for ground truth lemur positions

        Args:
            input_features: Features from backbone (B x C x H x W), torch.Tensor
            gt_labels: dictionary with ground truth labels
            bboxes: bounding boxes of lemurs in image, torch.Tensor, (B x 50 x 4)
            square_bboxes: do we want to distort images to be square (False) or cutout square parts of the image (True)
        """
        batch_size, _, _, _ = input_features.size()
        
        #Training
        if gt_labels is not None:

            # Prepare correct format for bboxes ((B * [K, 4]), List )
            bboxes = []

            if not square_bboxes:
                for i in range(batch_size):
                    mask = gt_labels["reg_mask"][i].bool()
                    data = gt_labels["bbox"][i][mask]
                    data = data[gt_labels["box_lemur_class"][i] == 0]
                    
                    # Calculate new quadratic bounding boxes
                    new_bboxes = []
                    for bbox in data:
                        x1, y1, x2, y2 = bbox.tolist()
                        width = x2 - x1
                        height = y2 - y1
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        
                        jitter = random.uniform(zoom_min, zoom_max) # this makes the squared bounding boxes slightly larger or smaller 
                        new_width = width * jitter
                        new_height = height * jitter

                        move = random.uniform(-move_px, move_px)
                        center_x = center_x + move
                        center_y = center_y + move
                        
                        # Calculate new coordinates
                        new_x1 = center_x - new_width / 2
                        new_y1 = center_y - new_height / 2
                        new_x2 = center_x + new_width / 2
                        new_y2 = center_y + new_height / 2

                        
                        new_bbox = [new_x1, new_y1, new_x2, new_y2]
                        new_bboxes.append(new_bbox)
                    
                    bboxes.append(torch.tensor(new_bboxes).cuda())

            else:
                for i in range(batch_size):
                    mask = gt_labels["reg_mask"][i].bool()
                    data = gt_labels["bbox"][i][mask]
                    data = data[gt_labels["box_lemur_class"][i] == 0]
                    
                    # Calculate new quadratic bounding boxes
                    new_bboxes = []
                    for bbox in data:
                        x1, y1, x2, y2 = bbox.tolist()
                        width = x2 - x1
                        height = y2 - y1
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # Determine the longer side
                        max_side = max(width, height)
                        
                        jitter = random.uniform(zoom_min, zoom_max) # this makes the squared bounding boxes slightly larger or smaller 
                        #jitter = 1
                        new_width = max_side * jitter
                        new_height = max_side * jitter

                        #move = 0
                        move = random.uniform(-move_px, move_px)
                        center_x = center_x + move
                        center_y = center_y + move
                        
                        # Calculate new coordinates
                        new_x1 = center_x - new_width / 2
                        new_y1 = center_y - new_height / 2
                        new_x2 = center_x + new_width / 2
                        new_y2 = center_y + new_height / 2

                        
                        new_bbox = [new_x1, new_y1, new_x2, new_y2]
                        new_bboxes.append(new_bbox)
                    
                    bboxes.append(torch.tensor(new_bboxes).cuda())
            
            
        else:

            def convert_to_quadratic(bbox, jitter = 1, move = 0):
                x1, y1, x2, y2 = bbox.tolist()
                width = x2 - x1
                height = y2 - y1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Determine the longer side
                max_side = max(width, height)
                
                #jitter = random.uniform(0.8, 1.0) # this makes the squared bounding boxes slightly larger or smaller 
                jitter = 1
                new_width = max_side * jitter
                new_height = max_side * jitter

                #move = random.uniform(-20, 20)
                move = 0
                center_x = center_x + move
                center_y = center_y + move
                
                # Calculate new coordinates
                new_x1 = center_x - new_width / 2
                new_y1 = center_y - new_height / 2
                new_x2 = center_x + new_width / 2
                new_y2 = center_y + new_height / 2
                return [new_x1, new_y1, new_x2, new_y2]

            # Apply the conversion for each bounding box in the tensor
            new_bboxes = []
            for bbox_list in bboxes:
                new_bbox_list = []
                for bbox in bbox_list:
                    new_bbox_list.append(convert_to_quadratic(bbox))
                new_bboxes.append(torch.tensor(new_bbox_list).cuda())

            bboxes = new_bboxes
        
        # Extract ROI features per ground truth box.
        roi_output = torchvision.ops.roi_pool(
            input_features, [bbox * 4 for bbox in bboxes], output_size=[224,224] 
        )

        # Extract classification feature vector per ROI box.
        #features = self.cnn_base(roi_output)
        #embedding = features.view(features.size(0), -1)
        #class_logits = self.classification_layer(embedding)
        
        embedding = None
        class_logits = self.cnn(roi_output).view(-1, self.num_classes)

        #class_logits = self.cnn_base(roi_output).view(-1, self.num_classes)
        #class_logits = class_logits.view(-1, self.num_classes)

        if gt_labels is None:
            # Compute classification logits.
            class_logits = self.softmax(class_logits)

        return class_logits, embedding
    
    



class Midpoint_Classification(nn.Module):
    def __init__(self, num_cls, clsID4GC=0, emb_dim=128):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_cls = num_cls
        self.clsID4GC = clsID4GC

        self.gc_classifier = nn.Linear(self.emb_dim, self.num_cls, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, gc_features, cls_id_map, target=None, target_ct=None):
        """additional function to forward if gc head is active to give feature vector for monkey detection out of feature map

        Parameters
        ----------
        gc_features : torch.tensor
        # TODO fix
            gc head feature map
        cls_id_map : torch.tensor
            map which entrys of feature map correspond to each specific class ID
        target : torch.tensor / None
            ground truth of classification task / None for inference

        Returns
        -------
        torch.tensor
            prediction of a classification task
        """
        if target is None:
            inds = cls_id_map
            feat = _tranpose_and_gather_feat(gc_features, inds)
            feat = feat.squeeze(0)
        else:
            inds_obj = torch.where(~torch.all(target_ct == 0, dim=2))
            inds = target_ct[inds_obj[0], inds_obj[1]]
 

            if len(inds) == 0:
                return torch.Tensor()
            elif inds[0].shape[0] == 0:
                return torch.Tensor()
            feat = gc_features[inds_obj[0], :, inds[:, 1], inds[:, 0]]

        pred = self.gc_classifier(feat).contiguous()

        if target is None:
            return self.softmax(pred)

#         # catch missing predictions for case that monkey is not in frame
#         pred_fix = torch.zeros(
#             (target.size()[0], self.num_cls), dtype=torch.float32
#         ).to(pred.device)

#         pred_fix[inds[0]] = pred
        return pred


class General_Classification(nn.Module):
    def __init__(self, num_cls, clsID4GC=0, emb_dim=128):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_cls = num_cls
        self.clsID4GC = clsID4GC

        self.gc_classifier = nn.Linear(self.emb_dim, self.num_cls, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, gc_features, cls_id_map, target=None, target_ct=None):
        """additional function to forward if gc head is active to give feature vector for monkey detection out of feature map

        Parameters
        ----------
        gc_features : torch.tensor
        # TODO fix
            gc head feature map
        cls_id_map : torch.tensor
            map which entrys of feature map correspond to each specific class ID
        target : torch.tensor / None
            ground truth of classification task / None for inference

        Returns
        -------
        torch.tensor
            prediction of a classification task
        """
        if target is None:
            inds = cls_id_map
            feat = _tranpose_and_gather_feat(gc_features, inds)
            feat = feat.squeeze(0)
        else:
            inds_obj = torch.where(~torch.all(target_ct == 0, dim=2))
            inds = target_ct[inds_obj[0], inds_obj[1]]
 
            if len(inds) == 0:
                print("aaaaaa")
                return torch.Tensor()
            elif inds[0].shape[0] == 0:
                return torch.Tensor()
            feat = gc_features[inds_obj[0], :, inds[:, 1], inds[:, 0]]

        pred = self.gc_classifier(feat).contiguous()

        if target is None:
            return self.softmax(pred).detach()

#         # catch missing predictions for case that monkey is not in frame
#         pred_fix = torch.zeros(
#             (target.size()[0], self.num_cls), dtype=torch.float32
#         ).to(pred.device)

#         pred_fix[inds[0]] = pred
        return pred