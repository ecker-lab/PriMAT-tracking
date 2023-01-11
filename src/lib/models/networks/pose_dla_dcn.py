from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join

import torch
from torch import int16, nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from dcn_v2 import DCN


from models.decode import mot_decode
from models.utils import _tranpose_and_gather_feat



BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights)
        # self.fc = fc


def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])  
            proj = DeformConv(c, o)
            node = DeformConv(o, o)
     
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
                 
        
    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])



class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x


class DLASeg(nn.Module):
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0, num_classes=5, num_poses=5, cat_spec_wh=True,
                 clsID4Pose=0, conf_thres=0.02):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = globals()[base_name](pretrained=pretrained)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], 
                            [2 ** i for i in range(self.last_level - self.first_level)])
        
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            # if 'mpc' in head:
            #     # in shape [4, 60, 152, 272] -> [batch, channel, (image_size)]
            #     in_size = channels[self.first_level]*152*272
            #     out_size = classes
            #     fc = nn.Sequential(
            #         nn.Flatten(),
            #         nn.Linear(in_size, out_size, bias=True)

            #         # 2,480,640 -> 1,240,320
            #         nn.Linear(in_size, in_size//2, bias=True)
            #         # 1,240,320 -> 620,160
            #         nn.Linear(in_size//2, in_size//4, bias=True),
            #         # 620,160 -> 310,080
            #         nn.Linear(in_size//4, in_size//8, bias=True),
            #         # 310,080 -> 5
            #         nn.Linear(in_size//8, out_size, bias=True)

            #         nn.AvgPool2d(kernel_size=3, stride=2, padding=0),
            #         nn.Linear(, out_size, bias=True)
            #     if not self.training:
            #         fc.append(nn.Sigmoid())

            # elif head_conv > 0:
            if head_conv > 0:
              fc = nn.Sequential(
                  nn.Conv2d(channels[self.first_level], head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))
              if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(channels[self.first_level], classes, 
                  kernel_size=final_kernel, stride=1, 
                  padding=final_kernel // 2, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)


        # new stuff for pose head --richard
        # -----------------------------------------
        if 'mpc' in self.heads:
            self.K = 50 # number of detections per frame
            # self.conf_thres = 0.02 # confidence threshold for heatmap detections
            self.nCls = self.heads['mpc']
            self.emb_scale = math.sqrt(2) * math.log(self.nCls - 1)
            self.emb_dim = 128
            # self.classifier = nn.Linear(self.emb_dim, self.nID)
            # --mine
            # self.num_classes = num_classes
            self.num_poses = num_poses
            self.pose_classifier = nn.Linear(self.emb_dim, self.num_poses, bias=True)
            # self.cat_spec_wh = cat_spec_wh
            self.clsID4Pose = clsID4Pose
            # self.conf_thres = conf_thres
            # self.MONKEY = 0
            # self.pose_classifier = nn.Sequential(
            #         nn.Linear(self.emb_dim, self.num_poses, bias=True))
            self.sm = nn.Softmax(dim=1)
            # -----------------------------------------


    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])

        return [z]  

    def pose_vec(self, mpc, cls_id_map, target=None):
        """ additional function to forward if mpc head is active to give feature vector for monkey detection out of feature map
        
        Parameters
        ----------
        mpc : torch.tensor
            monkey pose head feature map
        cls_id_map : torch.tensor
            map which entrys of feature map correspond to each specific class ID
        target : torch.tensor / None
            ground truth of monkey pose / None for inference

        Returns
        -------
        torch.tensor
            prediction of monkey pose
        """
        if target is None:
            inds = cls_id_map
            feat = _tranpose_and_gather_feat(mpc, inds)
            feat = feat.squeeze(0)
        else:
            inds = torch.where(cls_id_map == self.clsID4Pose)
            if inds[0].shape[0] == 0:
                stand_in = torch.tensor([0, 0, 0, 0, 1]).expand(target.numel(), 5).to(target)
                return stand_in
                #FIXME could work for Richard!
                # return torch.zeros_like(target)
            feat = mpc[inds[0],:,inds[2],inds[3]]
            # feat = self.emb_scale * F.normalize(feat)

        pred = self.pose_classifier(feat).contiguous()
        
        if target is None:
            return self.sm(pred).detach()
        
        # catch missing predictions for case that monkey is not in frame
        pred_fix = torch.zeros(target.size()[0], self.num_poses).to(pred.device)
        pred_fix += torch.tensor([0, 0, 0, 0, 1]).to(pred.device)
        pred_fix[inds[0]] = pred
        return pred_fix
    
#         # new stuff for pose head
#         hm = z['hm'].clone().sigmoid_()
#         wh = z['wh']
#         pose_feature = z['mpc']


#         reg = z['reg']
        
#         # hm: [1,5,152,272] wh: [1,2,152,272] reg: [1,2,152,272] num_cls: 5 cat_spec_wh: False K: 50
#         dets, inds, cls_inds_mask = mot_decode(hm, wh, reg=reg, num_classes=self.num_classes, cat_spec_wh=self.cat_spec_wh, K=self.K)

        
#         cls_id_feature = collect_pose_feature(z, pose_feature, dets, inds, cls_inds_mask, self.clsID4Pose, self.num_classes, self.conf_thres)
     
     
#         # if no monkey detected return NiS
#         if cls_id_feature.size(0) == 0:
#             z['pose'] = torch.tensor([[0.,0.,0.,0.,1.]])
#             if hm.size(0) > 1:
#                 z['pose'] = z['pose'].expand(hm.size(0), -1)
#         else:
#             z['pose'] = self.pose_classifier(cls_id_feature).contiguous()
     
#         if not self.training:
#                 z['pose'] = self.sm(z['pose'])  


    
    
# def collect_pose_feature(z, pose_feature, dets, inds, cls_inds_mask, clsID4Pose, num_classes, conf_thres):
#     # ----- get pose feature vector by object class
#     # get inds of MONKEY object class
#     cls_inds = inds[:,cls_inds_mask[clsID4Pose]]
    
#     # no monkey detections?
#     if cls_inds.numel() == 0:
#         return torch.tensor([])
    
#     # gather feats for each object class
#     # pose_feature: [1,128,152,272]
#     cls_id_feature = _tranpose_and_gather_feat(pose_feature, cls_inds)  # inds: 1×128
#     cls_id_feature = cls_id_feature.squeeze(0)  # n × FeatDim

#     dets = map2origCLEANUP(dets, num_classes)
#     # cls_id_feature = self.emb_scale * F.normalize(cls_id_feature, dim=1)

#     # filter out low confidence and non-monkey detections
#     cls_dets = dets[clsID4Pose]
#     remain_inds = cls_dets[:, 4] > conf_thres
    
#     # only keep one per batch
#     if cls_dets.size(0) > 1:
#         _, remain_inds = torch.max(cls_dets[:, 4], dim=0)
#         remain_inds = remain_inds.reshape(1,)
#     cls_dets = cls_dets[remain_inds]
#     cls_id_feature = cls_id_feature[remain_inds]

#     return cls_id_feature

    
# def map2origCLEANUP(dets, num_classes):
#     """
#     :param dets:
#     :param num_classes:
#     :return: dict of detections(key: cls_id)
#     """

#     dets = dets.reshape(1, -1, dets.size(2))  # default: 1×128×6
#     dets = dets[0]  # 128×6

#     dets_dict = {}

#     classes = dets[:, -1]
#     for cls_id in range(num_classes):
#         inds = (classes == cls_id)
#         dets_dict[cls_id] = dets[inds, :]

#     return dets_dict


def get_pose_net(num_layers, heads, head_conv=256, down_ratio=4, num_classes=5, num_poses=5, cat_spec_wh=True, clsID4Pose=0, conf_thres=0.02):
  model = DLASeg('dla{}'.format(num_layers), heads,
                 pretrained=True,
                 down_ratio=down_ratio,
                 final_kernel=1,
                 last_level=5,
                 head_conv=head_conv,
                 num_classes=num_classes,
                 num_poses=num_poses,
                 cat_spec_wh=cat_spec_wh,
                 clsID4Pose=clsID4Pose,
                 conf_thres=conf_thres)
  return model

