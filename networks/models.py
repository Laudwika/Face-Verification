import math

import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import Parameter
from collections import OrderedDict
from .anchors import Anchors
from helper.utils import RegressionTransform
from networks import losses
import torchvision.models.resnet as resnet
import torchvision.models._utils as _utils
import torchvision.models.detection.backbone_utils as backbone_utils

# from config import device, num_classes
from helper.utils import device

#RETINAFACE NETWORK

class ContextModule(nn.Module):
    def __init__(self,in_channels=256):
        super(ContextModule,self).__init__()
        self.det_conv1 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channels)
        )
        self.det_context_conv1 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels//2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True)
        )
        self.det_context_conv2 = nn.Sequential(
            nn.Conv2d(in_channels//2,in_channels//2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channels//2)
        )
        self.det_context_conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels//2,in_channels//2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True)
        )
        self.det_context_conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels//2,in_channels//2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channels//2)
        )
        self.det_concat_relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x1 = self.det_conv1(x)
        x_ = self.det_context_conv1(x)
        x2 = self.det_context_conv2(x_)
        x3_ = self.det_context_conv3_1(x_)
        x3 = self.det_context_conv3_2(x3_)

        out = torch.cat((x1,x2,x3),1)
        act_out = self.det_concat_relu(out)

        return act_out

class FeaturePyramidNetwork(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FeaturePyramidNetwork,self).__init__()
        self.lateral_blocks = nn.ModuleList()
        self.context_blocks = nn.ModuleList()
        self.aggr_blocks = nn.ModuleList()
        for i, in_channels in enumerate(in_channels_list):
            if in_channels == 0:
                continue
            lateral_block_module = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            aggr_block_module = nn.Sequential(
                nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            context_block_module = ContextModule(out_channels)
            self.lateral_blocks.append(lateral_block_module)
            self.context_blocks.append(context_block_module)
            if i > 0 :
                self.aggr_blocks.append(aggr_block_module)

        # initialize params of fpn layers
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)                

    def forward(self,x):
        names = list(x.keys())
        x = list(x.values())

        last_inner = self.lateral_blocks[-1](x[-1])
        results = []
        results.append(self.context_blocks[-1](last_inner))
        for feature, lateral_block, context_block, aggr_block in zip(
            x[:-1][::-1], self.lateral_blocks[:-1][::-1], self.context_blocks[:-1][::-1], self.aggr_blocks[::-1]
            ):
            if not lateral_block:
                continue
            lateral_feature = lateral_block(feature)
            feat_shape = lateral_feature.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = lateral_feature + inner_top_down
            last_inner = aggr_block(last_inner)
            results.insert(0, context_block(last_inner))

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out

class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)
        self.output_act = nn.LogSoftmax(dim=-1)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1)
        b, h, w, c = out.shape
        out = out.view(b, h, w, self.num_anchors, 2)
        out = self.output_act(out)
        
        return out.contiguous().view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1)

        return out.contiguous().view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1)

        return out.contiguous().view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(self,backbone,return_layers,anchor_nums=3):
        super(RetinaFace,self).__init__()
        # if backbone_name == 'resnet50':
        #     self.backbone = resnet.resnet50(pretrained)
        # self.backbone = resnet.__dict__[backbone_name](pretrained=pretrained)
        # self.return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}
        assert backbone,'Backbone can not be none!'
        assert len(return_layers)>0,'There must be at least one return layers'
        self.body = _utils.IntermediateLayerGetter(backbone, return_layers)
        in_channels_stage2 = 256
        # in_channels_stage2 = 64
        in_channels_list = [
            #in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = 256
        self.fpn = FeaturePyramidNetwork(in_channels_list,out_channels)
        # self.ClassHead = ClassHead()
        # self.BboxHead = BboxHead()
        # self.LandmarkHead = LandmarkHead()
        self.ClassHead = self._make_class_head()
        self.BboxHead = self._make_bbox_head()
        self.LandmarkHead = self._make_landmark_head()
        self.anchors = Anchors()
        self.regressBoxes = RegressionTransform()        
        self.losslayer = losses.LossLayer()

    def _make_class_head(self,fpn_num=3,inchannels=512,anchor_num=3):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=512,anchor_num=3):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=512,anchor_num=3):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self,inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        out = self.body(img_batch)
        features = self.fpn(out)

        # bbox_regressions = torch.cat([self.BboxHead(feature) for feature in features.values()], dim=1)
        # ldm_regressions = torch.cat([self.LandmarkHead(feature) for feature in features.values()], dim=1)
        # classifications = torch.cat([self.ClassHead(feature) for feature in features.values()],dim=1)      

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features.values())], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features.values())], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features.values())],dim=1)      

        anchors = self.anchors(img_batch)

        if self.training:
            return self.losslayer(classifications, bbox_regressions,ldm_regressions, anchors, annotations)
        else:
            bboxes, landmarks = self.regressBoxes(anchors, bbox_regressions, ldm_regressions, img_batch)

            return classifications, bboxes, landmarks

def create_retinaface(return_layers,backbone_name='resnet50',anchors_num=3,pretrained=True):
    backbone = resnet.__dict__[backbone_name](pretrained=pretrained)
    # freeze layer1
    for name, parameter in backbone.named_parameters():
        # if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #     parameter.requires_grad_(False)
        if name == 'conv1.weight':
            # print('freeze first conv layer...')
            parameter.requires_grad_(False)

    model = RetinaFace(backbone,return_layers,anchor_nums=3)

    return model



    


#ARCFACE AND MASK CLASSIFIER NETWORKS

__all__ = ['ResNet', 'resnet18', 'resnet101']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, use_se=True, im_size=112):
        self.inplanes = 64
        self.use_se = use_se
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout()

        if im_size == 112:
            self.fc = nn.Linear(512 * 7 * 7, 512)
        else:  # 224
            self.fc = nn.Linear(512 * 14 * 14, 512)
        self.bn3 = nn.BatchNorm1d(512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=self.use_se))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn3(x)

        return x


def resnet18(use_se, **kwargs):
    model = ResNet(IRBlock, [2, 2, 2, 2], use_se=use_se, **kwargs)
    # if args.pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet101(use_se, **kwargs):
    model = ResNet(IRBlock, [3, 4, 23, 3], use_se=use_se, **kwargs)
    # if args.pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model





if __name__ == "__main__":
    pass
    # from utils import parse_args
    # from torchscope import scope

    # args = parse_args()
    # model = resnet101(args)
    # scope(model, (3, 112, 112))
