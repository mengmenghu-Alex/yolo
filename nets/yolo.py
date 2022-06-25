import imp
import math 
from collections import OrderedDict
from turtle import forward
import torch.nn as nn
import torch
from nets.darknet53 import darnet53
from nets.CSPdarknet import CSPdarknet53

from .densenet import _Transition, densenet121, densenet169, densenet201
from .ghostnet import ghostnet
from .mobilenet_v1 import mobilenet_v1
from .mobilenet_v2 import mobilenet_v2
from .mobilenet_v3 import mobilenet_v3
from .resnet import resnet50
from .vgg import vgg

'''
yolov4和yolov3的模型大小为226M和237M，模型较大，如果直接使用该模型进行训练自己的数据集的话会使得效果很差（实测）
为此，使用轻量化的主干特征提取网络代替yolov4和yolov3的主干特征提取网络，提供的主干网络有
1、darknet53
2、cspdarknet53
3、densenet121 densenet169 densenet 201
4、mobilenet_v1
5、mobilenet_v2
6、mobilenet_v3
7、ghostnet
8、resnet50
9、vgg
'''
class MobileNetV1(nn.Module):
    def __init__(self, pretrained = False):
        super(MobileNetV1, self).__init__()
        self.model = mobilenet_v1(pretrained=pretrained)

    def forward(self, x):
        out3 = self.model.stage1(x)
        out4 = self.model.stage2(x)
        out5 = self.model.stage3(x)
        return out3, out4, out5

class MobileNetV2(nn.Module):
    def __init__(self, pretrained = False):
        super(MobileNetV2, self).__init__()
        self.model = mobilenet_v2(pretrained=pretrained)

    def forward(self, x):
        out3 = self.model.features[:7](x)
        out4 = self.model.features[7:14](out3)
        out5 = self.model.features[14:18](out4)
        return out3, out4, out5

class MobileNetV3(nn.Module):
    def __init__(self, pretrained = False):
        super(MobileNetV3, self).__init__()
        self.model = mobilenet_v3(pretrained=pretrained)

    def forward(self, x):
        out3 = self.model.features[:7](x)
        out4 = self.model.features[7:13](out3)
        out5 = self.model.features[13:16](out4)
        return out3, out4, out5

class GhostNet(nn.Module):
    def __init__(self, pretrained=True):
        super(GhostNet, self).__init__()
        model = ghostnet()
        if pretrained:
            state_dict = torch.load("model_data/ghostnet_weights.pth")
            model.load_state_dict(state_dict)
        del model.global_pool
        del model.conv_head
        del model.act2
        del model.classifier
        del model.blocks[9]
        self.model = model

    def forward(self, x):
        x = self.model.conv_stem(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        feature_maps = []

        for idx, block in enumerate(self.model.blocks):
            x = block(x)
            if idx in [2,4,6,8]:
                feature_maps.append(x)
        return feature_maps[1:]

class VGG(nn.Module):
    def __init__(self, pretrained=False):
        super(VGG, self).__init__()
        self.model = vgg(pretrained)

    def forward(self, x):
        feat1 = self.model.features[  :5 ](x)
        feat2 = self.model.features[5 :10](feat1)
        feat3 = self.model.features[10:17](feat2)
        feat4 = self.model.features[17:24](feat3)
        feat5 = self.model.features[24:  ](feat4)
        return [feat3, feat4, feat5]

class Densenet(nn.Module):
    def __init__(self, backbone, pretrained=False):
        super(Densenet, self).__init__()
        densenet = {
            "densenet121" : densenet121, 
            "densenet169" : densenet169, 
            "densenet201" : densenet201
        }[backbone]
        model = densenet(pretrained)
        del model.classifier
        self.model = model

    def forward(self, x):
        feature_maps = []
        for block in self.model.features:
            if type(block)==_Transition:
                for _, subblock in enumerate(block):
                    x = subblock(x)
                    if type(subblock)==nn.Conv2d:
                        feature_maps.append(x)
            else:
                x = block(x)
        x = F.relu(x, inplace=True)
        feature_maps.append(x)
        return feature_maps[1:]

class ResNet(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet, self).__init__()
        self.model = resnet50(pretrained)

    def forward(self, x):
        x       = self.model.conv1(x)
        x       = self.model.bn1(x)
        feat1   = self.model.relu(x)

        x       = self.model.maxpool(feat1)
        feat2   = self.model.layer1(x)

        feat3   = self.model.layer2(feat2)
        feat4   = self.model.layer3(feat3)
        feat5   = self.model.layer4(feat4)
        return [feat3, feat4, feat5]
'''
在这提供了两套后处理方法：1、传统FPN特征金字塔后处理；2、SPP+PAN特征金字塔后处理(yolov4)
'''
def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features

#---------------------------------------------------#
#   三次卷积块
#---------------------------------------------------#
def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

class YOLO(nn.Module):
    def __init__(self,anchors_mask,num_classes,process_model,backbone,pretrained=False):
        super(YOLO,self).__init__()
        self.process_model = process_model
        self.backbone = backbone
        if self.process_model == 'fpn':
            if self.backbone == "darknet53":
                self.backbone = darnet53(pretrained=pretrained)
                out_filters = self.backbone.layers_out_filters
            
            elif self.backbone == 'mobilenetv1':
                self.backbone = MobileNetV1(pretrained=pretrained)
                out_filters = [256, 512, 1024]

            elif self.backbone == 'mobilenetv2':
                self.backbone = MobileNetV2(pretrained=pretrained)
                out_filters = [32, 96, 320]

            elif self.backbone == "mobilenetv3":
                self.backbone = MobileNetV3(pretrained=pretrained)
                out_filters = [40, 112, 160]

            elif self.backbone == "ghostnet":
                self.backbone = GhostNet(pretrained=pretrained)
                out_filters = [40, 112, 160]
            
            elif self.backbone == "vgg":
                self.backbone = VGG(pretrained=pretrained)
                out_filters = [256, 512, 512]

            elif self.backbone in ["densenet121", "densenet169", "densenet201"]:
                self.backbone = Densenet(pretrained=pretrained)
                out_filters = {
                    "densenet121" : [256, 515, 1024],
                    "densenet169" : [256, 640, 1664],
                    "densenet201" : [256, 896, 1920]
                }[self.backbone]
            
            elif self.backbone == "resnet50":
                self.backbone = ResNet(pretrained=pretrained)
                out_filters = [512, 1024, 2048]

            elif self.backbone == "cspdarknet53":
                self.backbone   = CSPdarknet53(pretrained=pretrained)
                out_filters = [256, 512, 1024]

            else:
                raise ValueError(f'目前使用的backbone还没有支持，请使用支持的backbone')

            self.last_layer0 = make_five_conv([512,1024],out_filters[-1])
            self.last_layer0_out = yolo_head([1024,len(anchors_mask[0])*(num_classes+5)],512)

            self.conv1_upsample = Upsample(512,256)
            self.last_layer1 = make_five_conv([256,512],out_filters[-2]+256)
            self.last_layer1_out = yolo_head([512,len(anchors_mask[1])*(num_classes+5)],256)

            self.conv2_upsample = Upsample(256,128)
            self.last_layer2 = make_five_conv([128,256],out_filters[-3]+128)
            self.layer_layer2_out = yolo_head([256,len(anchors_mask[2])*(num_classes+5)],128)


        elif self.process_model == 'spp_fpn':
            if self.backbone == "darknet53":
                self.backbone = darnet53(pretrained=pretrained)
                out_filters = self.backbone.layers_out_filters
            
            elif self.backbone == 'mobilenetv1':
                self.backbone = MobileNetV1(pretrained=pretrained)
                out_filters = [256, 512, 1024]

            elif self.backbone == 'mobilenetv2':
                self.backbone = MobileNetV2(pretrained=pretrained)
                out_filters = [32, 96, 320]

            elif self.backbone == "mobilenetv3":
                self.backbone = MobileNetV3(pretrained=pretrained)
                out_filters = [40, 112, 160]

            elif self.backbone == "ghostnet":
                self.backbone = GhostNet(pretrained=pretrained)
                out_filters = [40, 112, 160]
            
            elif self.backbone == "vgg":
                self.backbone = VGG(pretrained=pretrained)
                out_filters = [256, 512, 512]

            elif self.backbone in ["densenet121", "densenet169", "densenet201"]:
                self.backbone = Densenet(pretrained=pretrained)
                out_filters = {
                    "densenet121" : [256, 515, 1024],
                    "densenet169" : [256, 640, 1664],
                    "densenet201" : [256, 896, 1920]
                }[self.backbone]
            
            elif self.backbone == "resnet50":
                self.backbone = ResNet(pretrained=pretrained)
                out_filters = [512, 1024, 2048]

            elif self.backbone == "cspdarknet53":
                self.backbone   = CSPdarknet53(pretrained=pretrained)
                out_filters = [256, 512, 1024]

            else:
                raise ValueError(f'目前使用的backbone还没有支持，请使用支持的backbone')

            self.conv1      = make_three_conv([512,1024],out_filters[2])
            self.SPP        = SpatialPyramidPooling()
            self.conv2      = make_three_conv([512,1024],2048)

            self.upsample1          = Upsample(512,256)
            self.conv_for_P4        = conv2d(out_filters[1],256,1)
            self.make_five_conv1    = make_five_conv([256, 512],512)

            self.upsample2          = Upsample(256,128)
            self.conv_for_P3        = conv2d(out_filters[0],128,1)
            self.make_five_conv2    = make_five_conv([128, 256],256)

            # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
            self.yolo_head3         = yolo_head([256, len(anchors_mask[0]) * (5 + num_classes)],128)

            self.down_sample1       = conv2d(128,256,3,stride=2)
            self.make_five_conv3    = make_five_conv([256, 512],512)

            # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
            self.yolo_head2         = yolo_head([512, len(anchors_mask[1]) * (5 + num_classes)],256)

            self.down_sample2       = conv2d(256,512,3,stride=2)
            self.make_five_conv4    = make_five_conv([512, 1024],1024)

            # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
            self.yolo_head1         = yolo_head([1024, len(anchors_mask[2]) * (5 + num_classes)],512)

    def forward(self,x):
        if self.process_model == 'fpn':
            x2,x1,x0 = self.backbone(x)
            out0_branch = self.last_layer0(x0)
            out0 = self.last_layer1_out(out0_branch)

            x1_in = self.conv1_upsample(out0_branch)
            x1_in = torch.cat([x1_in,x1],1)
            out1_branch = self.last_layer1(x1_in)
            out1 = self.last_layer1_out(out1_branch)

            x2_in = self.conv2_upsample(out1_branch)
            x2_in = self.cat([x2_in,x2],1)
            out2 = self.last_layer2(x2_in)
            out2 = self.layer_layer2_out(out2)
            return out0,out1,out2
        elif self.process_model == 'spp_fpn':
            x2, x1, x0 = self.backbone(x)

            # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,2048 
            P5 = self.conv1(x0)
            P5 = self.SPP(P5)
            # 13,13,2048 -> 13,13,512 -> 13,13,1024 -> 13,13,512
            P5 = self.conv2(P5)

            # 13,13,512 -> 13,13,256 -> 26,26,256
            P5_upsample = self.upsample1(P5)
            # 26,26,512 -> 26,26,256
            P4 = self.conv_for_P4(x1)
            # 26,26,256 + 26,26,256 -> 26,26,512
            P4 = torch.cat([P4,P5_upsample],axis=1)
            # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
            P4 = self.make_five_conv1(P4)

            # 26,26,256 -> 26,26,128 -> 52,52,128
            P4_upsample = self.upsample2(P4)
            # 52,52,256 -> 52,52,128
            P3 = self.conv_for_P3(x2)
            # 52,52,128 + 52,52,128 -> 52,52,256
            P3 = torch.cat([P3,P4_upsample],axis=1)
            # 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
            P3 = self.make_five_conv2(P3)

            # 52,52,128 -> 26,26,256
            P3_downsample = self.down_sample1(P3)
            # 26,26,256 + 26,26,256 -> 26,26,512
            P4 = torch.cat([P3_downsample,P4],axis=1)
            # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
            P4 = self.make_five_conv3(P4)

            # 26,26,256 -> 13,13,512
            P4_downsample = self.down_sample2(P4)
            # 13,13,512 + 13,13,512 -> 13,13,1024
            P5 = torch.cat([P4_downsample,P5],axis=1)
            # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
            P5 = self.make_five_conv4(P5)

            #---------------------------------------------------#
            #   第三个特征层
            #   y3=(batch_size,75,52,52)
            #---------------------------------------------------#
            out2 = self.yolo_head3(P3)
            #---------------------------------------------------#
            #   第二个特征层
            #   y2=(batch_size,75,26,26)
            #---------------------------------------------------#
            out1 = self.yolo_head2(P4)
            #---------------------------------------------------#
            #   第一个特征层
            #   y1=(batch_size,75,13,13)
            #---------------------------------------------------#
            out0 = self.yolo_head1(P5)

            return out0, out1, out2

        
            
