import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm
import argparse

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from interence import Yolo_interence

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',type=str,help='model_path指向训练结果权重，在logs下面存放',\
                        default='/yolo/model_data/yolo4_voc_weights.pth')
    parser.add_argument('--classes_path',type=str,help='classess_path指向训练数据类别',\
                        default='model_data/voc_classes.txt')
    parser.add_argument('--anchors_path',type=str,help='anchors_path指向先验框对应的txt文档',\
                        default='model_data/yolo_anchors.txt')
    parser.add_argument('--anchors_mask',help='anchors_mask用于帮助代码找到对应的先验框层',\
                        default=[[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    parser.add_argument('--input_shape',type=int,help='input_shpae用于指定输入图像的大小必须为32的倍数',\
                         default=[416,416])
    parser.add_argument('--confidence',type=float,help='confidence用来初筛一遍预测框，只有大于该阈值的框才会被保留下来',\
                         default=0.01)
    parser.add_argument('--nms_iou',type=float,help='confidence用来初筛一遍预测框，只有大于该阈值的框才会被保留下来',\
                         default=0.5)
    parser.add_argument('--letterbox_image',type=bool,help='letterbox_image变量用于控制是否使用letterbox_image对输入图像进行不失真的resize',\
                         default=False)
    parser.add_argument('--cuda',type=bool,help='是否使用GPU',\
                         default=True)
    parser.add_argument('--backbone',type=str,help='用于选择使用的主干特征提取网络，可根据需求\
                        修改，主要有mobilenetv1,mobilenetv2,mobilenetv3,ghostnet,\
                        vgg,densenet121,densenet169,densenet201,resnet50,cspdarknet53',
                        default='cspdarknet53')
    parser.add_argument('--process_model',type=str,help='选用后处理方法，有fpn和spp_fpn两种可选',default="spp_fpn") 
    parser.add_argument('--map_mode',type=int,help='map_mode用于指定该文件运行时计算的内容',\
                         default=0)
    parser.add_argument('--MINOVERLAP',type=float,help='MINOVERLAP用于指定想要获得的mAP0.x，mAP0.x',\
                         default=0.5)
    parser.add_argument('--score_threhold',type=float,help='map_mode用于指定该文件运行时计算的内容',\
                         default=0.5)
    parser.add_argument('--map_vis',type=bool,help='map_vis用于指定是否开启VOC_map计算的可视化',\
                         default=False)
    parser.add_argument('--map_out_path',type=str,help='结果输出的文件夹，默认为map_out',\
                         default="map_out")
    parser.add_argument('--VOCdevkit_path',type=str,help='结果输出的文件夹，默认为map_out',\
                         default="/yolo/VOCdevkit")
    
    
    args = parser.parse_args()
    '''
    Recall和Precision不像AP是一个面积的概念，因此在门限值（Confidence）不同时，网络的Recall和Precision值是不同的。
    默认情况下，本代码计算的Recall和Precision代表的是当门限值（Confidence）为0.5时，所对应的Recall和Precision值。

    受到mAP计算原理的限制，网络在计算mAP时需要获得近乎所有的预测框，这样才可以计算不同门限条件下的Recall和Precision值
    因此，本代码获得的map_out/detection-results/里面的txt的框的数量一般会比直接predict多一些，目的是列出所有可能的预测框，
    '''
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode用于指定该文件运行时计算的内容
    #   map_mode为0代表整个map计算流程，包括获得预测结果、获得真实框、计算VOC_map。
    #   map_mode为1代表仅仅获得预测结果。
    #   map_mode为2代表仅仅获得真实框。
    #   map_mode为3代表仅仅计算VOC_map。
    #   map_mode为4代表利用COCO工具箱计算当前数据集的0.50:0.95map。需要获得预测结果、获得真实框后并安装pycocotools才行
    #-------------------------------------------------------------------------------------------------------------------#
    # map_mode        = 0
    #--------------------------------------------------------------------------------------#
    #   此处的classes_path用于指定需要测量VOC_map的类别
    #   一般情况下与训练和预测所用的classes_path一致即可
    #--------------------------------------------------------------------------------------#
    # classes_path    = 'model_data/voc_classes.txt'
    #--------------------------------------------------------------------------------------#
    #   MINOVERLAP用于指定想要获得的mAP0.x，mAP0.x的意义是什么请同学们百度一下。
    #   比如计算mAP0.75，可以设定MINOVERLAP = 0.75。
    #
    #   当某一预测框与真实框重合度大于MINOVERLAP时，该预测框被认为是正样本，否则为负样本。
    #   因此MINOVERLAP的值越大，预测框要预测的越准确才能被认为是正样本，此时算出来的mAP值越低，
    #--------------------------------------------------------------------------------------#
    # MINOVERLAP      = 0.5
    # #--------------------------------------------------------------------------------------#
    # #   受到mAP计算原理的限制，网络在计算mAP时需要获得近乎所有的预测框，这样才可以计算mAP
    # #   因此，confidence的值应当设置的尽量小进而获得全部可能的预测框。
    # #   
    # #   该值一般不调整。因为计算mAP需要获得近乎所有的预测框，此处的confidence不能随便更改。
    # #   想要获得不同门限值下的Recall和Precision值，请修改下方的score_threhold。
    # #--------------------------------------------------------------------------------------#
    # confidence      = 0.001
    # #--------------------------------------------------------------------------------------#
    # #   预测时使用到的非极大抑制值的大小，越大表示非极大抑制越不严格。
    # #   
    # #   该值一般不调整。
    # #--------------------------------------------------------------------------------------#
    # nms_iou         = 0.5
    # #---------------------------------------------------------------------------------------------------------------#
    # #   Recall和Precision不像AP是一个面积的概念，因此在门限值不同时，网络的Recall和Precision值是不同的。
    # #   
    # #   默认情况下，本代码计算的Recall和Precision代表的是当门限值为0.5（此处定义为score_threhold）时所对应的Recall和Precision值。
    # #   因为计算mAP需要获得近乎所有的预测框，上面定义的confidence不能随便更改。
    # #   这里专门定义一个score_threhold用于代表门限值，进而在计算mAP时找到门限值对应的Recall和Precision值。
    # #---------------------------------------------------------------------------------------------------------------#
    # score_threhold  = 0.5
    # #-------------------------------------------------------#
    # #   map_vis用于指定是否开启VOC_map计算的可视化
    # #-------------------------------------------------------#
    # map_vis         = False
    # #-------------------------------------------------------#
    # #   指向VOC数据集所在的文件夹
    # #   默认指向根目录下的VOC数据集
    # #-------------------------------------------------------#
    # VOCdevkit_path  = 'VOCdevkit'
    # #-------------------------------------------------------#
    # #   结果输出的文件夹，默认为map_out
    # #-------------------------------------------------------#
    # map_out_path    = 'map_out'

    image_ids = open(os.path.join(args.VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()

    if not os.path.exists(args.map_out_path):
        os.makedirs(args.map_out_path)
    if not os.path.exists(os.path.join(args.map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(args.map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(args.map_out_path, 'detection-results')):
        os.makedirs(os.path.join(args.map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(args.map_out_path, 'images-optional')):
        os.makedirs(os.path.join(args.map_out_path, 'images-optional'))

    class_names, _ = get_classes(args.classes_path)

    if args.map_mode == 0 or args.map_mode == 1:
        print("Load model.")
        yolo = Yolo_interence(args=args)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(args.VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            if args.map_vis:
                image.save(os.path.join(args.map_out_path, "images-optional/" + image_id + ".jpg"))
            yolo.get_map_txt(image_id, image, class_names, map_out_path=args.map_out_path)
        print("Get predict result done.")
        
    if args.map_mode == 0 or args.map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(args.map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                root = ET.parse(os.path.join(args.VOCdevkit_path, "VOC2007/Annotations/"+image_id+".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox  = obj.find('bndbox')
                    left    = bndbox.find('xmin').text
                    top     = bndbox.find('ymin').text
                    right   = bndbox.find('xmax').text
                    bottom  = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if args.map_mode == 0 or args.map_mode == 3:
        print("Get map.")
        get_map(args.MINOVERLAP, True, score_threhold = args.score_threhold, path = args.map_out_path)
        print("Get map done.")

    if args.map_mode == 4:
        print("Get map.")
        get_coco_map(class_names = class_names, path = args.map_out_path)
        print("Get map done.")
