'''
该模块用来做预测推理功能，可以测试单张图片也可以测试视频流，不同的功能通过选择不同的
模式进行选择
'''
import time
from tkinter import Y
from turtle import left
import cv2
from PIL import Image
import argparse
import onnxruntime

import colorsys
from matplotlib import pyplot as plt
import os 
from tqdm import tqdm
import time 
import numpy as np
from requests import session
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont
from nets.yolo import YOLO
from utils.utils import *
from utils.utils_bbox import *

class Yolo_interence(object):
    def __init__(self, args):
        self.class_name,self.num_classes = get_classes(args.classes_path)
        self.anchors, self.num_anchors = get_anchors(args.anchors_path)
        self.anchors_mask = args.anchors_mask
        self.input_shape = args.input_shape
        self.model_path = args.model_path
        self.confidence = args.confidence
        self.nms_iou =args.nms_iou
        self.letterbox_image = args.letterbox_image
        self.cuda = args.cuda
        self.backbone = args.backbone
        self.process_model = args.process_model
        self.bbox_util = DecodeBox(self.anchors,self.num_classes,(args.input_shape[0],args.input_shape[1]),self.anchors_mask)
        #----------------------------------------------------#
        #   画框设置不同的颜色
        #----------------------------------------------------#
        hsv_tuples = [(x / self.num_classes,1.,1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x:colorsys.hsv_to_rgb(*x),hsv_tuples))
        self.colors = list(map(lambda x:(int(x[0]*255),int(x[1]*255),int(x[2]*255)),self.colors))
        self.generate()

    def generate(self,onnx=False):
        self.net = YOLO(self.anchors_mask,self.num_classes,self.process_model,self.backbone)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path,map_location=device))
        self.net = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    #----------------------------------------------------#
    #   检测图片
    #----------------------------------------------------#
    def detect_image(self,image,crop=False,count=False):
        image_shape = np.array(np.shape(image)[0:2])
        #------------------------------------------------------#
        # 代码仅仅支持RGB图像的预测，所有其他类型的图像都会转换为RGB
        #------------------------------------------------------#
        image = cvtColor(image)
        #------------------------------------------------------#
        # 给图像增加灰条，实现不失真的resize,也可以直接resize
        #------------------------------------------------------#
        image_data = resize_image(image,(self.input_shape[1],self.input_shape[0]),self.letterbox_image)
        #------------------------------------------------------#
        # 添加batch_size维度
        #------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data,dtype='float32')),(2,0,1)),0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #------------------------------------------------------#
            # 将预测框进行堆叠，然后进行非极大值抑制
            #------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs,1),self.num_classes,self.input_shape,image_shape,
                                                          self.letterbox_image,conf_thres=self.confidence,nms_thres=self.nms_iou)
            
            if results[0] is None:
                return image
            
            top_label = np.array(results[0][:,6],dtype = 'int32')
            top_conf = results[0][:,4] * results[0][:,5]
            top_boxes = results[0][:, :4]
        #------------------------------------------------------#
        # 设置字体和边框厚度，可以根据自己的喜好进行设置
        # 这里有JumpsHigher和simhei两种字体供选择
        #------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2*image.size[1]+0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape),1))
        #------------------------------------------------------#
        # 计数
        #------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_name[i], ":", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        #------------------------------------------------------#
        # 是否进行目标的裁剪
        #------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.sieze[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path,"crop_" + str(i) + ".png"),quality=95,subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        #------------------------------------------------------#
        # 图像绘制
        #------------------------------------------------------#
        for i,c in list(enumerate(top_label)):
            predicted_class = self.class_name[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class,score)
            draw = ImageDraw.Draw(image)  #创建绘图对象
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top-label_size[1]])
            else:
                text_origin = np.array([left,top+1])

            for i in range(thickness):
                draw.polygon([left + i, top + i, right - i, bottom - i,], outline=self.colors[c])
            draw.polygon([tuple(text_origin),tuple(text_origin + label_size)],fill=self.colors)
            draw.text(text_origin, str(label, 'Utf-8'), fill=(0,0,0),font=font)#第一个参数指定文字区域的左上角在图片中的位置，第二个参数是文字内容
            del draw

        return image  

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        #------------------------------------------------------#
        # 将输入图像转换为RGB图像，防止灰度图在预测时报错
        #------------------------------------------------------#
        image = cvtColor(image)  
        #------------------------------------------------------#
        # 给图像增加灰条，实现不失真的resize,也可以直接resize
        #------------------------------------------------------#
        image_data = resize_image(image,(self.input_shape[1],self.input_shape[0]),self.letterbox_image)
        #------------------------------------------------------#
        # 给图像加上batch_size维度
        #------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data,dtype='float32')),(2,0,1)),0)


        images = torch.from_numpy(image_data)
        if self.cuda:
            images = image.cuda()
            
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                #---------------------------------------------------------#
                #   将图像输入网络当中进行预测！
                #---------------------------------------------------------#
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                #---------------------------------------------------------#
                #   将预测框进行堆叠，然后进行非极大抑制
                #---------------------------------------------------------#
                results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                            image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
            
        t2 = time.time()
        tact_time = (t2-t1)/test_interval
        return tact_time

    def detect_heatmap(self,image,heatmap_save_path):
        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return Y

        #------------------------------------------------------#
        # 将输入图像转换为RGB图像，防止灰度图在预测时报错
        #------------------------------------------------------#
        image = cvtColor(image)  
        #------------------------------------------------------#
        # 给图像增加灰条，实现不失真的resize,也可以直接resize
        #------------------------------------------------------#
        image_data = resize_image(image,(self.input_shape[1],self.input_shape[0]),self.letterbox_image)
        #------------------------------------------------------#
        # 给图像加上batch_size维度
        #------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data,dtype='float32')),(2,0,1)),0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
        
        plt.imshow(image, alpha=1)
        plt.axis('off')
        mask = np.zeros((image.size[1],image.size[0]))
        for sub_output in outputs:
            sub_output = sub_output.cpu().numpy()
            b, c, h, w = np.shape(sub_output)
            sub_output = np.transpose(np.reshape(sub_output,[b,3,-1,h,w]), [0,3,4,1,2])[0]
            score = np.max(sigmoid(sub_output[...,4]), -1)
            score = cv2.resize(score,(image.size[0], image.size[1]))
            normed_score = (score*255).astype('uint8')
            mask = np.maximum(mask,normed_score)
        
        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap='jet')
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        plt.savefig(heatmap_save_path,dpi=200,bbox_inches='tight',pad_inches=-0.1)
        print("save to the" + heatmap_save_path)
        plt.show()

    #------------------------------------------------------#
    #      使用onnxruntime来进行推理验证
    #------------------------------------------------------#
    def onnx_interence(self,image,crop=False,count=False):
        image_shape = np.array(np.shape(image)[0:2])
        #------------------------------------------------------#
        # 代码仅仅支持RGB图像的预测，所有其他类型的图像都会转换为RGB
        #------------------------------------------------------#
        image = cvtColor(image)
        #------------------------------------------------------#
        # 给图像增加灰条，实现不失真的resize,也可以直接resize
        #------------------------------------------------------#
        image_data = resize_image(image,(self.input_shape[1],self.input_shape[0]),self.letterbox_image)
        #------------------------------------------------------#
        # 添加batch_size维度
        #------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data,dtype='float32')),(2,0,1)),0)
        with torch.no_grad():
            # images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #------------------------------------------------------#
            # 进行onnxruntime推理，这里存在一个问题，可能存在问题
            #------------------------------------------------------#
            session = onnxruntime.InferenceSession(os.path.splitext(self.model_path)[0] + '_simplify_' + '.onnx')
            onnxruntime_inputs = {session.get_inputs()[0].name:images}
            outputs = session.run(None,onnxruntime_inputs)
            outputs = torch.from_numpy(outputs)

            # outputs = self.net(images)
            # outputs = self.bbox_util.decode_box(outputs)
            #------------------------------------------------------#
            # 将预测框进行堆叠，然后进行非极大值抑制
            #------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs,1),self.num_classes,self.input_shape,image_shape,
                                                          self.letterbox_image,conf_thres=self.confidence,nms_thres=self.nms_iou)
            
            if results[0] is None:
                return image
            
            top_label = np.array(results[0][:,6],dtype = 'int32')
            top_conf = results[0][:,4] * results[0][:,5]
            top_boxes = results[0][:, :4]
        #------------------------------------------------------#
        # 设置字体和边框厚度，可以根据自己的喜好进行设置
        # 这里有JumpsHigher和simhei两种字体供选择
        #------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2*image.size[1]+0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape),1))
        #------------------------------------------------------#
        # 计数
        #------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_name[i], ":", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        #------------------------------------------------------#
        # 是否进行目标的裁剪
        #------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.sieze[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path,"crop_" + str(i) + ".png"),quality=95,subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        #------------------------------------------------------#
        # 图像绘制
        #------------------------------------------------------#
        for i,c in list(enumerate(top_label)):
            predicted_class = self.class_name[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class,score)
            draw = ImageDraw.Draw(image)  #创建绘图对象
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top-label_size[1]])
            else:
                text_origin = np.array([left,top+1])

            for i in range(thickness):
                draw.polygon([left + i, top + i, right - i, bottom - i,], outline=self.colors[c])
            draw.polygon([tuple(text_origin),tuple(text_origin + label_size)],fill=self.colors)
            draw.text(text_origin, str(label, 'Utf-8'), fill=(0,0,0),font=font)#第一个参数指定文字区域的左上角在图片中的位置，第二个参数是文字内容
            del draw

        return image 
 
    def get_map_txt(self, image_id,image,class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),'w')
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
         #------------------------------------------------------#
        # 给图像增加灰条，实现不失真的resize,也可以直接resize
        #------------------------------------------------------#
        image_data = resize_image(image,(self.input_shape[1],self.input_shape[0]),self.letterbox_image)
        #------------------------------------------------------#
        # 添加batch_size维度
        #------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data,dtype='float32')),(2,0,1)),0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #------------------------------------------------------#
            # 将预测框进行堆叠，然后进行非极大值抑制
            #------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs,1),self.num_classes,self.input_shape,image_shape,
                                                          self.letterbox_image,conf_thres=self.confidence,nms_thres=self.nms_iou)
            
            if results[0] is None:
                return image
            
            top_label = np.array(results[0][:,6],dtype = 'int32')
            top_conf = results[0][:,4] * results[0][:,5]
            top_boxes = results[0][:, :4]
        
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_name[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)),str(int(right)),str(int(bottom))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',type=str,help='model_path指向训练结果权重，在logs下面存放',\
                        default='model_data/yolo4_weights.pth')
    parser.add_argument('--classes_path',type=str,help='classess_path指向训练数据类别',\
                        default='model_data/voc_classes.txt')
    parser.add_argument('--anchors_path',type=str,help='anchors_path指向先验框对应的txt文档',\
                        default='model_data/yolo_anchors.txt')
    parser.add_argument('--anchors_mask',help='anchors_mask用于帮助代码找到对应的先验框层',\
                        default=[[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    parser.add_argument('--input_shape',type=int,help='input_shpae用于指定输入图像的大小必须为32的倍数',\
                         default=[416,416])
    parser.add_argument('--confidence',type=float,help='confidence用来初筛一遍预测框，只有大于该阈值的框才会被保留下来',\
                         default=0.5)
    parser.add_argument('--nms_iou',type=float,help='confidence用来初筛一遍预测框，只有大于该阈值的框才会被保留下来',\
                         default=0.3)
    parser.add_argument('--letterbox_image',type=bool,help='letterbox_image变量用于控制是否使用letterbox_image对输入图像进行不失真的resize',\
                         default=False)
    parser.add_argument('--cuda',type=bool,help='是否使用GPU',\
                         default=True)
    parser.add_argument('--backbone',type=str,help='用于选择使用的主干特征提取网络，可根据需求\
                        修改，主要有mobilenetv1,mobilenetv2,mobilenetv3,ghostnet,\
                        vgg,densenet121,densenet169,densenet201,resnet50,cspdarknet53',
                        default='cspdarknet53')
    parser.add_argument('--process_model',type=str,help='选用后处理方法，有fpn和spp_fpn两种可选',default="spp_fpn") 
    args = parser.parse_args()

    Yolo = Yolo_interence(args=args)
    #----------------------------------------------------------------------------------------------------------#
    # 以下代码是进行模式选择
    # mode用于指定测试的模式：
    #   'image'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'heatmap'           表示进行预测结果的热力图可视化，详情查看下方注释。
    #   'onnxruntimer'      表示使用onnxruntimer进行推理验证
    #----------------------------------------------------------------------------------------------------------#
    mode = "image"
    #----------------------------------------------------------------------------------------------------------#
    # crop                 指定了是否在单张图片预测后对目标进行截取
    # count                指定了是否进行目标的计数，一张图像有多少个目标
    # video_path           用于指定视频的路径，当video_path=0时表示检测摄像头,如果想要检测视频，那么将该路径改为视频路径
    # video_save_path      表示要保存的路径，当检测设备为相机时不保存，要想保存检测视频，则将参数改为路径
    # video_fps            用于保存视频的fps
    # video_path、video_save_path、video_fps仅在mode="video"时有效
    #----------------------------------------------------------------------------------------------------------#
    crop = False
    count = False
    video_path = 0
    video_save_path = ""
    video_fps = 25
    #----------------------------------------------------------------------------------------------------------#
    # test_interval        用于指定测量fps的时候，图片检测的次数，理论上test_interval越大，fps越准确
    # fps_image_path       用于指定测试的fps图片
    # test_interval、fps_image_path仅在mode="fps"时有效
    #----------------------------------------------------------------------------------------------------------#
    test_interval = 1000
    fps_image_path = "img/street.jpg"
    #----------------------------------------------------------------------------------------------------------#
    # dir_origin_path      用于指定检测图片的文件夹路径（检测一个文件夹下的所有图片）
    # dir_save_path        用于指定保存检测完的图像路径
    # dir_origin_path、dir_save_path仅在mode="dir_predict"时有效
    #----------------------------------------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path = "img_out/"
    #----------------------------------------------------------------------------------------------------------#
    # heatmap_save_path     热力图的保存路径
    # heatmap_save_path仅在mode="heatmap"时有效
    #----------------------------------------------------------------------------------------------------------#
    heatmap_save_path = "model_data/heatmap_vision.png"

    while True:
        if mode == "image":
            img = input('Input image filename')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = Yolo.detect_image(image,crop=crop, count=count)
                r_image.show()
        
        elif mode == "video":
            capture = cv2.VideoCapture(video_path)
            if video_save_path != "":
                #----------------------------------------------------------------------------------------------------------#
                # #指定编码格式，xvid保存为.avi格式
                # 编码格式有：I420、PIMI、XVID、THEO、FLVI、MP4V
                # I420:未压缩的YUV颜色编码格式，色度子采样为4：2：0，该编码格式具有较好的兼容性，但是生成的视频文件较大，文件扩展.avi
                # PIMI:MPEG-1编码格式，生成的文件扩展名为.avi
                # XVID:MPEG-4编码格式，如果希望得到的视频大小为平均值，可以选用该格式，文件扩展名为.avi
                # THEO:Ogg Vorbis视频格式，文件扩展名.ogv
                # FLVI:Flash视频格式，文件扩展名为.flv
                # MP4V:文件扩展名为.MP4
                #----------------------------------------------------------------------------------------------------------#
                fourcc = cv2.VideoWriter_fourcc(*'XVID') 
                size   = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                out    = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
            
            ref, frame = capture.read()
            if not ref:
                raise ValueError("Can not read or video correctly, please confirm the video path and the camera is turned on")
            
            fps = 0.0
            while(True):
                t1 = time.time()
                ref, frame = capture.read()
                if not ref:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(np.uint8(frame))
                frame = np.array(Yolo.detect_image(frame))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                t2 = time.time()
                fps = (fps + (1./(t2-t1))) / 2 #不断更新两帧间的处理时间（fps）
                print("fps= %.2f"%(fps))
                frame = cv2.putText(frame, "fps= %.2f"%(fps),(0,40),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.imshow("video",frame)
                c = cv2.waitkey(1) & 0xff

                if video_save_path !="":
                    out.write(frame)

                if c == 27:
                    capture.release()
                    break
            
            print("Video Detection Done!")
            capture.release()
            if video_save_path != "":
                print("Save processed video to the path :" + video_save_path)
                out.release()
            cv2.destroyAllWindows()
            

        elif mode == "fps":
            img = Image.open(fps_image_path)
            tact_time = Yolo.get_FPS(img, test_interval=test_interval)
            print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS')
        
        elif mode == "dir_predict":
            img_names = os.listdir(dir_origin_path)
            for img_name in tqdm(img_names):
                #.lower()函数实现将字符串中大写字母转换为小写
                #.endswith()函数实现对后缀的判断
                if img_name.lower().endswith(('.bmp','.dib','.png','.jpg','.jpeg','.pbm','.pgm','.ppm','.tif','.tiff')):
                    image_path = os.path.join(dir_origin_path,img_name)
                    image = Image.open(image_path)
                    r_image = Yolo.detect_image(image)
                    if not os.path.exists(dir_save_path):
                        os.makedirs(dir_save_path)
                    r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg",".png")),quality=95, subsampling=0)
        elif mode == "heatmap":
            while True:
                img = input('Input image filename:')
                try:
                    image = Image.open(img)
                except:
                    print('Open Error! Try again!')
                    continue
                else:
                    Yolo.detect_heatmap(image,heatmap_save_path=heatmap_save_path)
        elif mode == "onnxruntimer":
            while True:
                img = input('Input image filename')
                try:
                    image = Image.open(img)
                except:
                    print('Open Error! Try again!')
                    continue
                else:
                    r_image = Yolo.onnx_interence(image,crop=crop, count=count)
                    r_image.show()
        
        else:
            raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps','heatmap', 'onnxruntimer', 'dir_predict'")

            
                



    
    
                    
                    

