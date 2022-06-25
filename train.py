# from ctypes import get_last_error
from datetime import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

import argparse
from pickle import TRUE
import numpy as np
from sklearn.utils import shuffle
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from nets.yolo import YOLO
from nets.yolo_training import (YOLOLoss, get_lr_scheduler, set_optimizer_lr,
                                weights_init)
from utils.callbacks import LossHistory,EvalCallback
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_anchors, get_classes, show_config
from utils.utils_fit import fit_one_epoch
from utils.utils import download_weights

def train(args):
    ngpus_per_node = torch.cuda.device_count()
    if args.distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device('cuda',local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
    
    #--------------------------------------------------#
    #          下载主干特征提取网络预训练权重
    #--------------------------------------------------#
    if args.pretrained:
        if args.distributed:
            if args.local_rank == 0:
                download_weights(args.backbone)
            dist.barrier()
        else:
            download_weights(args.backbone)
    
    #--------------------------------------------------#
    #          获取classes和anchor
    #--------------------------------------------------#
    class_names,num_classes = get_classes(args.classes_path)
    anchors, num_anchors = get_anchors(args.anchors_path)

    #--------------------------------------------------#
    #          创建yolo模型
    #--------------------------------------------------#
    model = YOLO(args.anchors_mask,num_classes,process_model=args.process_model,
                 backbone = args.backbone,pretrained = args.pretrained)
    if not args.pretrained:
        weights_init(model)
    
    if args.model_path != '':
        if local_rank == 0:
            print('load weight {}'.format(args.model_path))
    #--------------------------------------------------#
    #          根据预训练权重的Key和模型的Key进行加载
    #--------------------------------------------------#    
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.model_path,map_location=device)
        load_key, no_load_key, temp_dict = [],[],{}
        for k,v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
    #--------------------------------------------------#
    #        显示没有匹配上的Key
    #--------------------------------------------------# 
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    #--------------------------------------------------#
    #        获得训练函数
    #--------------------------------------------------#
    yolo_loss = YOLOLoss(anchors,num_classes,args.input_shape,args.Cuda,args.anchors_mask)
    #--------------------------------------------------#
    #        记录loss
    #--------------------------------------------------#  
    if local_rank == 0:
        time_str = datetime.strftime(datetime.now(),'%Y_%M%D%H%M%S')
        log_dir = os.path.join(args.save_dir,"loss_"+str(time_str))
        loss_history = LossHistory(log_dir,model,input_shape=args.input_shape)
    else:
        loss_history = None
    #------------------------------------------------------------------#
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    #   浮点数运算
    #------------------------------------------------------------------#
    if args.fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler() 
    else:
        scaler = None
    model_train = model.train()

    #--------------------------------------------------#
    #        多卡同步Bn
    #--------------------------------------------------# 
    if args.sync_bn and ngpus_per_node > 1 and args.distributed:
        model_train = torch.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif args.sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")
    
    if args.Cuda:
        if args.distributed:
            #--------------------------------------------------#
            #        多卡平行运行
            #--------------------------------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParaller(
                model_train,device_ids = [local_rank],find_unused_parameters = True
            )
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    
    #--------------------------------------------------#
    #        读取数据集对应的txt
    #--------------------------------------------------# 
    with open(args.train_annotation_path) as f:
        train_lines = f.readlines()
    with open(args.val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(classes_path = args.classes_path, anchors_path = args.anchors_path, anchors_mask = args.anchors_mask, model_path = args.model_path, input_shape = args.input_shape, \
            Init_Epoch = args.Init_Epoch, Freeze_Epoch = args.Freeze_Epoch, UnFreeze_Epoch = args.UnFreeze_Epoch, Freeze_batch_size = args.Freeze_batch_size, Unfreeze_batch_size = args.Unfreeze_batch_size, Freeze_Train = args.Freeze_Train, \
            Init_lr = args.Init_lr, Min_lr = args.Min_lr, optimizer_type = args.optimizer_type, momentum = args.momentum, lr_decay_type = args.lr_decay_type, \
            save_period = args.save_period, save_dir = args.save_dir, num_workers = args.num_workers, num_train = num_train, num_val = num_val)
        wanted_step = 5e4 if args.optimizer_type == 'sgd' else 1.5e4
        total_step = num_train // args.Unfreeze_batch_size * args.UnFreeze_Epoch
        if total_step <= wanted_step:
            wanted_epoch = wanted_step // (num_train//args.Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m"%(args.optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"%(num_train,args.Unfreeze_batch_size, args.UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m"%(total_step, wanted_step, wanted_epoch))
    #--------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   冻结网络训练的过程中对主干之外的网络进行微调，同时
    #   冻结训练也可以防止训练初期权值被破坏
    #--------------------------------------------------#    
    if True:
        UnFreeze_flag = False
        if args.Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False
        batch_size = args.Freeze_batch_size if args.Freeze_Train else args.Unfreeze_batch_size 
    #--------------------------------------------------#
    #   判断当前batch_size，自适应调整学习率
    #--------------------------------------------------#  
        nbs = 64
        lr_limit_max = 1e-3 if args.optimizer_type == 'adam' else 5e-2
        lr_limit_min = 3e-4 if args.optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs* args.Init_lr, lr_limit_min),lr_limit_max)
        Min_lr_fit = min(max(batch_size /nbs * args.Min_lr,lr_limit_min * 1e-2),lr_limit_max * 1e-2)
    #--------------------------------------------------#
    #   根据optimizer_type选择优化器(不太清晰)
    #--------------------------------------------------#  
        pg0,pg1,pg2 = [],[],[]
        for k,v in model.named_modules():
            if hasattr(v,"bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v,nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)
            elif hasattr(v,"weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)
        optimizer = {
            'adam' : optim.Adam(pg0,Init_lr_fit,betas=(args.momentum,0.999)),
            'sgd'  :  optim.SGD(pg0, Init_lr_fit, momentum = args.momentum, nesterov=True)
        }[args.optimizer_type]
        optimizer.add_param_group({"params":pg1, "weight_decay":args.weight_decay})
        optimizer.add_param_group({"params":pg2})
    #--------------------------------------------------#
    #   获得学习率下降的公式
    #--------------------------------------------------#  
        lr_scheduler_func = get_lr_scheduler(args.lr_decay_type,Init_lr_fit,Min_lr_fit,args.UnFreeze_Epoch)
        
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集太小，无法继续进行训练，请扩充数据集")

    #--------------------------------------------------#
    #   构建数据集加载器，读取数据（前处理）
    #--------------------------------------------------# 
        train_dataset = YoloDataset(train_lines,args.input_shape,num_classes,epoch_length = args.UnFreeze_Epoch,\
                                    mosaic = args.mosaic, mixup = args.mixup,mosaic_prob = args.mosaic_prob,mixup_prob = args.mixup_prob,\
                                    train=True,special_aug_ratio=args.special_aug_ratio)
        val_dataset = YoloDataset(val_lines,args.input_shape,num_classes,epoch_length = args.UnFreeze_Epoch,\
                                    mosaic = False, mixup = False,mosaic_prob = 0,mixup_prob = 0,\
                                    train=False,special_aug_ratio=0)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle= False,)
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True
    #--------------------------------------------------#
    #   pin_memory锁页内存，可以加快训练速度
    #   drop_last当batch_size大小和数据大小不能整除时，
    #   batch_normalization运行时会报错,此时drop_last为
    #   true时，运行的时候会将最后不能整除的部分丢弃掉，防
    #   止BN层报错
    #--------------------------------------------------#         
        gen = DataLoader(train_dataset, shuffle=shuffle,batch_size=batch_size,num_workers=args.num_workers,
                         pin_memory=True, drop_last=True,collate_fn=yolo_dataset_collate,sampler=train_sampler)
        gen_val = DataLoader(val_dataset, shuffle=shuffle,batch_size=batch_size,num_workers=args.num_workers,
                         pin_memory=True, drop_last=True,collate_fn=yolo_dataset_collate,sampler=val_sampler)
        
        #----------------------#
        #   记录eval的map曲线
        #----------------------#
        if local_rank == 0:
            eval_callback   = EvalCallback(model, args.input_shape, anchors, args.anchors_mask, class_names, num_classes, val_lines, log_dir, args.Cuda, \
                                            eval_flag=True, period=20)
        else:
            eval_callback   = None
        
    #--------------------------------------------------#
    #   开始模型训练
    #--------------------------------------------------# 
        for epoch in range(args.Init_Epoch, args.UnFreeze_Epoch):
            #--------------------------------------------------#
            #   如果模型有冻结学习部分，则解冻，并设置参数
            #--------------------------------------------------# 
            if epoch >= args.Freeze_Epoch and not UnFreeze_flag and args.Freeze_Train:
                batch_size = args.Unfreeze_batch_size
                #--------------------------------------------------#
                #   判断当前batch_size，自适应调整学习率
                #--------------------------------------------------#  
                nbs = 64
                lr_limit_max = 1e-3 if args.optimizer_type == 'adam' else 5e-2
                lr_limit_min = 3e-4 if args.optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs* args.Init_lr, lr_limit_min),lr_limit_max)
                Min_lr_fit = min(max(batch_size /nbs * args.Min_lr,lr_limit_min * 1e-2),lr_limit_max * 1e-2)
                #--------------------------------------------------#
                #   获得学习率下降的公式
                #--------------------------------------------------#  
                lr_scheduler_func = get_lr_scheduler(args.lr_decay_type,Init_lr_fit,Min_lr_fit,args.UnFreeze_Epoch)
                for param in model.backbone.parameters():
                    param.requires_grad = True
                
                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size
                
                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集")
                
                if args.distributed:
                    batch_size = batch_size // ngpus_per_node
                
                gen = DataLoader(train_dataset, shuffle=shuffle,batch_size=batch_size,num_workers=args.num_workers,\
                         pip_memory=True, drop_last=True,collate_fn=yolo_dataset_collate,sampler=train_sampler)
                gen_val = DataLoader(val_dataset, shuffle=shuffle,batch_size=batch_size,num_workers=args.num_workers,\
                         pip_memory=True, drop_last=True,collate_fn=yolo_dataset_collate,sampler=val_sampler)

                UnFreeze_flag = True
            
            if args.distributed:
                train_sampler.set_epoch(epoch)
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val,\
                          gen, gen_val, args.UnFreeze_Epoch, args.Cuda, args.fp16, scaler, args.save_period, args.save_dir, local_rank)

        if local_rank == 0:
            loss_history.writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--Cuda',type=bool,help='是否使用GPU',
                        default=True)
    parser.add_argument('--distributed',type=bool,help='用于指定是否使用单机多卡分布式训练',
                        default=False)                    
    parser.add_argument('--sync_bn',type=bool,help='是否使用BN同步',
                        default=False)
    parser.add_argument('--fp16',type=bool,help='是否使用混合精度训练',
                        default=True)
    parser.add_argument('--classes_path',type=str,help='用于指向自己训练数据类别路径文件',
                        default='model_data/voc_classes.txt')
    parser.add_argument('--anchors_path',type=str,help='用于指向先验框的路径',
                        default='model_data/yolo_anchors.txt')
    parser.add_argument('--anchors_mask',help='用于帮助代码找到对应的先验框',
                        default=[[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    parser.add_argument('--model_path',type=str,help='用于存放整个模型的训练模型',
                        default='/yolo/model_data/yolo4_voc_weights.pth')
    parser.add_argument('--input_shape',help='输入的shape大小，一定要是32的倍数',
                        default=[416, 416])
    parser.add_argument('--backbone',type=str,help='用于选择使用的主干特征提取网络，可根据需求\
                        修改，主要有mobilenetv1,mobilenetv2,mobilenetv3,ghostnet,\
                        vgg,densenet121,densenet169,densenet201,resnet50,cspdarknet53',
                        default='cspdarknet53')
    parser.add_argument('--process_model',type=str,help='选用后处理方法，有fpn和spp_fpn两种可选',default="spp_fpn")                                        
    parser.add_argument('--mosaic',type=bool,help='马赛克数据增强',default=True)
    parser.add_argument('--mosaic_prob',type=float,help='每个step有多少概率使用mosaic数据增强，\
                        默认50%',default=0.5)
    parser.add_argument('--mixup',type=bool,help='是否使用mixup数据增强，仅在mosaic=True时有效'\
                         ,default=True)
    parser.add_argument('--mixup_prob',type=float,help='有多少概率在mosaic后使用mixup数据增强，默认50%\
                        ',default=0.5)
    parser.add_argument('--label_smoothing',type=float,help='标签平滑。一般0.01以下。如0.01、0.005\
                        ',default=0.005)
    parser.add_argument('--special_aug_ratio',type=float,help='参考YoloX，由于Mosaic生成的训练图片，\
                        远远脱离自然图片的真实分布。当mosaic=True时，本代码会在special_aug_ratio范围内开启mosaic。\
    #                   默认为前70%个epoch，100个世代会开启70个世代',default=0.7)                    
    parser.add_argument('--pretrained',type=bool,help='是否使用主干网络的预训练权重，此控制的\
                         是主干的权重',default=False)
    parser.add_argument('--Init_Epoch',type=int,help='模型当前开始的训练世代',
                        default=0)
    parser.add_argument('--Freeze_Epoch',type=int,help='模型冻结训练的Freeze_Epoch',
                        default=0)
    parser.add_argument('--Freeze_batch_size',type=int,help='模型冻结训练的batch_size',
                        default=64)
    parser.add_argument('--UnFreeze_Epoch',type=int,help='模型总共训练的epoch',
                        default=200)
    parser.add_argument('--Unfreeze_batch_size',type=int,help='模型在解冻后的batch_size',
                        default=16)
    parser.add_argument('--Freeze_Train',type=bool,help='是否进行冻结训练',
                        default=False)
    parser.add_argument('--Init_lr',type=float,help='模型的最大学习率',
                        default=5e-3)
    parser.add_argument('--Min_lr',type=float,help='模型的最小学习率',
                        default=1e-5)
    parser.add_argument('--optimizer_type',type=str,help='使用到的优化器种类，可选的有adam、sgd',
                        default='adam')
    parser.add_argument('--momentum',type=int,help='优化器内部使用到的momentum参数',
                        default=0.937)
    parser.add_argument('--weight_decay',type=float,help='权值衰减，可防止过拟合',
                        default=0)
    parser.add_argument('--lr_decay_type',type=str,help='使用到的学习率下降方式，可选的有step、cos',
                        default='cos')
    parser.add_argument('--focal_loss',type=bool,help='是否使用Focal Loss平衡正负样本',
                        default=True)
    parser.add_argument('--focal_alpha',type=float,help='Focal Loss的正负样本平衡参数',
                        default=0.25)
    parser.add_argument('--focal_gamma',type=int,help='Focal Loss的难易分类样本平衡参数',
                        default=2)
    parser.add_argument('--save_period',type=int,help='多少个epoch保存一次权值',
                        default=10)
    parser.add_argument('--save_dir',type=str,help='多少个epoch保存一次权值',
                        default='logs')
    parser.add_argument('--num_workers',type=int,help='用于设置是否使用多线程读取数据',
                        default=0)
    parser.add_argument('--train_annotation_path',type=str,help='用于保存训练数据路径文件',
                        default='2007_train.txt')
    parser.add_argument('--val_annotation_path',type=str,help='用于保存验证数据路径文件',
                        default='2007_val.txt')
    args = parser.parse_args()
    train(args)