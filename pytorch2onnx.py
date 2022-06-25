from tabnanny import verbose
from typing import OrderedDict
import onnx
import onnxsim
import torch
import argparse
import os
from nets.yolo import YoloBody

#---------------------------------#
# Remove 'module.' of dataparallel
#---------------------------------#
def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]
        new_state_dict[name] = v
    return new_state_dict

def Toonnx(args):
    model = YoloBody(args.anchors_mask,args.num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(args.model_path,map_location=device)
    fix_state_dict = fix_model_state_dict(state_dict)
    model = model.load_state_dict(fix_state_dict)
    model = model.eval()
    input_image = torch.ones(1,3,args.input_shape)
    input_image = input_image.to(device=device)

    #---------------------------------#
    #   Export the model
    #  dynamic_axes = None该参数控制是否
    #   动态输入输出
    #---------------------------------#
    print(f'Starting export with onnx {onnx.__version__}')
    torch.onnx.export(model,input_image,os.path.splitext(args.model_path)[0] + '.onnx',verbose=True,
                      input_names = args.input_layer_name,output_names = args.output_layer_name,dynamic_axes = None)
    
    #---------------------------------#
    # Checks
    #---------------------------------#
    if args.check:
        model_onnx = onnx.load(os.path.splitext(args.model_path)[0] + '.onnx')
        onnx.checker.check_model(model_onnx)

    #---------------------------------#
    # Simplify onnx
    #---------------------------------#
    if args.simplify:
        print(f'Simplifying with onnx-simplifter {onnxsim.__version__}')
        model_onnx = onnx.load(os.path.splitext(args.model_path)[0] + '.onnx')
        model_onnx, check = onnxsim.simplify(model_onnx,dynamic_input_shape=False,input_shapes=None)
        onnx.save(model_onnx,os.path.splitext(args.model_path)[0] + '_simplify_' + '.onnx')
    print('Onnx model save as {}'.format((args.model_path)[0] + '_simplify_' + '.onnx'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_layer_name',type=str,help='用来指定转换成onnx模型之后第一层的名称\
                        ，方便在之后的TVM框架的推理使用',default='input')
    parser.add_argument('--output_layer_name',type=str,help='用来指定转换成onnx模型之后最后一层的名称\
                        ，方便在之后的TVM框架的推理使用',default='output')
    parser.add_argument('--input_shape',help='用来指定原网络模型的输入大小，必须和训练时候输入保持一致',
                        default=[416,416])
    parser.add_argument('--anchors_mask',help='anchors_mask用于帮助代码找到对应的先验框',
                        default=[[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    parser.add_argument('--num_classes',type=int,help='根据自己的数据集或者下载的开源数据集的类别进行确定',
                        default=20)
    parser.add_argument('--model_path',type=int,help='自己训练好的模型文件路径',
                        default='model_data/yolo4_weights.pth')
    parser.add_argument('--simplify',type=bool,help='是否对生成的onnx模型进行简化',
                        default= True)
    parser.add_argument('--check',type=bool,help='是否对生成的onnx模型进行简化',
                        default= True)
    args = parser.parse_args()
    Toonnx(args)
    
