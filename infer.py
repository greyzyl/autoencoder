import argparse
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import ImageDataset
from model.model_resnet import UNet_ds16, UNet_ds32
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import pytorch_warmup as warmup
def infer_single_img( model, img,transform, criterion):
    model.eval()
    img=transform(img)
    total = 0
    test_examples = None
    reconstruction=None
    val_loss=0
    # 通过循环遍历测试数据加载器，获取一个批次的图像数据
    with torch.no_grad():  # 使用 torch.no_grad() 上下文管理器，确保在该上下文中不会进行梯度计算
        batch_features = img.unsqueeze(0).cuda()  
        total += batch_features.size(0)
        test_examples = batch_features.cuda()
        reconstruction = model(test_examples)  # 使用训练好的自编码器模型对测试数据进行重构，即生成重构的图像
        
        val_loss+=criterion(reconstruction, batch_features)
            # break     
        val_loss=val_loss.cuda()
        total=torch.tensor(total).cuda()
        avg_loss = val_loss.item() / total.item()

        number = 1  # 设置要显示的图像数量
        input_img,recon_img=test_examples[0].permute(1,2,0).detach().cpu().numpy().reshape(512, 512,3),reconstruction[0].permute(1,2,0).cpu().numpy().reshape(512, 512,3)
    return val_loss,input_img,recon_img
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def initialize_argparse():
    """
    Initialize argparse for command line arguments.
    
    Returns:
        argparse.ArgumentParser: An argument parser object which can be used to parse the command line arguments.
    """
    parser = argparse.ArgumentParser(description="A simple script to process an input file and generate an output.")
    
    # 添加命令行参数
    parser.add_argument("--model_path",help="Path to the checkpoint",default="/home/fdu02/fdu02_dir/zyl/code/AE_pure/workdir/(6-13实验)pureAE_加深网络/checkpoints/bestmodel.pth", required=True, type=str)
    parser.add_argument("--img_path",help="Path to the data dir",default="/home/fdu02/fdu02_dir/zyl/code/AE_pure/微信图片_20240615113816.jpg", required=True, type=str)
    parser.add_argument("--output_name",default='t.png',   required=True, type=str)
    parser.add_argument("--downsample_rate",default=32,  required=True, type=int)
    parser.add_argument("--resolution",default=1024,   required=True, type=int)

    
    # 解析之前可以添加更多参数或设置
    
    return parser

def model_init(pth_path,downsample_rate):
    setup(0,1)
    if downsample_rate==32:
        model = UNet_ds32(n_channels=3,n_classes=3).cuda()
    elif downsample_rate==16:
        model = UNet_ds16(n_channels=3,n_classes=3).cuda()
    ddp_model = DDP(model, device_ids=[0])
    checkpoint=torch.load(pth_path)
    ddp_model.load_state_dict(checkpoint)
    return ddp_model

if __name__=="__main__":
    parser=initialize_argparse()
    args=parser.parse_args()
    model_path=args.model_path
    img_path=args.img_path
    output_name=args.output_name
    downsample_rate=args.downsample_rate
    resolution=args.downsample_rate
    model=model_init(model_path,downsample_rate)
    resolution=(resolution,resolution)
    transform= transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )
    criterion = nn.MSELoss().cuda()
    img=Image.open(img_path)
    loss,input,recon=infer_single_img(model,img,transform,criterion)
    print(loss)
    plt.figure(figsize=(11, 10)) 
    plt.imshow(recon)
    plt.gray()
    plt.axis('off')
    plt.savefig(output_name)


