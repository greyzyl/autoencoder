import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import ImageDataset
from model.model_resnet import UNet
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
        batch_features = img.cuda()  # 获取当前批次的图像数据
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
def model_init(pth_path):
    model = UNet(n_channels=3,n_classes=3).cuda()
    checkpoint=torch.load(pth_path)
    model.load_state_dict(checkpoint)
    return model
if __name__=="__main__":
    model=model_init()
    resolution=(512,512)
    transform= transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )
    criterion = nn.MSELoss().cuda()
    img=Image.open('')
    loss,input,recon=infer_single_img(model,img,transform,criterion)
    print(loss)
    cv2.imwrite('t.png',recon)

