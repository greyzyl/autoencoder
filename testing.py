import lpips
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores ,lpips=False
# loss_fn_vgg = lpips.LPIPS(net='vgg',lpips=False) # closer to "traditional" perceptual loss, when used for optimization

import torch
from torchvision import datasets, transforms
from PIL import Image
resolution=(512,512)
transform= transforms.Compose(
    [
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ]
)
gt = Image.open('0.jpg').convert('RGB')
gt=transform(gt).unsqueeze(0)
img0 = Image.open('downsample16_wrose.png').convert('RGB')
img0=transform(img0).unsqueeze(0)
img1 = Image.open('downsaple16_best.png').convert('RGB')
img1=transform(img1).unsqueeze(0)
d0 = loss_fn_alex(gt, img0,normalize=True)
d1 = loss_fn_alex(gt, img1,normalize=True)
print(d0)
print(d1)