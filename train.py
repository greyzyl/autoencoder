import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import ImageDataset
from loss.loss import GradientPriorLoss
from model.model_resnet import UNet_ds16, UNet_ds32, UNet_ds64_deep_channel, UNet_ds64_ori
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_warmup as warmup
import lpips
# save_path='workdir/(6-23实验)1024_AEwithGPP_GPPW1e-2_preceptual_加深网络_downsample32'
class Log():
    def __init__(self,file_path, sep=' ', end='\n', file_mode='a'):
        self.file_path=file_path 
        self.sep=sep
        self.end=end
        self.mode=file_mode
    def __call__(self, *args) :
        with open(self.file_path, self.mode) as file:
            print(*args, sep=self.sep, end=self.end, file=file)
def append_to_file(file_path, *args, sep=' ', end='\n', file_mode='a'):
    """
    将内容追加到指定的txt文件中。
    
    参数:
    - file_path: str, 要写入的文件路径。
    - *args: 可变数量的参数，这些参数将会被转换成字符串并按照sep分隔。
    - sep: str, 分隔符，默认为空格。
    - end: str, 结束符，默认为换行符。
    - file_mode: str, 文件打开模式，默认为追加模式'a'。
    
    示例:
    append_to_file('log.txt', 'Hello,', 'World!', sep=' ', end='!\n')
    """
    with open(file_path, file_mode) as file:
        print(*args, sep=sep, end=end, file=file)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def validate(rank, world_size, model, dataloader, criterion,epoch,iteration,log,save_path,resolution):
    model.eval()
    total = 0
    test_examples = None
    reconstruction=None
    val_loss=0
    if rank == 0:
        # 通过循环遍历测试数据加载器，获取一个批次的图像数据
        with torch.no_grad():  # 使用 torch.no_grad() 上下文管理器，确保在该上下文中不会进行梯度计算
            for i,batch_features in tqdm(enumerate(dataloader)):  # 历测试数据加载器中的每个批次的图像数据
                batch_features = batch_features[0].to(rank)  # 获取当前批次的图像数据
                total += batch_features.size(0)
                test_examples = batch_features.to(
                    rank)  # 将当前批次的图像数据转换为大小为 (批大小, 784) 的张量，并加载到指定的设备（CPU 或 GPU）上
                reconstruction = model(test_examples)  # 使用训练好的自编码器模型对测试数据进行重构，即生成重构的图像
                
                val_loss+=criterion(reconstruction, batch_features)
                # if i>5:
                #     break   
        val_loss=val_loss.to(rank)
        total=torch.tensor(total).to(rank)
        avg_loss = val_loss.item() / total.item()
        print(f'epoch: {epoch} \titeration: {iteration} \tValidation Loss: {avg_loss:.8f}')
        log(f'epoch: {epoch} \titeration: {iteration}\tValidation Loss: {avg_loss:.8f}')

        number = 2  # 设置要显示的图像数量

        plt.figure(figsize=(22, 20))  # 创建一个新的 Matplotlib 图形，设置图形大小为 (20, 4)1
        for index in range(number):  # 遍历要显示的图像数量
            # 显示原始图
            ax = plt.subplot(2, number, index + 1)
            plt.imshow(test_examples[index].permute(1,2,0).detach().cpu().numpy().reshape( resolution[0],  resolution[1],3))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
            # 显示重构图
            ax = plt.subplot(2, number, index + 1 + number)
            plt.imshow(reconstruction[index].permute(1,2,0).cpu().numpy().reshape( resolution[0],  resolution[1],3))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        print(os.path.join(save_path,'val_vis',f'reconstruction_results_epoch{epoch}_loss={val_loss}.png'))
        plt.savefig(os.path.join(save_path,'val_vis',f'reconstruction_results_epoch{epoch}_loss={val_loss}.png'),dpi=300)  # 保存图像
    return val_loss
    

def train2(rank, world_size,
           save_path,data_root,
           batch_size,epochs,save_eval_iteration,learning_rate,
           downsample_rate,resolution,
           gradien_loss_weight):
    #init
    os.makedirs(os.path.join(save_path,'checkpoints'),exist_ok=True)
    os.makedirs(os.path.join(save_path,'val_vis'),exist_ok=True)
    log=Log(os.path.join(save_path,'log.txt'))

    setup(rank, world_size)
    center_crop =False
    random_flip=False
    torch.cuda.set_device(rank)



    print('build dataset')
    train_transform = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            # transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )
    '''
    cifar10数据集
    '''
    # train_dataset = torchvision.datasets.CIFAR10(
    #     root="./data", train=True, transform=train_transform, download=True
    # )  # 加载 MNIST 数据集的训练集，设置路径、转换和下载为 True
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=batch_size, sampler=train_sampler,num_workers=10
    # )  # 创建一个数据加载器，用于加载训练数据，设置批处理大小和是否随机打乱数据
    # test_dataset = torchvision.datasets.CIFAR10(
    #     root="./data", train=False, transform=validation_transform, download=True
    # )  # 加载 MNIST 测试数据集
 
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=10, shuffle=False
    # )  # 创建一个测试数据加载器
    '''
    vary数据集
    '''
    # data_root='/home/fdu02/fdu02_dir/zyl/code/diffusers-main/data/vary_data'
    train_dataset = ImageDataset(data_root,train_transform,mode='train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,num_workers=4
    )  # 创建一个数据加载器，用于加载训练数据，设置批处理大小和是否随机打乱数据

    test_dataset = ImageDataset(data_root,validation_transform,mode='test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,num_workers=1
    )  # 创建一个测试数据加载器
    print('build model')
    if downsample_rate==32:
        model = UNet_ds32(n_channels=3,n_classes=3).to(rank)
    elif downsample_rate==16:
        model = UNet_ds16(n_channels=3,n_classes=3).to(rank)
    elif downsample_rate==64:
        # model = UNet_ds64_ori(n_channels=3,n_classes=3).to(rank)
        model = UNet_ds64_deep_channel(n_channels=3,n_classes=3).to(rank)
    print('build ddp')
    ddp_model = DDP(model, device_ids=[rank])
    print('build opt')
    optimizer = optim.Adam(ddp_model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    print('build loss')
    # 使用均方误差（MSE）损失函数
    criterion = nn.MSELoss().to(rank)
    GPP_criterion=GradientPriorLoss().to(rank)

    # loss_fn_vgg = lpips.LPIPS(net='vgg').to(rank) # best forward scores
    print('start train')

    iteration=0
    best_val_loss=9999
    save_recon_loss=0
    save_gpp_loss=0
    for epoch in range(epochs):
        loss=0
        save_loss=0
        for _,(batch_features, _) in enumerate(tqdm(train_loader)):
            
            # 将小批数据变形为 [N, 784] 矩阵，并加载到 CPU 设备
            batch_features = batch_features.to(rank)
            # print(batch_features.shape)
            # 梯度设置为 0，因为 torch 会累加梯度
            optimizer.zero_grad()
             
            # 计算重构
            outputs = ddp_model(batch_features)
 
            # 计算训练重建损失
            recon_loss=criterion(outputs, batch_features)
            train_loss = recon_loss
            GPP_loss=GPP_criterion(outputs, batch_features)*gradien_loss_weight
            train_loss+=GPP_loss
            # print(train_loss)
            # train_loss+=loss_fn_vgg(outputs, batch_features).mean()*0.25
            # 计算累积梯度
            train_loss.backward()
 
            # 根据当前梯度更新参数
            optimizer.step()
            with warmup_scheduler.dampening():
                lr_scheduler.step()
 
            # 将小批量训练损失加到周期损失中
            loss += train_loss.item()
            save_recon_loss+=recon_loss
            save_gpp_loss+=GPP_loss
            save_loss += train_loss.item()
            if iteration % save_eval_iteration == 0 and rank == 0 and iteration>0:
                print("interation : {}, train loss = {:.8f}".format(iteration, save_loss/save_eval_iteration))
                if rank==0:
                    log("interation : {}, train loss = {:.8f}".format(iteration, save_loss/save_eval_iteration))
                    log("interation : {}, train recon loss = {:.8f}".format(iteration, save_recon_loss/save_eval_iteration))
                    log("interation : {}, train GPP loss = {:.8f}".format(iteration, save_gpp_loss/save_eval_iteration))
                save_loss=0
                save_recon_loss=0
                save_gpp_loss=0
                validate(rank, world_size, ddp_model, test_loader, criterion,epoch,iteration,log,save_path,resolution)
                checkpoint_path = os.path.join(save_path,'checkpoints',f'checkpoint_iter_{iteration}.pth')
                torch.save(ddp_model.state_dict(), checkpoint_path)
                print(f'Saved checkpoint: {checkpoint_path}')
            iteration+=1
            # if iteration>5:
            #     break
        loss = loss / len(train_loader)
        print("epoch : {}/{}, loss = {:.8f}".format(epoch + 1, epochs, loss))
        log("epoch : {}/{}, train loss = {:.8f}".format(epoch + 1, epochs, loss))
        val_loss=validate(rank, world_size, ddp_model, test_loader, criterion,epoch,iteration,log,save_path,resolution)
        if rank == 0 and val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_path,'checkpoints','bestmodel.pth')
            torch.save(ddp_model.state_dict(), best_model_path)
            print(f'Saved best model: {best_model_path}, Validation Loss: {val_loss:.4f}')
        # break
    cleanup()
def initialize_argparse():
    """
    Initialize argparse for command line arguments.
    
    Returns:
        argparse.ArgumentParser: An argument parser object which can be used to parse the command line arguments.
    """
    parser = argparse.ArgumentParser(description="A simple script to process an input file and generate an output.")
    
    # 添加命令行参数
    parser.add_argument("--save_path",
                        default='workdir/(6-23实验)1024_AEwithGPP_GPPW1e-2_preceptual_加深网络_downsample32', 
                        help="Path to the save dir", required=True, type=str)
    parser.add_argument("--data_root",
                        default='/home/fdu02/fdu02_dir/zyl/code/diffusers-main/data/vary_data',
                          help="Path to the data dir", required=True, type=str)
    parser.add_argument("--batch_size",default=10 ,  required=True, type=int)
    parser.add_argument("--epochs",default=15,   required=True, type=int)
    parser.add_argument("--learning_rate", "-lr", default=1e-5, required=True, type=float)
    parser.add_argument("--downsample_rate",default=32,  required=True, type=int)
    parser.add_argument("--gradien_loss_weight",default=1e-3,   required=True, type=float)
    parser.add_argument("--resolution",default=1024,   required=True, type=int)
    parser.add_argument("--save_eval_iteration",  default=500, required=True, type=int)
    
    
    # 解析之前可以添加更多参数或设置
    
    return parser

def main():
    parser=initialize_argparse()
    args=parser.parse_args()
    save_path=args.save_path
    data_root=args.data_root
    batch_size=args.batch_size
    epochs=args.epochs
    downsample_rate=args.downsample_rate
    gradien_loss_weight=args.gradien_loss_weight
    resolution=args.resolution
    resolution=(resolution,resolution)
    save_eval_iteration=args.save_eval_iteration
    learning_rate=args.learning_rate
    print('start')
    world_size = torch.cuda.device_count()
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    torch.multiprocessing.spawn(train2, 
                                args=(
                                        world_size,
                                        save_path,data_root,
                                        batch_size,epochs,save_eval_iteration,learning_rate,
                                        downsample_rate,resolution,
                                        gradien_loss_weight
                                    ), 
                                nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
