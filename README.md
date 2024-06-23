## Contents

- Install
  - Base Environment
  - Package Install
- Download Pretrained Weight
- Train
- Infer



## Install
1. Base Environment
```
Python 3.9
torch  2.3.0
```
We suggest that you use the conda environment to install Python and torch in advance.

2. Package Install
- Note: if you have installed Python and torch already, you need to ignore the torch, torchaudio, torchvision in the requirement.txt, manually. Otherwise, the torch, torchaudio, torchvision will be reinstalled again.

```sh
git clone http://gitslab.yiqing.com/declare/about.git
cd autoencoder
pip install -r requirements.txt
```

# dataset

data_root

​	--train

​		--0.jpg(.png)

​		...

​	--test

​		--0.jpg(.png)

​		...

## Train
1. Run the training script
   - there are some template scripts in the train.sh
   ```sh
   CUDA_VISIBLE_DEVICES=0,1 python train.py  --save_path workdir/(6-23实验)1024_AEwithGPP_GPPW1e-2_preceptual_加深网络_downsample32 
               --data_root /home/fdu02/fdu02_dir/zyl/code/diffusers-main/data/vary_data 
               --batch_size 1 --epochs 15 --learning_rate 1e-5 --save_eval_iteration 500 
               --downsample_rate 32 --gradien_loss_weight 1e-3 --resolution 1024
   ```

## Infer
1. Run the infer_single_img script

   - there are some template scripts in the infer.sh

   ```sh
   CUDA_VISIBLE_DEVICES=0 python infer.py  --model_path  /home/fdu02/fdu02_dir/zyl/code/AE_pure/workdir/(6-13实验)pureAE_加深网络/checkpoints/bestmodel.pth
               --img_path /home/fdu02/fdu02_dir/zyl/code/AE_pure/微信图片_20240615113816.jpg
               --output_name t.png
   			--resolution 1024
   ```



