CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python train.py  --save_path workdir/"(6-28实验)1024_AEwithGPP_GPPW1e-2_加深网络_加大channel_downsample64" --data_root /home/fdu02/fdu02_dir/zyl/code/diffusers-main/data/vary_data --batch_size 6 --epochs 15 --learning_rate 1e-5 --save_eval_iteration 500 --downsample_rate 64 --gradien_loss_weight 1e-2 --resolution 1024