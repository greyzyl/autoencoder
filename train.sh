CUDA_VISIBLE_DEVICES=0 1 python train.py  --save_path workdir/(6-23实验)1024_AEwithGPP_GPPW1e-2_preceptual_加深网络_downsample32 
            --data_root /home/fdu02/fdu02_dir/zyl/code/diffusers-main/data/vary_data 
            --batch_size 1 --epochs 15 --learning_rate 1e-5 --save_eval_iteration 500 
            --downsample_rate 32 --gradien_loss_weight 1e-3 --resolution 1024