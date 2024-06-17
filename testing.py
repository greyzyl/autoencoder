import torch
from Loss.regularizer import DiagonalGaussianRegularizer
from model.SD_VAE import AutoencodingEngine, Decoder, Encoder


if __name__=='__main__':
    encoder=Encoder(
        attn_type= 'vanilla',
        double_z= True,
        z_channels= 8,
        resolution= 256,
        in_channels= 3,
        out_ch= 3,
        ch= 128,
        ch_mult= [1, 2, 4, 4],
        num_res_blocks= 2,
        attn_resolutions= [],
        dropout= 0.0,
        resamp_with_conv=False
    ).cuda()
    decoder=Decoder(
        attn_type= 'vanilla',
        double_z= True,
        z_channels= 8,
        resolution= 256,
        in_channels= 3,
        out_ch= 3,
        ch= 128,
        ch_mult= [1, 2, 4, 4],
        num_res_blocks= 2,
        attn_resolutions= [],
        dropout= 0.0,
        resamp_with_conv=False
    ).cuda()
    regularizer=DiagonalGaussianRegularizer()
    VAE=AutoencodingEngine(encoder=encoder,decoder=decoder,regularizer=regularizer).cuda()
    x=torch.rand((1,3,256,256)).cuda()
    z,recon=VAE(x)
    print(z.shape)
    print(recon.shape)


