from net.decoder import *
from net.encoder import *
from loss.distortion import Distortion
from net.channel import Channel
from random import choice
import torch.nn as nn


class SwinJSCC(nn.Module):
    def __init__(self, args):
        super(SwinJSCC, self).__init__()
        self.encoder = SwinJSCC_Encoder(img_size=args.image_dims, patch_size=args.patch_size, in_chans=args.in_channel,
                                        embed_dims=args.encoder_embed_dims, depths=args.encoder_depth,
                                        num_heads=args.encoder_heads, C=args.C, window_size=args.window_size,
                                        mlp_ratio=args.mlp_ratio, qkv_bias=args.qkv_bias, qk_scale=args.qk_scale,
                                        norm_layer=nn.LayerNorm, patch_norm=args.patch_norm)
        self.decoder = SwinJSCC_Decoder(img_size=args.image_dims, embed_dims=args.decoder_embed_dims,
                                        depths=args.decoder_depth, num_heads=args.decoder_heads, C=args.C,
                                        window_size=args.window_size, mlp_ratio=args.mlp_ratio, qkv_bias=args.qkv_bias,
                                        qk_scale=args.qk_scale, norm_layer=nn.LayerNorm, patch_norm=args.patch_norm)
        self.distortion_loss = Distortion(args)
        self.channel = Channel(args)
        self.pass_channel = args.pass_channel
        self.squared_difference = torch.nn.MSELoss(reduction='none')
        self.H = self.W = 0
        self.multiple_snr = args.multiple_snr
        self.downsample = args.downsample

    def distortion_loss_wrapper(self, x_gen, x_real):
        distortion_loss = self.distortion_loss.forward(x_gen, x_real)
        return distortion_loss

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature

    def forward(self, input_image, given_SNR=None):
        B, _, H, W = input_image.shape

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H = H
            self.W = W

        if given_SNR is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR

        feature = self.encoder(input_image)
        CBR = feature.numel() / 2 / input_image.numel()
        if self.pass_channel:
            noisy_feature = self.feature_pass_channel(feature, chan_param)
        else:
            noisy_feature = feature
        recon_image = self.decoder(noisy_feature)

        mse = self.squared_difference(input_image * 255., recon_image.clamp(0., 1.) * 255.)
        loss_G = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.))
        return recon_image, CBR, chan_param, mse.mean(), loss_G.mean()
