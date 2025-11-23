from net.modules import *
import torch
from net.encoder import SwinTransformerBlock
import datetime


class BasicLayer(nn.Module):

    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, upsample=None, ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for _, blk in enumerate(self.blocks):
            x = blk(x)

        if self.upsample is not None:
            x = self.upsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
            print("blk.flops()", blk.flops())
        if self.upsample is not None:
            flops += self.upsample.flops()
            print("upsample.flops()", self.upsample.flops())
        return flops

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        for _, blk in enumerate(self.blocks):
            blk.input_resolution = (H, W)
            blk.update_mask()
        if self.upsample is not None:
            self.upsample.input_resolution = (H, W)


class SwinJSCC_Decoder(nn.Module):
    def __init__(self, img_size, embed_dims, depths, num_heads, C,
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, patch_norm=True):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dims = embed_dims
        self.patch_norm = patch_norm
        self.num_features = C
        self.mlp_ratio = mlp_ratio
        self.H = img_size
        self.W = img_size
        # 这在算最小的特征图分辨率，后面在解码阶段，要对这个patches_resolution逐倍增加
        self.patches_resolution = (img_size // 2 ** len(depths), img_size // 2 ** len(depths))

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dims[i_layer]),
                               out_dim=int(embed_dims[i_layer + 1]) if (i_layer < self.num_layers - 1) else 3,
                               input_resolution=(self.patches_resolution[0] * (2 ** i_layer),
                                                 self.patches_resolution[1] * (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               norm_layer=norm_layer,
                               upsample=PatchReverseMerging)
            self.layers.append(layer)
            print("Decoder ", layer.extra_repr())
        if C != None:
            self.head_list = nn.Linear(C, embed_dims[0])
        self.apply(self._init_weights)
        self.hidden_dim = int(self.embed_dims[0] * 1.5)
        self.layer_num = 7

    def forward(self, x):
        x = self.head_list(x)
        for i_layer, layer in enumerate(self.layers):
            x = layer(x)
        B, L, N = x.shape
        x = x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def flops(self):
        flops = 0
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        return flops

    def update_resolution(self, H, W):
        # 这个函数约束了整个网络的patch必须是2的倍数，否则这里恢复的H和W就会出问题
        self.input_resolution = (H, W)
        self.H = H * 2 ** len(self.layers)
        self.W = W * 2 ** len(self.layers)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H * (2 ** i_layer),
                                    W * (2 ** i_layer))


def create_decoder(**kwargs):
    model = SwinJSCC_Decoder(**kwargs)
    return model


def build_model(args):
    input_image = torch.ones([1, 1536, 1256]).to(args.device)
    model = SwinJSCC_Decoder(img_size=args.image_dims, embed_dims=args.decoder_embed_dims,
                             depths=args.decoder_depth, num_heads=args.decoder_heads, C=args.C,
                             window_size=args.window_size, mlp_ratio=args.mlp_ratio, qkv_bias=args.qkv_bias,
                             qk_scale=args.qk_scale, norm_layer=nn.LayerNorm, patch_norm=args.patch_norm)
    model.to(args.device)
    model(input_image)
    print("TOTAL FLOPs {}G".format(model.flops() / 10 ** 9))
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("TOTAL Params {}M".format(num_params / 10 ** 6))
