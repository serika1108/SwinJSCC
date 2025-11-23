from net.modules import *
import torch


class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # 图像分辨率小于窗口大小就不做切分
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        # shift_size代表移动窗口的步长
        if self.shift_size > 0:
            # 移动窗口注意力，需要额外计算mask
            H, W = self.input_resolution
            # 初始化一个HW大小的掩码图
            img_mask = torch.zeros((1, H, W, 1))
            # 尽管发生了窗口移动，但是绝大多数的patch还是在完整的窗口里的，只有极少部分的窗口需要合并后做注意力
            # 理论上可以先按照位移划分窗口，再把边缘的合并在一起，但这样太繁琐，代码的做法是先计算那些patch是要合并的，再划分窗口

            # 这里的slice是在创建一个切片对象，等价于[0:-window_size]，这个切片是为了后面对掩码图HW索引，这里一共做了9个切片
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            # 用切片做索引，为掩码图标号。实际上发生窗口位移后的图像里，在[0:-window_size, 0:-window_size]范围内的窗口都不需要做合并
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            # 若图像是8*8的，windows是4*4的，这样划分的结果如下
            # 0 0 0 0 | 1 1 | 2 2
            # 0 0 0 0 | 1 1 | 2 2
            # 0 0 0 0 | 1 1 | 2 2
            # 0 0 0 0 | 1 1 | 2 2
            # --------+-----+----
            # 3 3 3 3 | 4 4 | 5 5
            # 3 3 3 3 | 4 4 | 5 5
            # --------+-----+----
            # 6 6 6 6 | 7 7 | 8 8
            # 6 6 6 6 | 7 7 | 8 8
            # 只有右半，下半和右下角需要做掩码

            # 给mask划分窗口
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            # 去掉最后1的维度，然后把window_size展开，得到N个token的长度，对应于每次窗口注意力的patch数量
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            # 等价于(nW, 1, N)-(nW, N, -1)，通过广播将会计算N个元素两两相减的结果。标号代表了所述的窗口，因此编号两两相减将得到对哪些patch做mask
            # 如果在一个窗口内，标号则相同，相减为0，不在一个窗口内相减将不为零
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            # 不为零的位置赋予很大的负数，这将在计算注意力分数的时候让网络忽略这个位置的注意力值
            # tensor.masked_fill(mask, value)，对tensor中mask为True的位置，用value覆盖
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):

        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 做循环平移
        if self.shift_size > 0:
            # torch.roll(tensor, shifts, dims)在指定维度做循环平移，行列分别循环移动shift_size
            # 经过循环平移后，shifted_x的位置信息将和attn_mask相同，一种巧妙的方法做了不同窗口之间patch的合并
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 窗口划分
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        # 变换为token数
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        B_, N, C = x_windows.shape

        # merge windows
        attn_windows = self.attn(x_windows, add_token=False, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # 要恢复循环平移造成的影响
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

    def update_mask(self):
        # 这个函数的意义是，mask取决于输入的分辨率，如果分辨率发生变化，mask的维度也要改。为了让block支持动态分辨率，这里写了update函数
        if self.shift_size > 0:
            H, W = self.input_resolution
            device = next(self.parameters()).device

            img_mask = torch.zeros((1, H, W, 1), device=device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.attn_mask = attn_mask
        else:
            pass


class BasicLayer(nn.Module):
    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm,
                 downsample=None):

        super().__init__()
        # 输入维度
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        # 根据深度构建Swin Transformer模块
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=out_dim,
                                 # 因为在forward里面，先downsample的，再经过block的，所以输入分辨率要除2
                                 input_resolution=(input_resolution[0] // 2, input_resolution[1] // 2),
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # 这个是patch merge操作，也就是下采样
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        for _, blk in enumerate(self.blocks):
            x = blk(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def update_resolution(self, H, W):
        for _, blk in enumerate(self.blocks):
            # 分辨率发生变化，需要更新掩码
            blk.input_resolution = (H, W)
            blk.update_mask()
        if self.downsample is not None:
            self.downsample.input_resolution = (H * 2, W * 2)


class SwinJSCC_Encoder(nn.Module):
    def __init__(self, img_size, patch_size, in_chans,
                 embed_dims, depths, num_heads, C,
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, patch_norm=True):
        super().__init__()
        self.num_layers = len(depths)
        self.patch_norm = patch_norm
        self.num_features = C
        self.mlp_ratio = mlp_ratio
        self.embed_dims = embed_dims
        self.in_chans = in_chans
        self.patch_size = patch_size
        # 这里其实没必要单独赋值的
        self.patches_resolution = (img_size, img_size)
        # self.H = img_size[0] // (2 ** self.num_layers)
        # self.W = img_size[1] // (2 ** self.num_layers)
        self.patch_embed = PatchEmbed(self.patches_resolution, self.patch_size, in_chans, embed_dims[0])
        self.hidden_dim = int(self.embed_dims[len(embed_dims) - 1] * 1.5)
        self.layer_num = layer_num = 7

        # 构建网络骨干
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dims[i_layer - 1]) if i_layer != 0 else 3,
                               out_dim=int(embed_dims[i_layer]),
                               input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                                 self.patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               norm_layer=norm_layer,
                               downsample=PatchMerging if i_layer != 0 else None)
            print("Encoder ", layer.extra_repr())
            self.layers.append(layer)
        self.norm = norm_layer(embed_dims[-1])
        # 模型的输出头，取决于最终想要的输出维度C
        if C != None:
            self.head_list = nn.Linear(embed_dims[-1], C)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.patch_embed(x)
        for i_layer, layer in enumerate(self.layers):
            x = layer(x)
        x = self.norm(x)
        x = self.head_list(x)
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
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        return flops

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H // (2 ** (i_layer + 1)),
                                    W // (2 ** (i_layer + 1)))


def create_encoder(**kwargs):
    model = SwinJSCC_Encoder(**kwargs)
    return model


def build_model(args):
    input_image = torch.ones([1, 256, 256]).to(args.device)
    model = SwinJSCC_Encoder(img_size=args.image_dims, patch_size=args.patch_size, in_chans=args.in_channel,
                             embed_dims=args.encoder_embed_dims, depths=args.encoder_depth,
                             num_heads=args.encoder_heads, C=args.C, window_size=args.window_size,
                             mlp_ratio=args.mlp_ratio, qkv_bias=args.qkv_bias, qk_scale=args.qk_scale,
                             norm_layer=nn.LayerNorm, patch_norm=args.patch_norm)
    model.to(args.device)
    model(input_image)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("TOTAL Params {}M".format(num_params / 10 ** 6))
    print("TOTAL FLOPs {}G".format(model.flops() / 10 ** 9))
