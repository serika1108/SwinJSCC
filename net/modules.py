import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
from bisect import bisect
import torch.nn.functional as F
import numpy as np


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 你就不能好好的叫激活层吗
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        # 图像压缩任务感觉没什么必要做dropout
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Swin Transformer的窗口拆分操作
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    # 这里是在拆窗口。在高度上按window_size划分，每个窗口为window_size，在宽度上按window_size划分，每个窗口为window_size
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # 修改维度为(B, H//ws, W//ws, ws, ws, C)，合并得到(B * (H//ws) * (W//ws), ws, ws, C)，第一维是窗口的数量总和
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Swin Transformer窗口拆分的逆操作
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # 如果qk_scale为None，则取head_dim ** -0.5
        self.scale = qk_scale or head_dim ** -0.5

        # 相对位置编码的偏置，根据Swin Transformer的论文，大小为[2*windows_size-1, 2*windows_size-1]
        # 每个注意力分数一个，所以要考虑注意力的头数。但注意对于每个头是一个1维的相对位置列表
        # 列表维度是[(2 * window_size[0] - 1) * (2 * window_size[1] - 1)]，而不是[(2 * window_size[0] - 1), (2 * window_size[1] - 1)]
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # 这里是在计算窗口内patch的相对位置，以索引的形式呈现。窗口共包含Wh*Ww个patch元素，计算自注意力就需要计算Wh*Ww个元组之间的相符位置
        # 最后得到的相对位置矩阵是(Wh*Ww) * (Wh*Ww)维的，每个元素是一个编码好的索引，对应于上一步相对位置表里面的1维位置编码

        # 根据注意力窗口的大小Wh和Ww，生成两个列表
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        # meshgrid会根据给定的两个列表，生成两个矩阵，它们代表了两个列表所有的笛卡尔乘积结果
        # 注意后面的操作对象是窗口内patch的数量，而不限于窗口的高和宽。窗口一共有Wh*Ww个元素，meshgrid将的得到两个[Wh, Ww]维的矩阵
        # 第一个矩阵记录了窗口内所有Wh*Ww个元素的纵坐标(Wh)，第二个矩阵则记录了所有元素的横坐标。取两个矩阵对应位置的值，例如(i,j)位置
        # 组合得到的将是窗口内(i,j)位置的横纵坐标(y_i. x_i)。stack是叠起来，最后得到维度是[2, Wh, Ww]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        # 展开列表，将窗口内Wh*Ww个元素所有的y，x坐标展开成1维列表
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # 这一步是在计算相对位置，用None扩展维度后相减，得到两个[Wh*Ww, Wh*Ww]的矩阵
        # 第一个矩阵记录的是所有元素y坐标的相对位置差，第二个矩阵记录的是所有元素x坐标的相对位置差
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        # 修改维度，把2放在后面，此时矩阵是[Wh*Ww, Wh*Ww]的，每个元素都是相对位置的差[y1-y2, x1-x2]
        # permute只更改视角，不改变内存布局，只会告诉程序以这种方式读取数据，但数据在内存里存储的位置不会改变
        # 这会导致tensor是非连续的，contiguous改变数据的存储位置，以连续的形式存储，将提高读取性能
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        # 相对位置的取值是[-Wh+1, Wh-1]和[-Ww+1, Ww-1]，这两步是为了移动数值，转变为从0开始的正数
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        # 这里是在降维，将[Wh*Ww, Wh*Ww, 2]转化为[Wh*Ww, Wh*Ww]，就是生成一个每个位置对应的相对位置的索引
        # relative_position_index的所以将对应于relative_position_bias_table里面的位置
        # 转化维度的方式比较直接，采用index = dy_shift * Wx + dx_shift的形式，和二维数组转换一维数组的方式一样
        # 这个列表中将有大量重复的值，因为有(Wh*Ww) * (Wh*Ww)个相对位置，但只有(2*Wh-1) * (2*Ww-1)个相对位置取值
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # 把这个索引存入buffer，不参与训练，但会随着模型移动
        self.register_buffer("relative_position_index", relative_position_index)

        # 映射QKV矩阵
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 用方差0.02的截断正态分布随机初始化相对位置编码参数relative_position_bias_table
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, add_token=True, token_num=0, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        Swin Transformer为了和普通的transformer保持一致，没有为num_windows单独开一个维度
        因此第一维是num_windows*B，和普通transformer的(B, N, C)保持一致
        每个Batch里面包含的图像是一样的，尺寸也是一样的，所以mask规则也是一样的，mask的第一维没有batch
        """
        # N是窗口内patch数量，也就是token数量
        B_, N, C = x.shape

        # 计算QKV并取出
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # (B_, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        # 调换k的最后两个维度，因为前面还有个B和head，所以不能直接写转置，用调换维度来转置
        attn = (q @ k.transpose(-2, -1))

        # view(-1)将二维的索引展平成一维[(Wh*Ww) * (Wh*Ww)]的列表，然后从相对位置参数中取值，也就是相对位置编码
        # 注意这里的索引，python支持维度不同的索引，会默认在第一维取索引值。relative_position_bias_table是二维的[..., Head]
        # 展平的结果1维的，那么将会默认从relative_position_bias_table的第一维取，得到的结果是[(Wh*Ww) * (Wh*Ww), Head]
        # 取值之后再恢复原始的维度，(Wh*Ww, Wh*Ww, Head)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww, Wh*Ww, Head
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # Head, Wh*Ww, Wh*Ww

        # 这个分支是在判断是否有预留的token，例如CLS token，预留的token_num个token不需要添加位置向量
        if add_token:
            # unsqueeze是为了匹配batch的维度
            attn[:, :, token_num:, token_num:] = attn[:, :, token_num:, token_num:] + relative_position_bias.unsqueeze(0)
        else:
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            if add_token:
                # 如果有额外的token，要对mask做填充，pad的参数是(pad_left, pad_right, pad_top, pad_bottom)
                mask = F.pad(mask, (token_num, 0, token_num, 0), "constant", 0)
            # 每个batch的window数量
            nW = mask.shape[0]
            # 把注意力分数转换成(B, num_windows, num_heads, N, N)
            # unsqueeze(1): (nW, 1, N, N)，unsqueeze(0): (1, nW, 1, N, N)，然后做广播
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            # 最后恢复原来的维度来做softmax
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # (B_, num_heads, N, N) @ (B_, num_heads, N, head_dim) = (B_, num_heads, N, head_dim)
        # 把num_heads和N换位，通过reshape合并head，C=num_heads*head_dim
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # 估算一个窗口的注意力计算需要多少flops
        flops = 0
        # qkv = self.qkv(x)，维度从dim扩展到3*dim，计算量为N * dim * (3*dim)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))，(N, head_dim) @ (head_dim, N) -> (N, N)
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)，(N, N) @ (N, head_dim) -> (N, head_dim)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)，投影层维持dim维度不变
        flops += N * self.dim * self.dim
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        if out_dim is None:
            out_dim = dim
        self.dim = dim
        # patch merge操作
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        # print(x.shape)
        # print(self.input_resolution)
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.view(B, H, W, C)
        # 切片语法为[start:stop:step]，因此下面的操作均是从不同的起点，每隔2个取一个patch
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        # patch组合
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H*W//4, 4 * C)  # B H/2*W/2 4*C
        x = self.norm(x)
        # merge
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchMerging4x(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, use_conv=False):
        super().__init__()
        H, W = input_resolution
        self.patch_merging1 = PatchMerging((H, W), dim, norm_layer=norm_layer)
        self.patch_merging2 = PatchMerging((H // 2, W // 2), dim, norm_layer=norm_layer)

    def forward(self, x, H=None, W=None):
        if H is None:
            H, W = self.input_resolution
        x = self.patch_merging1(x, H, W)
        x = self.patch_merging2(x, H//2, W//2)
        return x


class PatchReverseMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

    """

    def __init__(self, input_resolution, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        # 逆merge，其实就是扩展维度后重新拼接，让特征图的分辨率加倍
        self.increment = nn.Linear(dim, out_dim * 4, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = self.norm(x)
        x = self.increment(x)
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
        # nn.PixelShuffle(upscale_factor=r)，用于扩展分辨率，和列表提取+拼接是等价的
        # 输入: (B, C_in, H, W)，输出: (B, C_out, H*r, W*r)，要求C_in = C_out * (r^2)
        x = nn.PixelShuffle(2)(x)
        x = x.flatten(2).permute(0, 2, 1)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * 2 * W * 2 * self.dim // 4
        flops += (H * 2) * (W * 2) * self.dim // 4 * self.dim
        return flops


class PatchReverseMerging4x(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, use_conv=False):
        super().__init__()
        self.use_conv = use_conv
        self.input_resolution = input_resolution
        self.dim = dim
        H, W = input_resolution
        self.patch_reverse_merging1 = PatchReverseMerging((H, W), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)
        self.patch_reverse_merging2 = PatchReverseMerging((H * 2, W * 2), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)

    def forward(self, x, H=None, W=None):
        if H is None:
            H, W = self.input_resolution
        x = self.patch_reverse_merging1(x, H, W)
        x = self.patch_reverse_merging2(x, H*2, W*2)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * 2 * W * 2 * self.dim // 4
        flops += (H * 2) * (W * 2) * self.dim // 4 * self.dim
        return flops


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # 卷积的patch embedding写法，和ViT保持一致
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # flatten代表从第2维开始，把后面的展平成1维(B, embed_dim, H, W) -> (B, embed_dim, N)，将二维的patch展平成一个token列表
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
