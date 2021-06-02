# ---------------------------------------------------------------------------------------------
# Modified from Swin Transformer, please refer to https://github.com/microsoft/Swin-Transformer
# ---------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def partition(x, patch_size):
    """
    Args:
        x: (B, H, W, C)
        patch_size (int): patch size

    Returns:
        patches: (num_patches*B, patch_size, patch_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // patch_size, patch_size, W // patch_size, patch_size, C)
    patches = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, patch_size, patch_size, C)
    return patches


def reverse(patches, patch_size, H, W):
    """
    Args:
        patches: (num_patches*B, patch_size, patch_size, C)
        patch_size (int): Patch size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(patches.shape[0] / (H * W / patch_size / patch_size))
    x = patches.view(B, H // patch_size, W // patch_size, patch_size, patch_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    """ Multilayer perceptron."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbedding(nn.Module):
    """ Image to Patch Embedding

    Args:
        img_size (int): Image size.
        patch_emb_size (int): Patch token size.
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of linear projection output channels. 
        norm_layer (nn.Module, optional): Normalization layer.
    """

    def __init__(self, img_size=224, patch_emb_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_emb_size = to_2tuple(patch_emb_size)
        patches_resolution = [img_size[0] // patch_emb_size[0], img_size[1] // patch_emb_size[1]]
        self.img_size = img_size
        self.patch_emb_size = patch_emb_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_emb_size, stride=patch_emb_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # Ensure the shape is same.
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model input size ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_emb_size[0] * self.patch_emb_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
    

class PatchSliceEmbedding(nn.Module):
    """ Image to Patch Embedding with slice method.

    Args:
        img_size (int): Image size.
        patch_emb_size (int): Patch token size.
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of linear projection output channels.
        norm_layer (nn.Module, optional): Normalization layer.
    """

    def __init__(self, img_size=224, patch_emb_size=4, in_chans=48, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_emb_size)
        patches_resolution = [img_size[0] // patch_emb_size[0], img_size[1] // patch_emb_size[1]]
        self.img_size = img_size
        self.patch_emb_size = patch_emb_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # H, W, C --> H/2, W/2, 2C
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        # H/2, W/2, 2C --> H/2, W/2, 4C
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        B, C, H, W = x.shape
        # Ensure the shape is same.
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model input size ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

    
class PatchProjection(nn.Module):
    """ Patch Projection Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops
    

class Attention(nn.Module):
    """ Basic attention of IPSA and CPSA.

    Args:
        dim (int): Number of input channels.
        patch_size (tuple[int]): Patch size.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value.
        qk_scale (float | None, optional): Default qk scale is head_dim ** -0.5.
        attn_drop (float, optional): Dropout ratio of attention weight.
        proj_drop (float, optional): Dropout ratio of output.
        rpe (bool): Use relative position encoding or not.
    """

    def __init__(self, dim, patch_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., rpe=True):

        super().__init__()
        self.dim = dim
        self.patch_size = patch_size  # Ph, Pw
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.rpe = rpe
        
        if self.rpe:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * patch_size[0] - 1) * (2 * patch_size[1] - 1), num_heads))  # 2*Ph-1 * 2*Pw-1, nH

            # get pair-wise relative position index for each token inside one patch
            coords_h = torch.arange(self.patch_size[0])
            coords_w = torch.arange(self.patch_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Ph, Pw
            coords_flatten = torch.flatten(coords, 1)  # 2, Ph*Pw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Ph*Pw, Ph*Pw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Ph*Pw, Ph*Pw, 2
            relative_coords[:, :, 0] += self.patch_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.patch_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.patch_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Ph*Pw, Ph*Pw
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_patches*B, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.patch_size[0] * self.patch_size[1], self.patch_size[0] * self.patch_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, patch_size={self.patch_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for one patch with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class CATBlock(nn.Module):
    """ Implementation of CAT Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        patch_size (int): Patch size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value.
        qk_scale (float | None, optional): Default qk scale is head_dim ** -0.5.
        drop (float, optional): Dropout rate.
        attn_drop (float, optional): Attention dropout rate. 
        drop_path (float, optional): Stochastic depth rate. 
        act_layer (nn.Module, optional): Activation layer.
        norm_layer (nn.Module, optional): Normalization layer.
        rpe (bool): Use relative position encoding or not.
    """

    def __init__(self, dim, input_resolution, num_heads, patch_size=7, mlp_ratio=4., 
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_type="ipsa", rpe=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.mlp_ratio = mlp_ratio
        self.attn_type = attn_type
        if min(self.input_resolution) <= self.patch_size:
            self.patch_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim if attn_type == "ipsa" else self.patch_size ** 2, patch_size=to_2tuple(self.patch_size), 
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, rpe=rpe)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # partition
        patches = partition(x, self.patch_size)  # nP*B, patch_size, patch_size, C
        patches = patches.view(-1, self.patch_size * self.patch_size, C)  # nP*B, patch_size*patch_size, C

        # IPSA or CPSA
        if self.attn_type == "ipsa":
            attn = self.attn(patches)  # nP*B, patch_size*patch_size, C
        elif self.attn_type == "cpsa":
            patches = patches.view(B, (H // self.patch_size) * (W // self.patch_size), self.patch_size ** 2, C).permute(0, 3, 1, 2).contiguous()
            patches = patches.view(-1, (H // self.patch_size) * (W // self.patch_size), self.patch_size ** 2) # nP*B*C, nP*nP, patch_size*patch_size
            attn = self.attn(patches).view(B, C, (H // self.patch_size) * (W // self.patch_size), self.patch_size ** 2)
            attn = attn.permute(0, 2, 3, 1).contiguous().view(-1, self.patch_size ** 2, C) # nP*B, patch_size*patch_size, C
        else :
            raise NotImplementedError(f"Unkown Attention type: {self.attn_type}") 

        # reverse opration of partition
        attn = attn.view(-1, self.patch_size, self.patch_size, C)
        x = reverse(attn, self.patch_size, H, W)  # B H' W' C

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"patch_size={self.patch_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # 2 norm
        flops += 2 * self.dim * H * W
        # Attention
        N = H / self.patch_size * W / self.patch_size
        if self.attn_type == "ipsa":
            flops += N * self.attn.flops(self.patch_size * self.patch_size)
        elif self.attn_type == "cpsa":
            flops += self.attn.flops(N) * self.dim
        else:
            raise NotImplementedError(f"Unkown Attention type: {self.attn_type}") 
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        return flops


class CATLayer(nn.Module):
    """ Basic CAT layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        patch_size (int): Patch size of IPSA or CPSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value.
        qk_scale (float | None, optional): Default qk scale is head_dim ** -0.5.
        drop (float, optional): Dropout rate.
        ipsa_attn_drop (float): Attention dropout rate of InnerPatchSelfAttention.
        cpsa_attn_drop (float): Attention dropout rate of CrossPatchSelfAttention.
        drop_path (float | tuple[float], optional): Stochastic depth rate.
        norm_layer (nn.Module, optional): Normalization layer.
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer.
        use_checkpoint (bool): Whether to use checkpointing to save memory.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, patch_size, mlp_ratio=4., qkv_bias=True, 
                 qk_scale=None, drop=0., ipsa_attn_drop=0., cpsa_attn_drop=0., drop_path=0., 
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.pre_ipsa_blocks = nn.ModuleList()
        self.cpsa_blocks = nn.ModuleList()
        self.post_ipsa_blocks = nn.ModuleList()
        for i in range(depth):
            self.pre_ipsa_blocks.append(CATBlock(dim=dim, input_resolution=input_resolution,
                                                 num_heads=num_heads, patch_size=patch_size,
                                                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                                                 qk_scale=qk_scale, drop=drop, 
                                                 attn_drop=ipsa_attn_drop, drop_path=drop_path[i],
                                                 norm_layer=norm_layer, attn_type="ipsa", rpe=True))
            
            self.cpsa_blocks.append(CATBlock(dim=dim, input_resolution=input_resolution,
                                             num_heads=1, patch_size=patch_size,
                                             mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                                             qk_scale=qk_scale, drop=drop,
                                             attn_drop=cpsa_attn_drop, drop_path=drop_path[i],
                                             norm_layer=norm_layer, attn_type="cpsa", rpe=False))
            
            self.post_ipsa_blocks.append(CATBlock(dim=dim, input_resolution=input_resolution,
                                                  num_heads=num_heads, patch_size=patch_size,
                                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                                                  qk_scale=qk_scale, drop=drop, 
                                                  attn_drop=ipsa_attn_drop, drop_path=drop_path[i],
                                                  norm_layer=norm_layer, attn_type="ipsa", rpe=True))

        # patch projection layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        num_blocks = len(self.cpsa_blocks)
        for i in range(num_blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(self.pre_ipsa_blocks[i], x)
                x = checkpoint.checkpoint(self.cpsa_blocks[i], x)
                x = checkpoint.checkpoint(self.post_ipsa_blocks[i], x)
            else:
                x = checkpoint.checkpoint(self.pre_ipsa_blocks[i], x)
                x = checkpoint.checkpoint(self.cpsa_blocks[i], x)
                x = checkpoint.checkpoint(self.post_ipsa_blocks[i], x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for i in range(self.depth):
            flops += self.pre_ipsa_blocks[i].flops()
            flops += self.cpsa_blocks[i].flops()
            flops += self.post_ipsa_blocks[i].flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class CAT(nn.Module):
    """ Implementation of "CAT: Cross Attetion in Vision Transformer".

    Args:
        img_size (int | tuple(int)): Input image size.
        patch_emb_size (int | tuple(int)): Patch size in Patch Embedding layer.
        in_chans (int): Number of input image channels.
        num_classes (int): Number of classes for classification head.
        embed_dim (int): Patch embedding dimension.
        depths (tuple(int)): Number of layers in each stage.
        num_heads (tuple(int)): Number of attention heads in different layers.
        patch_size (int): Patch size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
        qk_scale (float): Default qk scale is head_dim ** -0.5.
        drop_rate (float): Dropout rate.
        ipsa_attn_drop (float): Attention dropout rate of InnerPatchSelfAttention.
        cpsa_attn_drop (float): Attention dropout rate of CrossPatchSelfAttention.
        drop_path_rate (float): Stochastic depth rate.
        norm_layer (nn.Module): Normalization layer.
        ape (bool): If True, add absolute position encoding to the patch embedding.
        patch_norm (bool): If True, add normalization after patch embedding.
        slice_emb (bool): If True, use slice method in Patch Embedding layer.
        use_checkpoint (bool): Whether to use checkpointing to save memory.
    """

    def __init__(self, img_size=224, patch_emb_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[1, 1, 3, 1], num_heads=[3, 6, 12, 24],
                 patch_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., ipsa_attn_drop=0., cpsa_attn_drop=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, slice_emb=False,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.use_ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches with conv method or slice method
        if slice_emb:
            self.patch_embed = PatchSliceEmbedding(
                img_size=img_size, patch_emb_size=patch_emb_size, in_chans=in_chans*16, embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None)
        else:
            self.patch_embed = PatchEmbedding(
                img_size=img_size, patch_emb_size=patch_emb_size, in_chans=in_chans, embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position encoding
        if self.use_ape:
            self.ape = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.ape, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = CATLayer(dim=int(embed_dim * 2 ** i_layer),
                             input_resolution=(patches_resolution[0] // (2 ** i_layer), 
                                               patches_resolution[1] // (2 ** i_layer)),
                             depth=depths[i_layer],
                             num_heads=num_heads[i_layer],
                             patch_size=patch_size,
                             mlp_ratio=self.mlp_ratio,
                             qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, 
                             ipsa_attn_drop=ipsa_attn_drop, cpsa_attn_drop=cpsa_attn_drop,
                             drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                             norm_layer=norm_layer,
                             downsample=PatchProjection if (i_layer < self.num_layers - 1) else None,
                             use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

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
        return {'ape'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.use_ape:
            x = x + self.ape
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'CAT':
        model = CAT(img_size=config.DATA.IMG_SIZE,
                    patch_emb_size=config.MODEL.CAT.PATCH_EMB_SIZE,
                    in_chans=config.MODEL.CAT.IN_CHANS,
                    num_classes=config.MODEL.NUM_CLASSES,
                    embed_dim=config.MODEL.CAT.EMBED_DIM,
                    depths=config.MODEL.CAT.DEPTHS,
                    num_heads=config.MODEL.CAT.NUM_HEADS,
                    patch_size=config.MODEL.CAT.PATCH_SIZE,
                    mlp_ratio=config.MODEL.CAT.MLP_RATIO,
                    qkv_bias=config.MODEL.CAT.QKV_BIAS,
                    qk_scale=config.MODEL.CAT.QK_SCALE,
                    drop_rate=config.MODEL.DROP_RATE,
                    ipsa_attn_drop=config.MODEL.IPSA_ATTN_DROP,
                    cpsa_attn_drop=config.MODEL.CPSA_ATTN_DROP,
                    drop_path_rate=config.MODEL.DROP_PATH_RATE,
                    ape=config.MODEL.CAT.APE,
                    patch_norm=config.MODEL.CAT.PATCH_NORM,
                    slice_emb=config.MODEL.CAT.SLICE_EMB,
                    use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
