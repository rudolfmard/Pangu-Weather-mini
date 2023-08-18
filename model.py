import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from timm.models.layers import DropPath, trunc_normal_

def LoadConstantMask(batch_size):   # TODO: Replace dummy masks with actual data
    # Mask shape must equal (B, 1, H=1440, W=721) + padding
    land_mask = torch.ones((batch_size, 1, 1440, 721))
    soil_type = torch.ones((batch_size, 1, 1440, 721))
    topography = torch.ones((batch_size, 1, 1440, 721))
    # Make sure the padding is the same as performed on "input_surface" in PatchEmbedding
    return F.pad(land_mask, (2,1)), F.pad(soil_type, (2,1)), F.pad(topography, (2,1))

class WeatherModel(nn.Module):
    def __init__(self, C, depth, n_heads, D, batch_size, log_GPU_mem=False):
        super().__init__()
        self.log_GPU_mem = log_GPU_mem
        # Drop path rate is linearly increased as the depth increases:
        drop_path_list = torch.linspace(0, 0.2, 8)

        # Patch embedding:
        self.input_layer = PatchEmbedding((2,4,4), C, batch_size)

        # Four main layers:
        self.layer1 = EarthSpecificLayer(depth=depth[0], dim=C, input_resolution=(8, 360, 181),
                                        heads=n_heads[0], drop_path_ratio_list=drop_path_list[:2], D=D)
        #self.middleLayers = nn.Sequential(
        self.layer2 = EarthSpecificLayer(depth=depth[1], dim=2*C, input_resolution=(8, 180, 91), heads=n_heads[1], drop_path_ratio_list=drop_path_list[2:], D=D)
        self.layer3 = EarthSpecificLayer(depth=depth[2], dim=2*C, input_resolution=(8, 180, 91), heads=n_heads[2], drop_path_ratio_list=drop_path_list[2:], D=D)
        #)
        self.layer4 = EarthSpecificLayer(depth=depth[3], dim=C, input_resolution=(8, 360, 181),
                                        heads=n_heads[3], drop_path_ratio_list=drop_path_list[:2], D=D)

        # Upsample and downsample:
        self.downsample = DownSample(C)
        self.upsample = UpSample(2*C, C)

        # Patch recovery:
        self.output_layer = PatchRecovery((2,4,4), 2*C)

    def forward(self, data):
        print(f'    Start GPU: {torch.cuda.memory_allocated(0) / 1e9} GB | Peak GPU: {torch.cuda.max_memory_allocated(0) / 1e9} GB') if self.log_GPU_mem else None
        # Patch embedding of the input fields:
        #   air: (B, Z=13, H=1440, W=721, C=5) & surface: (B, H=1440, W=721, C=4) 
        #   ->   (B, Z=8, H=360, W=181, C)
        x = self.input_layer(data[0], data[1])
        
        ###     Encoder     ###
        x = checkpoint(self.layer1, x)
        skip = x
        # Downsample the spatial resolution:    (B, 8, 360, 181, C) -> (B, 8, 180, 91, 2C)
        x = self.downsample(x)
        x = checkpoint(self.layer2, x)

        #x = checkpoint(self.middleLayers, x)

        ###     Decoder     ###
        x = checkpoint(self.layer3, x)
        # Restore the spatial resolution:       (B, 8, 180, 91, 2C) -> (B, 8, 360, 181, C)
        x = self.upsample(x)
        x = checkpoint(self.layer4, x)

        # Concatenate skip connection along last dimension:
        #   (B, 8, 360, 181, C) -> (B, 8, 360, 181, 2C)
        x = torch.cat((skip, x), -1)

        # Patch recovery extracts air & surface variable predictions:
        #   (B, 8, 360, 181, 2C) -> (B, 13, 1440, 721, 5) & (B, 1440, 721, 4)
        output_air, output_surface = self.output_layer(x)
        print(f'    End GPU: {torch.cuda.memory_allocated(0) / 1e9} GB | Peak GPU: {torch.cuda.max_memory_allocated(0) / 1e9} GB') if self.log_GPU_mem else None
        return (output_air, output_surface)

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, dim, batch_size): # patch_size = (2,4,4)
        super().__init__()
        self.conv_air = nn.Conv3d(in_channels=5, out_channels=dim, kernel_size=patch_size, stride=patch_size)
        self.conv_surface = nn.Conv2d(in_channels=7, out_channels=dim, kernel_size=patch_size[1:], stride=patch_size[1:])

        # Load constant masks from the disc
        land_mask, soil_type, topography = LoadConstantMask(batch_size)
        self.register_buffer('land_mask', land_mask, persistent=False)
        self.register_buffer('soil_type', soil_type, persistent=False)
        self.register_buffer('topography', topography, persistent=False)

    def forward(self, input_air, input_surface):
        # input shapes, as in the paper (+ batch_size B): 
        #       upper-air variables:    (B, Z=13, H=1440, W=721, C=5)
        #       surface variables:      (B, H=1440, W=721, C=4)

        # torch conv layers take inputs of shape (B, C (, Z), H, W), therefore permute:
        input_air = input_air.permute(0, 4, 1, 2, 3)
        input_surface = input_surface.permute(0, 3, 1, 2)
        
        # Add padding to the data
        #   (B, 5, 13, 1440, 721) -> (B, 5, 14, 1440, 724)
        input_air = F.pad(input_air, (2,1,0,0,1,0))
        #   (B, 4, 1440, 721) -> (B, 4, 1440, 724)
        input_surface = F.pad(input_surface, (2,1))

        # Apply a linear projection for patches of size patch_size[0]*patch_size[1]*patch_size[2]
        #   (B, 5, 14, 1440, 724) -> (B, C, 7, 360, 181)
        input_air = self.conv_air(input_air)

        # Add three constant fields to the surface fields
        #   (B, 4, 1440, 724) -> (B, 7, 1440, 724)
        input_surface = torch.cat((input_surface, self.land_mask, self.soil_type, self.topography), 1)

        # Apply a linear projection for patches of size patch_size[1]*patch_size[2]
        #   (B, 7, 1440, 721) -> (B, C, 360, 181)
        input_surface = self.conv_surface(input_surface)

        # Concat the air and surface data in Z dimension -> (B, C, Z=8, H=360, W=181)
        x = torch.cat((input_air, torch.unsqueeze(input_surface, 2)), 2)

        # torch.premute back to shape familiar from the paper: 
        #   (B, C, Z, H, W) -> (B, Z, H, W, C)
        x = torch.permute(x, (0,2,3,4,1))

        return x

class PatchRecovery(nn.Module):
    def __init__(self, patch_size, dim):
        super().__init__()
        self.tconv_air = nn.ConvTranspose3d(dim, 5, patch_size, patch_size)
        self.tconv_surface = nn.ConvTranspose2d(dim, 4, patch_size[1:], patch_size[1:])

    def forward(self, x):
        # input shape: (B, 8, 360, 181, 2C)

        # torch conv layers take inputs of shape (B, C (, Z), H, W), therefore permute:
        #   (B, 8, 360, 181, 2C) -> (B, 2C, 8, 360, 181)
        x = torch.permute(x, (0,4,1,2,3))

        # Recover the air variables from [1:] slice of Z dimension:
        #   (B, 2C, 7, 360, 181) -> (B, 5, 14, 1440, 724)
        output_air = self.tconv_air(x[:, :, 1:, :, :])

        # Recover the surface variables from [0] slice of the Z dimension:
        #   (B, 2C, 360, 181) -> (B, 4, 1440, 724)
        output_surface = self.tconv_surface(x[:, :, 0, :, :])

        # Crop the padding added in patch embedding:
        #   (B, 5, 14, 1440, 724) -> (B, 5, 13, 1440, 721)
        output_air = output_air[:, :, 1:, :, 2:-1]
        #   (B, 4, 1440, 724) -> (B, 4, 1440, 721)
        output_surface = output_surface[:, :, :, 2:-1]

        # Restore the original shape:
        #   (B, 5, 13, 1440, 721) -> (B, 13, 1440, 721, 5)
        output_air = output_air.permute(0, 2, 3, 4, 1)
        #   (B, 4, 1440, 721) -> (B, 1440, 721, 4)
        output_surface = output_surface.permute(0, 2, 3, 1)
        return output_air, output_surface

class DownSample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4*dim)
        self.linear = nn.Linear(4*dim, 2*dim, bias=False)

    def forward(self, x):
        """
        input shape:    (B, 8, 360, 181, C)
        output shape:   (B, 8, 180, 91, 2C)
        """
        B, Z, H, W, C = x.shape

        # Add padding to the input:     (B, 8, 360, 181, C) -> (B, 8, 360, 182, C)
        x = F.pad(x, (0,0,0,1))
        W = W+1

        # Merge four tokens into one to halve H and W dimensions:
        # (B, 8, 360, 182, C) -> (B, 8, 180, 91, 4C)
        x = x.view(B, Z, H//2, 2, W//2, 2, C).permute(0,1,2,4,3,5,6)
        x = x.reshape(B, Z, H//2, W//2, 4*C)
        
        # Normalize and halve the number of channels:
        # (B, 8, 180, 91, 4C) -> (B, 8, 180, 91, 2C)
        x = self.norm(x)
        x = self.linear(x)
        return x

class UpSample(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim*4, bias=False)
        self.linear2 = nn.Linear(output_dim, output_dim, bias=False)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        """
        input shape:    (B, 8, 180, 91, 2C)
        output shape:   (B, 8, 360, 181, C)
        """
        B, Z, H, W, C_in = x.shape
        C_out = C_in//2

        # Increase the number of channels:  (B, 8, 180, 91, 2C) -> (B, 8, 180, 91, 4C)
        x = self.linear1(x)

        # Reshape to recover the original spatial resolution:
        #   (B, 8, 180, 91, 4C) -> (B, 8, 360, 182, C)
        x = x.view(B, Z, H, W, 2, 2, C_out).permute(0,1,2,4,3,5,6)
        x = x.reshape(B, Z, H*2, W*2, C_out)

        # Crop the padding added in DownSample: (B, 8, 360, 182, C) -> (B, 8, 360, 181, C)
        x = x[:, :, :, :-1, :]

        # Apply LayerNorm and a Linear layer:
        x = self.norm(x)
        x = self.linear2(x)
        return x

class EarthSpecificLayer(nn.Module):
    def __init__(self, depth, dim, input_resolution, heads, drop_path_ratio_list, D):
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList(
            [EarthSpecificBlock(dim, input_resolution, heads, drop_path_ratio_list[i], D) for i in range(depth)]
        )

    def forward(self, x):
        for i in range(self.depth):
            # Shifted window attention for every other block:
            if i % 2 == 0:
                x = self.blocks[i](x, roll=False)
            else:
                x = self.blocks[i](x, roll=True)
        return x

class EarthSpecificBlock(nn.Module):
    def __init__(self, dim, input_resolution, heads, drop_path_ratio, D):
        super().__init__()
        self.window_size = (2, 12, 6)
        self.input_resolution = input_resolution

        self.attention = EarthAttention3D(dim, heads, self.window_size, input_resolution, 0)
        self.norm1 = nn.LayerNorm(dim)      # Normalize over last dimension (channels), expects last dimension to be size dim
        self.feedforward = MLP(dim, D, 0)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path_ratio)

        self.register_buffer('mask', self.generate_mask(), persistent=False)    # shape: (1, nW, 1, T, T)
        self.register_buffer('zero_mask', torch.zeros(self.mask.shape), persistent=False)

    def generate_mask(self):
        # The data is shifted by half of the window size:
        shift_size = (self.window_size[0]//2, self.window_size[1]//2, self.window_size[2]//2)
        Z, H, W = self.input_resolution

        # Add +5 zero padding as done to the data:
        W = W + 5

        # Initialize image mask as zero tensor with shape of the original data cube, omitting batch and channel dimensions:
        img_mask = torch.zeros((Z, H, W))

        # Give values to mask elements indicating whether the elements are
        # warped from another window after cyclic shifting:
        z_slices = [slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -shift_size[0]),
                    slice(-shift_size[0], None)]
        h_slices = [slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -shift_size[1]),
                    slice(-shift_size[1], None)]
        w_slices = [slice(0, -self.window_size[2]),
                    slice(-self.window_size[2], -shift_size[2]),
                    slice(-shift_size[2], None)]
        window_number = 0
        for z in z_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[z, h, w] = window_number
                    window_number += 1
        
        # Partition of image mask into windows as done to the data:
        # (Z, H, W) -> (nW, T)     where T = window_size[0]*window_size[1]*window_size[2]
        nW = (Z//self.window_size[0])*(H//self.window_size[1])*(W//self.window_size[2])
        T = self.window_size[0]*self.window_size[1]*self.window_size[2]

        mask_windows = img_mask.view(Z//self.window_size[0], self.window_size[0], H//self.window_size[1], self.window_size[1], W//self.window_size[2], self.window_size[2])
        mask_windows = mask_windows.permute(0,2,4,1,3,5).contiguous().view(nW, T)

        # Calculate pairwise "distances" of mask elements within windows:
        #   (nW, 1, T) - (nW, T, 1) = (nW, T, T)
        attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)

        # Mask the attention between elements with distance != 0:
        attention_mask = attention_mask.masked_fill(attention_mask != 0, -1000).float()

        # Add dummy dimensions for batch & attention head dimensions:
        # (nW, T, T) -> (1, nW, 1, T, T)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(0)

        return attention_mask

    def forward(self, x, roll):
        # x is of shape: (B, Z=8, H=360, W=181, C)
        #            or  (B, Z=8, H=180, W=91, 2C)
        # Sanity check for input shape:
        assert (x.shape[1], x.shape[2], x.shape[3]) == self.input_resolution, "Unexpected input resolution"

        skip_connection = x
        
        # Add +5 zero padding to make W dimension (181 or 91) divisible by window size (6):
        x = F.pad(x, (0,0,3,2))

        B, Z, H, W, C = x.shape

        # Shifting windows:
        if roll:
            # Roll x for half of the window for 3 dimensions (Z, H, W):
            x = torch.roll(x, (self.window_size[0]//2, self.window_size[1]//2, self.window_size[2]//2), (1,2,3))

        # Partition of patches/tokens into nW number of windows of volume T = window_size[0]*window_size[1]*window_size[2]:
        #   (B, Z, H, W, C) -> (nW*B, T, C)
        # The number of windows in Z, H and W dimensions:
        nWz, nWh, nWw = Z//self.window_size[0], H//self.window_size[1], W//self.window_size[2]
        nW = nWz*nWh*nWw
        T = self.window_size[0]*self.window_size[1]*self.window_size[2]

        x = x.view(B, nWz, self.window_size[0], nWh, self.window_size[1], nWw, self.window_size[2], C)
        x = x.permute(0,1,3,5,2,4,6,7).contiguous().view(B*nW, T, C)

        # Calculate attention for each nW(*B) window:
        if roll:
            # If two pixels are not adjacent, then mask the attention between them
            x = self.attention(x, self.mask)
        else:
            # no mask, use zero matrix
            x = self.attention(x, self.zero_mask)

        # Restore the original shape of the data:
        #   (nW*B, T, C) -> (B, Z, H, W, C)
        x = x.view(B, nWz, nWh, nWw, self.window_size[0], self.window_size[1], self.window_size[2], C)
        x = x.permute(0,1,4,2,5,3,6,7).contiguous().view(B, Z, H, W, C)

        if roll:
            # Roll x back for half of the window:
            x = torch.roll(x, (-self.window_size[0]//2, -self.window_size[1]//2, -self.window_size[2]//2), (1,2,3))

        # Crop the zero padding
        x = x[:, :, :, 3:-2, :]
        # Apply the final feed-forward of the transformer block along skip connections: 
        x = skip_connection + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.feedforward(x)))
        return x

class EarthAttention3D(nn.Module):
    def __init__(self, dim, heads, window_size, input_resolution, dropout_rate):
        super().__init__()
        
        self.linear_qkv = nn.Linear(dim, dim*3, bias=True)
        self.linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.n_heads = heads
        self.head_size = dim//heads
        self.dim = dim
        self.scale = (dim//heads)**-0.5
        self.window_size = window_size                  # shape: (2, 12, 6)
        self.input_resolution = (input_resolution[0], input_resolution[1], input_resolution[2]+5)        # shape: (8, 360, 181) or (8, 180, 91) (+5 padding to W dim)

        ### Earth-specific positional bias ###

        # The bias is composed of submatrices, with the number of submatrices being equal to the number of 
        # windows in the pressure level (Z) and latitude (W) dimensions (longitudes (H) share the bias parameters):
        self.n_submatrices = (self.input_resolution[0]//window_size[0])*((self.input_resolution[2])//window_size[2])

        # Each submatrix has P learnable parameters:
        #       Wz^2 absolute positions along Z axis,
        #       Ww^2 absolute positions along W axis,
        #       2*Wh-1 relative positions along H axis,
        # -> P = (2*Wh-1)*(Wz^2)*(Ww^2) parameters in total, one for each position
        # Construct the parameter tensor of shape (P, n_submatrices, n_heads):
        self.earth_specific_bias = nn.Parameter(torch.zeros(
            (window_size[0]**2)*(2*window_size[1]-1)*(window_size[2]**2), self.n_submatrices, heads
        ))

        # Initialize the parameters using Truncated Normal distribution:
        trunc_normal_(self.earth_specific_bias, std=0.02)

        # Construct position index to reuse self.earth_specific_bias
        self.position_index = self._construct_index()

    def _construct_index(self):
        # Create a list of token pair coordinates in the attention matrix:
        T = self.window_size[0]*self.window_size[1]*self.window_size[2]
        aux1, aux2 = torch.meshgrid(torch.arange(T), torch.arange(T))
        attention_coordinates = torch.stack((aux1.flatten(), aux2.flatten()), dim=1)

        position_index = torch.zeros(T**2)
        for idx, coord in enumerate(attention_coordinates):
            position_index[idx] = self._get_bias_index(coord)
        return position_index.int()

    def _get_bias_index(self, coord):
        i, j = int(coord[0]), int(coord[1])

        # Recover 3D intra-window coordinates
        Wz, Wh, Ww = self.window_size[0], self.window_size[1], self.window_size[2]  # (2, 12, 6)
        z1 = i // (Wh * Ww)
        h1 = (i % (Wh * Ww)) // Ww
        w1 = i % Ww

        z2 = j // (Wh * Ww)
        h2 = (j % (Wh * Ww)) // Ww
        w2 = j % Ww
        
        # Calculate the 3D coordinate of positional bias:
        Bz = z1 + z2*Wz
        Bh = h1 - h2 + Wh -1
        Bw = w1 + w2*Ww

        # Transform to flattened index of the positional bias:
        D1, D2, D3 = (self.window_size[0]**2), (2*self.window_size[1]-1), (self.window_size[2]**2)
        bias_idx = Bz*D2*D3 + Bh*D3 + Bw
        return bias_idx

    def forward(self, x, mask):
        # x is of shape: (B_=nW*B, T, C)
        B_, T, C = x.shape
        
        # Linear layer to create query, key and value:
        #   (nW*B, T, C) -> (nW*B, T, 3*C)
        qkv = self.linear_qkv(x)

        # reshape the data to extract the key, query and value tensors, each of shape:
        #   (nW*B, n_heads, T, head_size)
        qkv = qkv.view(B_, T, 3, self.n_heads, self.head_size).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        del qkv

        # Calculate attention weights and scale:
        #   (nW*B, n_heads, T, head_size) @ (nW*B, n_heads, head_size, T) -> (nW*B, n_heads, T, T)
        attention = (query@key.transpose(-2,-1))*self.scale
        del query
        del key

        # Fetch positional bias values for each token pair in attention matrix:
        positional_bias = self.earth_specific_bias[self.position_index]     # shape: (T**2, n_submatrices, n_heads)
        assert positional_bias.shape[0] == T**2, f"The number of tokens ({T}) squared doesnt match positional bias of shape {positional_bias.shape[0]}"

        # Number of windows in Z, H and W dimensions:
        nWz, nWh, nWw = self.input_resolution[0]//self.window_size[0], self.input_resolution[1]//self.window_size[1], self.input_resolution[2]//self.window_size[2]
        nW = nWz*nWh*nWw

        # Separate the windows in Z and W dimensions:
        #   (T**2, n_submatrices, n_heads) -> (T**2, nWz, nWw, n_heads)
        positional_bias = positional_bias.view(T**2, nWz, nWw, self.n_heads)
        # Bias parameters are shared in H dimension; repeat the tensor in H dimension to obtain nW windows in total:
        #   (T**2, nWz, nWw, n_heads) -> (T**2, nWz, nWh, nWw, n_heads)
        positional_bias = positional_bias.unsqueeze(2)
        positional_bias = positional_bias.repeat(1,1,nWh,1,1)
        # Merge window dimensions:
        #   (T**2, nWz, nWh, nWw, n_heads) -> (T**2, nW, n_heads)
        positional_bias = positional_bias.view(T**2, nW, self.n_heads)
        # Finally reshape to match the shape of attention weights:
        #   (T**2, nW, n_heads) -> (1, nW, n_heads, T, T)
        positional_bias = positional_bias.permute(1,2,0).view(nW, self.n_heads, T, T).unsqueeze(0)

        # Separate the batch dimension from the number of windows in attention weights before adding bias and mask:
        #   (nW*B, n_heads, T, T) -> (B, nW, n_heads, T, T)
        attention = attention.view(B_//nW, nW, self.n_heads, T, T)

        # Add the Earth-specific bias to the attention matrix
        attention = attention + positional_bias + mask                       # shape: (B, nW, n_heads, T, T) + (1, nW, n_heads, T, T)

        # Apply attention mask:
        assert nW == mask.shape[1], f"nW does not match with mask shape {mask.shape}"
        #attention = attention + mask                                    # shape: (B, nW, n_heads, T, T) + (1, nW, 1, T, T)
        del mask
        attention = attention.view(B_, self.n_heads, T, T)              # shape: (B, nW, n_heads, T, T) -> (nW*B, n_heads, T, T)

        # Apply softmax + dropout:
        attention = self.dropout(F.softmax(attention, dim=-1))
        #attention = self.dropout(attention)

        # Multiply the value tensor by attention weights:
        #    (nW*B, n_heads, T, T) @ (nW*B, n_heads, T, head_size) -> (nW*B, n_heads, T, head_size)
        x = attention@value
        del attention
        del value

        # Concatenate attention heads by reshaping x to its original shape:
        #    (nW*B, n_heads, T, head_size) -> (nW*B, T, C)
        x = x.permute(0, 2, 1, 3).reshape(B_, T, C)

        # Final linear layer + dropout, perserves the shape of x
        x = self.linear(x)
        x = self.dropout(x)
        return x

class MLP(nn.Module):
    def __init__(self, dim, D, dropout_rate):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim*D)
        self.linear2 = nn.Linear(dim*D, dim)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x