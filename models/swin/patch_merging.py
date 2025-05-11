# models/swin/blocks.py (continued)
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        
        # Get relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        # Query, key, value projection
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        
        # Attention dropout
        self.attn_drop = nn.Dropout(attn_dropout)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        
        # Initialize relative position bias
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def forward(self, x, mask=None):
        """Forward pass"""
        B_, N, C = x.shape
        
        # Project query, key, value
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Calculate attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # Add attention mask if needed
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        
        # Apply attention dropout
        attn = self.attn_drop(attn)
        
        # Apply attention
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        
        # Output projection
        x = self.proj(x)
        
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block"""
    
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        dropout=0.0,
        attn_dropout=0.0,
        drop_path=0.0
    ):
        """
        Args:
            dim: Input dimension
            input_resolution: Input resolution
            num_heads: Number of attention heads
            window_size: Window size
            shift_size: Shift size
            mlp_ratio: MLP hidden dim ratio
            dropout: Dropout rate
            attn_dropout: Attention dropout rate
            drop_path: Drop path rate
        """
        super().__init__()
        
        # Save params
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        # Check if window size is larger than input resolution
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        
        # Window attention
        self.attn = WindowAttention(
            dim=dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            attn_dropout=attn_dropout
        )
        
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # Layer normalization
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            dropout=dropout
        )
        
        # Calculate attention mask for SW-MSA
        if self.shift_size > 0:
            # Calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
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
                    
            # Generate window mask
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
            
        self.register_buffer("attn_mask", attn_mask)
    
    def forward(self, x):
        """Forward pass"""
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"
        
        # Shortcut
        shortcut = x
        
        # Layer normalization
        x = self.norm1(x)
        
        # Reshape to (B, H, W, C)
        x = x.view(B, H, W, C)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            
        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # Window attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            
        # Reshape to (B, L, C)
        x = x.view(B, H * W, C)
        
        # Apply dropout
        x = shortcut + self.drop_path(x)
        
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class MLP(nn.Module):
    """MLP module"""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.0):
        """
        Args:
            in_features: Input features
            hidden_features: Hidden features
            out_features: Output features
            dropout: Dropout rate
        """
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        """Forward pass"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x


class BasicLayer(nn.Module):
    """Basic Swin Transformer Layer"""
    
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        dropout=0.0,
        attn_dropout=0.0,
        drop_path=0.0,
        downsample=None
    ):
        """
        Args:
            dim: Input dimension
            input_resolution: Input resolution
            depth: Number of blocks
            num_heads: Number of attention heads
            window_size: Window size
            mlp_ratio: MLP hidden dim ratio
            dropout: Dropout rate
            attn_dropout: Attention dropout rate
            drop_path: Drop path rate
            downsample: Downsample layer
        """
        super().__init__()
        
        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path
            )
            for i in range(depth)
        ])
        
        # Downsample layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution=input_resolution,
                dim=dim,
                norm_layer=nn.LayerNorm
            )
        else:
            self.downsample = None
    
    def forward(self, x):
        """Forward pass"""
        # Apply Swin blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply downsample if needed
        if self.downsample is not None:
            x = self.downsample(x)
            
        return x