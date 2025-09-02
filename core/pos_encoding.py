import torch, math

def rope_rotate(head_dim, context_length, device='cpu'):
    """
    x: (B, H, L, Dh) queries or keys (Dh even)
    positions: (L,) absolute positions (0..L-1)
    """

    half = head_dim // 2
    # Generate position indices
    positions = torch.arange(context_length, dtype=torch.float32, device=device)
    freqs = torch.exp(-math.log(1000000) * torch.arange(0, half, device=device) / half)  # (half,)
    angles = torch.einsum('l, h -> lh', positions.float(), freqs)                  # (L, half)

    combined_angles = torch.cat([angles, angles], dim=1)                         # (L, head_dim)

    cos = combined_angles.cos()[None, None, :, :]                                        # (1,1,L,head_dim)

    sin = combined_angles.sin()[None, None, :, :]                                          # (1,1,L,head_dim)

    return cos, sin


def apply_rope(x, cos, sin):
    """
    x: (B, H, L, Dh) queries or keys
    positions: (L,) absolute positions (0..L-1)
    """
    B, H, L, Dh = x.shape
    half = Dh // 2


    x_upper = x[..., :half]
    x_lower = x[..., half:]

    x_bar = torch.cat((-x_lower, x_upper), dim=-1)              # (B,H,L,Dh)

    cos = cos[:,:,:L, :]                                               # (1,1,L,Dh)
    sin = sin[:,:,:L, :]                                                # (1,1,L,Dh)

    x_rot = (x * cos) + (x_bar * sin)
    return x_rot


def test_apply_rope():
    x = torch.randn(2, 8, 512, 128)
    cos, sin = rope_rotate(128, 512)
    x_out = apply_rope(x, cos, sin)
    assert x_out.shape == (2, 8, 512, 128)


# if __name__ == '__main__':
#     # test_apply_rope()
#     pass