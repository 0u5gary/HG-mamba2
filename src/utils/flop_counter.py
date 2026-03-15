import torch
import torch.nn as nn
from thop import profile, clever_format

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None
    print("⚠️ Warning: 'mamba_ssm' not found. Mamba FLOPs cannot be calculated.")

def count_multihead_attention(m, x, y):
    """
    Customize the computation rules of nn.MultiheadAttention for use by thop
    """

    attn_output, attn_weights = y[0], y[1]
    D = m.embed_dim
    
    # Derive the Batch Size (B) and Query Sequence Length (T_q)
    if m.batch_first:
        B, T_q, _ = attn_output.size()
    else:
        T_q, B, _ = attn_output.size()
        
    # 3. Derive the key/value Sequence Length (T_k)
    T_k = attn_weights.size(-1)
        
    # calculate FLOPs based on standard multi-head attention operations
    q_proj = B * T_q * D * D
    k_proj = B * T_k * D * D
    v_proj = B * T_k * D * D
    
    attn_scores = B * T_q * T_k * D
    attn_values = B * T_q * T_k * D
    
    out_proj = B * T_q * D * D
    
    total_macs = q_proj + k_proj + v_proj + attn_scores + attn_values + out_proj
    
    # add to module's total_ops
    m.total_ops += torch.DoubleTensor([int(total_macs)])

def count_skip_ops(m, x, y):
    """Skip Mamba and let our own function calculate its FLOPs."""
    m.total_ops += torch.DoubleTensor([0])

def calculate_mamba_flops_manual(model, batch_size, seq_len):
    if Mamba is None: return 0.0
    total_flops = 0.0
    for m in model.modules():
        if isinstance(m, Mamba):
            d_model = m.d_model
            d_inner = m.d_inner 
            d_conv = m.d_conv
            dt_rank = m.dt_rank
            d_state = m.d_state
            
            proj_flops = batch_size * seq_len * d_model * (2 * d_inner)
            conv_flops = batch_size * d_inner * seq_len * d_conv
            ssm_flops = batch_size * seq_len * d_inner * (dt_rank + d_state * 2)
            out_flops = batch_size * seq_len * d_inner * d_model
            dt_flops = batch_size * seq_len * dt_rank * d_inner
            x_flops = batch_size * seq_len * d_inner * (dt_rank + 2 * d_state)

            block_flops = proj_flops + conv_flops + ssm_flops + out_flops + dt_flops + x_flops
            total_flops += block_flops
    return total_flops

def count_model_flops(model, embedding_size=512, seq_len=500, device='cuda'):
    dummy_audio = torch.randn(1, seq_len, 257).to(device) 
    video_len = seq_len // 4 
    dummy_video = torch.randn(1, video_len, embedding_size).to(device)
    
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    try:
        custom_ops = {
            nn.MultiheadAttention: count_multihead_attention
        }
        
        if Mamba is not None:
            custom_ops[Mamba] = count_skip_ops

        base_flops, _ = profile(
            model, 
            inputs=(dummy_audio, dummy_video), 
            custom_ops=custom_ops,
            verbose=False
        )
        
        mamba_flops = calculate_mamba_flops_manual(model, batch_size=1, seq_len=seq_len)
        final_flops = base_flops + mamba_flops
        
        flops_formatted, params_formatted = clever_format([final_flops, total_params], "%.3f")
        mamba_formatted = clever_format([mamba_flops], "%.3f")[0]
        print(f"{'='*40}\n")
        print(f"Calculating FLOPs based on 5 seconds of audio (500 frames)...")
        print(f"🔹 Model: {model.__class__.__name__}")
        print(f"🔹 Total FLOPs: {flops_formatted}")    
        print(f"🔹 Total Params: {params_formatted}")
        print(f"{'='*40}\n")
        
        return final_flops, total_params
        
    except Exception as e:
        print(f"❌ FLOPs calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return 0, total_params