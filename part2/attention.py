import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import math
import os
from einops import rearrange, reduce

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, hidden_states, qkv=None, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        # Unpack hidden states into query, key, and value
        if hidden_states != None:
            q, k, v = hidden_states, hidden_states, hidden_states
        else:
            q, k, v = qkv

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        if head_mask is not None:
            attention = attention * head_mask
        output = torch.matmul(attention, v)

        if use_cache:
            present = (k, v)
        else:
            present = None

        outputs = (output, present)
        if output_attentions:
            outputs += (attention,)

        return outputs

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.scaled_dot_product = ScaledDotProductAttention(self.d_k)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        batch_size = hidden_states.size(0)

        q = self.linear_q(hidden_states).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.linear_k(hidden_states).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(hidden_states).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attn_outputs = self.scaled_dot_product(None, (q, k, v), layer_past, attention_mask, head_mask, use_cache, output_attentions)

        output = attn_outputs[0].transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.linear_out(output)

        outputs = (output,) + attn_outputs[1:]
        return outputs

class LinearAttention(nn.Module):
    def __init__(self, d_model):
        super(LinearAttention, self).__init__()
        self.scale = 1.0 / math.sqrt(d_model)
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        batch_size, seq_len, d_model = hidden_states.size()

        q = self.linear_q(hidden_states)
        k = self.linear_k(hidden_states)
        v = self.linear_v(hidden_states)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)
        v = v.softmax(dim=-2)
        k = k * self.scale

        attn_weights = torch.einsum("bqd,bkd->bqk", q, k)
        output = torch.einsum("bqk,bkd->bqd", attn_weights, v)

        if use_cache:
            present = (k, v)
        else:
            present = None

        outputs = (output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class NystromAttention(nn.Module):
    def __init__(self, d_model, num_landmarks, heads=1, pinv_iterations=6, eps=1e-8, residual=False):
        super(NystromAttention, self).__init__()
        self.num_landmarks = num_landmarks
        self.heads = heads
        self.pinv_iterations = pinv_iterations
        self.eps = eps
        self.residual = residual
        self.scale = 1.0 / math.sqrt(d_model // heads)
        self.to_qkv = nn.Linear(d_model, d_model * 3)
        self.to_out = nn.Linear(d_model, d_model)
        if residual:
            self.res_conv = nn.Conv1d(d_model, d_model, 1, groups=heads)

    def moore_penrose_iter_pinv(self, x, num_iters):
        I = torch.eye(x.size(-1), device=x.device)
        K = x
        V = torch.clone(K)
        for _ in range(num_iters):
            V = 0.25 * V @ (13 * I - K @ V @ (15 * I - K @ V @ (7 * I - K @ V)))
        return V

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        b, n, d_model = hidden_states.shape
        h, m, iters, eps = self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # pad so that sequence can be evenly divided into m landmarks
        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            hidden_states = F.pad(hidden_states, (0, 0, 0, padding), value=0)

            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (padding, 0), value=False)

        # derive query, keys, values
        q, k, v = self.to_qkv(hidden_states).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # set masked positions to 0 in queries, keys, values
        if attention_mask is not None:
            attention_mask = rearrange(attention_mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * attention_mask[..., None], (q, k, v))

        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask
        l = torch.ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l=l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l=l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean
        divisor = l
        if attention_mask is not None:
            mask_landmarks_sum = reduce(attention_mask, '... (n l) -> ... n', 'sum', l=l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)
        q_landmarks = q_landmarks / divisor
        k_landmarks = k_landmarks / divisor

        # similarities
        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = torch.einsum(einops_eq, q, k_landmarks)
        sim2 = torch.einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = torch.einsum(einops_eq, q_landmarks, k)

        # masking
        if attention_mask is not None:
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(attention_mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * attention_mask[..., None, :]), mask_value)

        # eq (15) in the paper and aggregate values
        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
        attn2_inv = self.moore_penrose_iter_pinv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # add depth-wise conv residual of values
        if self.residual:
            v_reshaped = v.permute(0, 2, 1).contiguous()
            out = out + self.res_conv(v_reshaped).permute(0, 2, 1).contiguous()

        # merge and combine heads
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)
        out = out[:, -n:]

        if use_cache:
            present = (k, v)
        else:
            present = None

        outputs = (out,)
        if present is not None:
            outputs += (present,)
        if output_attentions:
            outputs += (None,)

        return outputs

def get_attention(args):
    if args.attention == "scaled_dot_product":
        attn = ScaledDotProductAttention(args.d_model)
    elif args.attention == "multi_head":
        attn = MultiHeadAttention(args.d_model, args.num_heads)
    elif args.attention == "linear":
        attn = LinearAttention(args.d_model)
    elif args.attention == "nystrom":
        attn = NystromAttention(args.d_model, args.num_landmarks)
    else:
        raise ValueError(f"Unknown attention type: {args.attention}")
    return attn

def main(args):
    # Get attention method
    attn = get_attention(args)
    print(f"Using attention mechanism: {args.attention}")

    # Set input tensors (batch_size, seq_len, d_model)
    q = torch.rand(args.batch_size, args.seq_len, args.d_model)
    k = torch.rand(args.batch_size, args.seq_len, args.d_model)
    v = torch.rand(args.batch_size, args.seq_len, args.d_model)

    # Generate attention vectors
    output, attention_weights = attn(q, k, v)

    print(f"Output tensor shape: {output.shape}")
    print(output.shape)

    # Save output to file if provided
    if args.output_path:
        output_dir = os.path.dirname(args.output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(args.output_path, 'w') as f:
            f.write(f"Output tensor shape: {output.shape}\n")
            f.write(f"Output tensor: {output}\n")
        print(f"Output saved to: {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply different attention mechanisms to input tensors.")
    parser.add_argument('--attention', type=str, choices=['scaled_dot_product', 'multi_head', 'linear', 'nystrom'], default='scaled_dot_product', help="Type of attention mechanism to use.")
    parser.add_argument('--d_model', type=int, default=512, help="Dimension of the model.")
    parser.add_argument('--text', type=str, default="Hello, how are you?", help="Text to encode. Ignored if --text_path is provided.")
    parser.add_argument('--text_path', type=str, default=None, help="Path to a text file to encode.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for positional encoder.")
    parser.add_argument('--seq_len', type=int, default=50, help="Length of the sequence to be encoded.")
    parser.add_argument('--num_heads', type=int, default=8, help="Number of attention heads (for multi-head attention).")
    parser.add_argument('--num_landmarks', type=int, default=10, help="Number of landmarks (for Nyström attention).")
    parser.add_argument('--output_path', type=str, default=None, help="Path to save the encoded and decoded output.")
    args = parser.parse_args()

    main(args)
