import torch
import torch.nn.functional as F
import argparse
import logging
import math
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, d_model):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(1, 2)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, v)
        return output, attention

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linear_q = torch.nn.Linear(d_model, d_model)
        self.linear_k = torch.nn.Linear(d_model, d_model)
        self.linear_v = torch.nn.Linear(d_model, d_model)
        self.linear_out = torch.nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scaled_dot_product = ScaledDotProductAttention(self.d_k)
        output, attention = scaled_dot_product(q, k, v, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.linear_out(output)
        return output, attention

class LinearAttention(torch.nn.Module):
    def __init__(self, d_model):
        super(LinearAttention, self).__init__()
        self.scale = 1.0 / math.sqrt(d_model)
        self.linear_q = torch.nn.Linear(d_model, d_model)
        self.linear_k = torch.nn.Linear(d_model, d_model)
        self.linear_v = torch.nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)
        v = v.softmax(dim=-2)
        k = k * self.scale

        attn_weights = torch.einsum("bqd,bkd->bqk", q, k)
        output = torch.einsum("bqk,bkd->bqd", attn_weights, v)

        return output, attn_weights

class NystromAttention(torch.nn.Module):
    def __init__(self, d_model, num_landmarks):
        super(NystromAttention, self).__init__()
        self.num_landmarks = num_landmarks
        self.scale = 1.0 / math.sqrt(d_model)
        self.proj_q = torch.nn.Linear(d_model, d_model)
        self.proj_k = torch.nn.Linear(d_model, d_model)
        self.proj_v = torch.nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        batch_size, seq_len, d_model = q.size()
        q = self.proj_q(q)
        k = self.proj_k(k)
        v = self.proj_v(v)

        k = k.softmax(dim=-1)
        q = q.softmax(dim=-1)

        # Partition sequences into landmarks
        q_landmarks = q.view(batch_size, self.num_landmarks, -1, d_model).mean(dim=-2)
        k_landmarks = k.view(batch_size, self.num_landmarks, -1, d_model).mean(dim=-2)

        # Compute the approximated attention
        kernel_1 = torch.einsum("bqd,bkd->bqk", q, k_landmarks)
        kernel_2 = torch.einsum("bkd,bld->bkl", k_landmarks, k)
        kernel_3 = torch.einsum("bld,bld->bld", k, v)

        kernel_2_inv = torch.linalg.pinv(kernel_2)
        output = torch.einsum("bqk,bkl,bld->bqd", kernel_1, kernel_2_inv, kernel_3)

        return output, None

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
    try:
        # Get attention method
        attn = get_attention(args)
        logger.info(f"Using attention mechanism: {args.attention}")

        # Set input tensors (batch_size, seq_len, d_model)
        q = torch.rand(32, 50, args.d_model)
        k = torch.rand(32, 50, args.d_model)
        v = torch.rand(32, 50, args.d_model)
        mask = None  

        # Generate attention vectors
        output, attention_weights = attn(q, k, v)

        logger.info(f"Output tensor shape: {output.shape}")
        print(output.shape)

        # Save output to file if provided
        if args.output_path:
            output_dir = os.path.dirname(args.output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            with open(args.output_path, 'w') as f:
                f.write(f"Output tensor shape: {output.shape}\n")
                f.write(f"Output tensor: {output}\n")
            logger.info(f"Output saved to: {args.output_path}")

    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply different attention mechanisms to input tensors.")
    parser.add_argument('--attention', type=str, choices=['scaled_dot_product', 'multi_head', 'linear', 'nystrom'], default='scaled_dot_product', help="Type of attention mechanism to use.")
    parser.add_argument('--d_model', type=int, default=512, help="Dimension of the model.")
    parser.add_argument('--num_heads', type=int, default=8, help="Number of attention heads (for multi-head attention).")
    parser.add_argument('--num_landmarks', type=int, default=10, help="Number of landmarks (for Nyström attention).")
    parser.add_argument('--output_path', type=str, default=None, help="Path to save the encoded and decoded output.")
    args = parser.parse_args()

    main(args)
