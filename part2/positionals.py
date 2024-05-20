import torch
import math
import argparse
import os

class SinusoidalPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(SinusoidalPositionalEncoding, self).__init__()
        
        # Initialize matrix with shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # Calculate encoding values
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)    # Even
        pe[:, 1::2] = torch.cos(position * div_term)    # Odd
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register buffer so it won't be considered a model parameter but will be in the state dict
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class LearnablePositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(LearnablePositionalEmbedding, self).__init__()
        self.position_embeddings = torch.nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(0)
        if seq_len > self.position_embeddings.num_embeddings:
            raise ValueError(f"Sequence length {seq_len} is greater than max_len {self.position_embeddings.num_embeddings}")
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(1).expand_as(x[:, :, 0])
        return x + self.position_embeddings(position_ids)

def get_pos_encoder(args):
    if args.positional == "spe":
        pos_encoder = SinusoidalPositionalEncoding(args.d_model, args.max_len)
    elif args.positional == "lpe":
        pos_encoder = LearnablePositionalEmbedding(args.d_model, args.max_len)
    else:
        raise ValueError(f"Unknown encoder type: {args.positional}")
    return pos_encoder

def main(args):
    # Get positional encoder
    pos_encoder = get_pos_encoder(args)
    print(f"Using encoder: {args.positional}")

    # Set input tensor (seq_len, batch_size, d_model)
    input_tensor = torch.zeros(args.seq_len, args.batch_size, args.d_model)
    
    # Encode positions
    output_tensor = pos_encoder(input_tensor)
    print(f"Output tensor shape: {output_tensor.shape}")
    print(output_tensor)
    print(output_tensor.shape)  

    # Save output to file if provided
    if args.output_path:
        output_dir = os.path.dirname(args.output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(args.output_path, 'w') as f:
            f.write(f"Output tensor shape: {output_tensor.shape}\n")
            f.write(f"Output tensor: {output_tensor}\n")
        print(f"Output saved to: {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply positional encoding to an input tensor.")
    parser.add_argument('--positional', type=str, choices=['spe', 'lpe'], default='spe', help="Type of positional encoding to use.")
    parser.add_argument('--d_model', type=int, default=512, help="Dimension of the model.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for positional encoder.")
    parser.add_argument('--seq_len', type=int, default=50, help="Length of the sequence to be encoded.")
    parser.add_argument('--max_len', type=int, default=1000, help="Maximum length of the sequences to be encoded.")
    parser.add_argument('--output_path', type=str, default=None, help="Path to save the encoded and decoded output.")
    args = parser.parse_args()

    main(args)
