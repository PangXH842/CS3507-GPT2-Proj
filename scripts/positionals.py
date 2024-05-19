import torch
import math
import argparse

class SinusoidalPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        
        # Create a matrix of shape (max_len, d_model) to store the positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate the division term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices in the array; 2i
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices in the array; 2i+1
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register buffer so it won't be considered a model parameter but will be in the state dict
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class LearnablePositionalEmbedding(torch.nn.Module):
    def __init__(self, max_len, d_model):
        super(LearnablePositionalEmbedding, self).__init__()
        self.position_embeddings = torch.nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(0)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(1).expand_as(x[:, :, 0])
        return x + self.position_embeddings(position_ids)

def choose_encoder(args):
    match args.encoder:
        case "spe":
            pos_encoder = SinusoidalPositionalEncoding(args.d_model, args.max_len)
        case "lpe":
            pos_encoder = LearnablePositionalEmbedding(args.d_model, args.max_len)
    return pos_encoder

def main(args):
    pos_encoder = choose_encoder(args)

    # Sample input tensor (seq_len, batch_size, d_model)
    input_tensor = torch.zeros(50, 32, args.d_model)
    output_tensor = pos_encoder(input_tensor)
    print(output_tensor.shape)  # Should print torch.Size([50, 32, 512])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='bpe')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--max_len', type=int, default=5000)
    args = parser.parse_args()

    main(args)