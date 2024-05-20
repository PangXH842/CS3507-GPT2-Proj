import torch
import math
import argparse
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SinusoidalPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Sinusoidal Positional Encoding module.

        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the sequences to be encoded.
        """
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
        """
        Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor with positional encoding added.
        """
        return x + self.pe[:x.size(0), :]

class LearnablePositionalEmbedding(torch.nn.Module):
    def __init__(self, max_len, d_model):
        """
        Learnable Positional Embedding module.

        Args:
            max_len (int): The maximum length of the sequences to be encoded.
            d_model (int): The dimension of the model.
        """
        super(LearnablePositionalEmbedding, self).__init__()
        self.position_embeddings = torch.nn.Embedding(max_len, d_model)

    def forward(self, x):
        """
        Adds learnable positional embeddings to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor with positional embeddings added.
        """
        seq_len = x.size(0)
        if seq_len > self.position_embeddings.num_embeddings:
            raise ValueError(f"Sequence length {seq_len} is greater than max_len {self.position_embeddings.num_embeddings}")
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(1).expand_as(x[:, :, 0])
        return x + self.position_embeddings(position_ids)

def choose_encoder(args):
    """
    Chooses the positional encoding method based on the provided arguments.

    Args:
        args (argparse.Namespace): The command-line arguments.

    Returns:
        torch.nn.Module: The selected positional encoding module.
    """
    if args.encoder == "spe":
        pos_encoder = SinusoidalPositionalEncoding(args.d_model, args.max_len)
    elif args.encoder == "lpe":
        pos_encoder = LearnablePositionalEmbedding(args.max_len, args.d_model)
    else:
        raise ValueError(f"Unknown encoder type: {args.encoder}")
    return pos_encoder

def main(args):
    """
    Main function to apply positional encoding to a sample input tensor.

    Args:
        args (argparse.Namespace): The command-line arguments.
    """
    try:
        pos_encoder = choose_encoder(args)
        logger.info(f"Using encoder: {args.encoder}")

        # Sample input tensor (seq_len, batch_size, d_model)
        seq_len = 50
        batch_size = 32
        input_tensor = torch.zeros(seq_len, batch_size, args.d_model)
        
        output_tensor = pos_encoder(input_tensor)
        logger.info(f"Output tensor shape: {output_tensor.shape}")
        print(output_tensor)
        print(output_tensor.shape)  # Should print torch.Size([50, 32, 512])

        # Save output to a file if output path is provided
        if args.output_path:
            output_dir = os.path.dirname(args.output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            with open(args.output_path, 'w') as f:
                f.write(f"Output tensor shape: {output_tensor.shape}\n")
                f.write(f"Output tensor: {output_tensor}\n")
            logger.info(f"Output saved to: {args.output_path}")

    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply positional encoding to an input tensor.")
    parser.add_argument('--encoder', type=str, choices=['spe', 'lpe'], default='spe', help="Type of positional encoding to use.")
    parser.add_argument('--d_model', type=int, default=512, help="Dimension of the model.")
    parser.add_argument('--max_len', type=int, default=5000, help="Maximum length of the sequences to be encoded.")
    parser.add_argument('--output_path', type=str, default=None, help="Path to save the encoded and decoded output.")
    args = parser.parse_args()

    main(args)
