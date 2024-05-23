import argparse
from transformers import GPT2Tokenizer, BertTokenizer, AlbertTokenizer, XLNetTokenizer
import os

def get_tokenizer(args):
    match args.tokenizer:
        case "bpe":         # Byte Pair Encoding
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        case "wordpiece":
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        case "sentencepiece":
            tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        case "unigram":
            tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        case _:
            raise ValueError(f"Unknown tokenizer type: {args.tokenizer}")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def main(args):
    # Get tokenizer
    tokenizer = get_tokenizer(args)
    print(f"Using tokenizer: {args.tokenizer}")

    # Read text from file if path is provided, else use provided text
    if args.text_path:
        if not os.path.exists(args.text_path):
            raise FileNotFoundError(f"Text file not found: {args.text_path}")
        with open(args.text_path, 'r') as f:
            text = f.read()
    else:
        text = args.text

    # Print text from input
    print(f"Original text: {text}")

    # Encode tokens
    tokens = tokenizer.encode(text)
    print(f"Encoded tokens: {tokens}")

    # Decode text for viewing
    decoded_text = tokenizer.decode(tokens)
    print(f"Decoded text: {decoded_text}")

    # Save output to file if provided
    if args.output_path:
        output_dir = os.path.dirname(args.output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(args.output_path, 'w') as f:
            f.write(f"Original text: {text}\n")
            f.write(f"Encoded tokens: {tokens}\n")
            f.write(f"Decoded text: {decoded_text}\n")
        print(f"Output saved to: {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text encoding and decoding using different tokenizers.")
    parser.add_argument('--tokenizer', type=str, choices=['bpe', 'wordpiece', 'sentencepiece', 'unigram'], default='bpe', help="Type of tokenizer to use.")
    parser.add_argument('--text', type=str, default="Hello, how are you?", help="Text to encode. Ignored if --text_path is provided.")
    parser.add_argument('--text_path', type=str, default=None, help="Path to a text file to encode.")
    parser.add_argument('--output_path', type=str, default=None, help="Path to save the encoded and decoded output.")
    args = parser.parse_args()

    main(args)
