import argparse
from transformers import GPT2Tokenizer, BertTokenizer, AlbertTokenizer, XLNetTokenizer
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_tokenizer(t):
    match t:
        case "bpe":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        case "wordpiece":
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        case "sentencepiece":
            tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        case "unigram":
            tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        case _:
            raise ValueError(f"Unknown tokenizer type: {t}")
    return tokenizer

def main(args):
    try:
        tokenizer = get_tokenizer(args.tokenizer)
        logger.info(f"Using tokenizer: {args.tokenizer}")

        # Read text from file if path is provided, else use the provided text
        if args.text_path:
            if not os.path.exists(args.text_path):
                raise FileNotFoundError(f"Text file not found: {args.text_path}")
            with open(args.text_path, 'r') as f:
                text = f.read()
        else:
            text = args.text

        logger.info(f"Text to encode: {text}")

        tokens = tokenizer.encode(text)
        logger.info(f"Encoded tokens: {tokens}")

        decoded_text = tokenizer.decode(tokens)
        logger.info(f"Decoded text: {decoded_text}")

        # Print output to console
        print(f"Original text: {text}")
        print(f"Encoded tokens: {tokens}")
        print(f"Decoded text: {decoded_text}")

        # Save tokenized output to a file if output path is provided
        if args.output_path:
            output_dir = os.path.dirname(args.output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(args.output_path, 'w') as f:
                f.write(f"Original text: {text}\n")
                f.write(f"Encoded tokens: {tokens}\n")
                f.write(f"Decoded text: {decoded_text}\n")
            logger.info(f"Output saved to: {args.output_path}")

    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text encoding and decoding using different tokenizers.")
    parser.add_argument('--tokenizer', type=str, choices=['bpe', 'wordpiece', 'sentencepiece', 'unigram'], default='bpe', help="Type of tokenizer to use.")
    parser.add_argument('--text', type=str, default="Hello, how are you?", help="Text to encode. Ignored if --text_path is provided.")
    parser.add_argument('--text_path', type=str, default=None, help="Path to a text file to encode.")
    parser.add_argument('--output_path', type=str, default=None, help="Path to save the encoded and decoded output.")
    args = parser.parse_args()

    main(args)
