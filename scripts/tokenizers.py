import argparse
from transformers import GPT2Tokenizer, BertTokenizer, AlbertTokenizer, XLNetTokenizer

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
    return tokenizer

def main(args):
    # Get the selected tokenizer
    tokenizer = get_tokenizer(args.tokenizer)

    text = args.text
    tokens = tokenizer.encode(text)
    print(tokens)
    decoded_text = tokenizer.decode(tokens)
    print(decoded_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=str, default='bpe')
    parser.add_argument('--text', type=str, default="Hello, how are you?")
    args = parser.parse_args()

    main(args)