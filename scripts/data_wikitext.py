from datasets import load_dataset
from transformers import GPT2Tokenizer

def load_and_tokenize_data():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Set the pad token to the eos token
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length')

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_datasets

if __name__ == "__main__":
    tokenized_datasets = load_and_tokenize_data()
    tokenized_datasets.save_to_disk('./data/wikitext/')
