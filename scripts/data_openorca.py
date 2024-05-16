from datasets import load_dataset
from transformers import GPT2Tokenizer

def load_and_tokenize_data():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Set the pad token to the eos token
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("Open-Orca/OpenOrca")

    def tokenize_function(examples):
        return tokenizer(examples['id'], truncation=True, padding='max_length')

    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["id", "system_prompt", "question", "response"])

    return tokenized_datasets

if __name__ == "__main__":
    tokenized_datasets = load_and_tokenize_data()
    tokenized_datasets.save_to_disk('./data/openorca/')
