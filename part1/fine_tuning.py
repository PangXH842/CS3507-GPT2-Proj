import os
import torch
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_from_disk

def fine_tune_model(data_path, model_path):
    try:
        tokenized_datasets = load_from_disk(data_path)
    except:
        print("Invalid data path!")
        exit(-1)

    # Check if CUDA is available and print device name
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    model_dir = './models/gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Use absolute paths for directories
    base_dir = os.path.abspath(".")
    output_dir = os.path.join(base_dir, "results")
    logging_dir = os.path.join(output_dir, 'logs')

    # Ensure directories exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    print(f"Output Directory: {output_dir}")
    print(f"Logging Directory: {logging_dir}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=logging_dir,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),  # Enable fp16 only if CUDA is available
    )

    # Trainer automatically uses GPU if available
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
    )

    trainer.train()
    model.save_pretrained(os.path.join(base_dir, model_path))
    tokenizer.save_pretrained(os.path.join(base_dir, model_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/wikitext/')
    parser.add_argument('--model_path', type=str, default='./models/wikitext/')
    args = parser.parse_args()

    fine_tune_model(args.data_path, args.model_path)