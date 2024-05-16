import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

def load_cbt_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        parts = line.strip().split('|')
        context = parts[0]
        options = parts[1:5]
        correct = parts[5]
        data.append((context, options, correct))
    return data

def evaluate_cbt(model, tokenizer, data):
    correct_predictions = 0
    total_predictions = 0

    for context, options, correct in tqdm(data):
        input_ids = tokenizer(context, return_tensors='pt').input_ids
        logits = model(input_ids).logits
        scores = []

        for option in options:
            option_ids = tokenizer(option, return_tensors='pt').input_ids
            option_score = torch.sum(logits[:, -option_ids.size(1):, :] == option_ids)
            scores.append(option_score.item())

        predicted_option = options[scores.index(max(scores))]
        if predicted_option == correct:
            correct_predictions += 1

        total_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy

if __name__ == "__main__":
    # Load models
    model = GPT2LMHeadModel.from_pretrained('./model/wikitext')
    tokenizer = GPT2Tokenizer.from_pretrained('./model/wikitext')
    tokenizer.pad_token = tokenizer.eos_token

    # Load CBT dataset
    cbt_data_path = 'path_to_cbt_dataset'  # Update this with the correct path
    cbt_data = load_cbt_data(cbt_data_path)

    # Evaluate model on CBT
    accuracy = evaluate_cbt(model, tokenizer, cbt_data)
    print(f"CBT Accuracy: {accuracy}")
