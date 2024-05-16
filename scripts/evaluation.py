import torch
import argparse
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.translate.bleu_score import sentence_bleu

def generate_text(model, tokenizer, prompt, max_length=100):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        temperature=0.9,
        max_length=max_length,
        eos_token_id=tokenizer.eos_token_id
    )
    gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]

    return gen_text

def calculate_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return perplexity.item()

def evaluate_bleu(reference, candidate):
    reference_tokens = [ref.split() for ref in reference]
    candidate_tokens = candidate.split()
    return sentence_bleu(reference_tokens, candidate_tokens)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./models/wikitext", help="Path to the model directory")
    parser.add_argument('--tokenizer_path', type=str, default="./models/wikitext", help="Path to the tokenizer directory")
    parser.add_argument('--eval_file', type=str, default="./evaluation/evaluation_texts.csv", help="Path to the evaluation CSV file")
    args = parser.parse_args()

    try:
        # Load model and tokenizer
        model = GPT2LMHeadModel.from_pretrained(args.model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        exit(1)

    try:
        # Load evaluation texts from CSV
        df = pd.read_csv(args.eval_file)
        if 'prompt' not in df.columns or 'references' not in df.columns:
            print("CSV file must contain 'prompt' and 'references' columns.")
            exit(1)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        exit(1)

    for index, row in df.iterrows():
        prompt = row['prompt']
        references = row['references'].split('|')
        print(f"\nEvaluating prompt {index+1} of {df.shape[0]}: {prompt}")

        # Generate text
        generated_text = generate_text(model, tokenizer, prompt)
        generated_text = str(generated_text).strip()
        print(f"Generated Text: {generated_text}")

        # Calculate perplexity
        perplexity = calculate_perplexity(model, tokenizer, generated_text)
        print(f"Perplexity: {perplexity}")

        # Evaluate BLEU score
        bleu_score = evaluate_bleu(references, generated_text)
        print(f"BLEU score: {bleu_score}")
