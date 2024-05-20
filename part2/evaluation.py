import torch
import argparse
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Config
from nltk.translate.bleu_score import sentence_bleu
import os

from attention import get_attention
from positionals import get_pos_encoder
from token_encodings import get_tokenizer

def generate_text(model, tokenizer, prompt, max_length=50):
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
    parser.add_argument('--eval_file', type=str, default="./evaluation/evaluation_texts.csv", help="Path to the evaluation CSV file")
    parser.add_argument('--tokenizer', type=str, choices=['bpe', 'wordpiece', 'sentencepiece', 'unigram'], default='bpe', help="Type of tokenizer to use.")
    parser.add_argument('--attention', type=str, choices=['scaled_dot_product', 'multi_head', 'linear', 'nystrom'], default='scaled_dot_product', help="Type of attention mechanism to use.")
    parser.add_argument('--positional', type=str, choices=['spe', 'lpe'], default='spe', help="Type of positional encoding to use.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for positional encoder.")
    parser.add_argument('--max_len', type=int, default=1000, help="Maximum length for positional encodings")
    parser.add_argument('--d_model', type=int, default=768, help="Model dimension size")
    parser.add_argument('--num_landmarks', type=int, default=10, help="Number of landmarks (for Nyström attention).")
    args = parser.parse_args()

    # Load model configuration
    config = GPT2Config.from_pretrained(args.model_path)
    config.num_landmarks = 10  # Set number of landmarks for Nystrom attention

    # Load model
    model = GPT2LMHeadModel.from_pretrained(args.model_path, config=config)

    # Load tokenizer
    tokenizer = get_tokenizer(args)

    # Load positional encoder
    positional_encoding = get_pos_encoder(args)

    # Load attention mechanism
    attention = get_attention(args)

    # Apply for each layer
    for layer in model.transformer.h:
        layer.attn = attention
        layer.attn.register_buffer('positional_encoding', positional_encoding(torch.zeros(1, 1, config.n_embd)))

    # Load evaluation texts from CSV
    df = pd.read_csv(args.eval_file)
    if 'prompt' not in df.columns or 'references' not in df.columns:
        print("[ERROR] CSV file must contain 'prompt' and 'references' columns.")
        exit(1)

    total_perplexity = 0
    total_bleu = 0
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
        total_perplexity += perplexity

        # Evaluate BLEU score
        bleu_score = evaluate_bleu(references, generated_text)
        print(f"BLEU score: {bleu_score}")
        total_bleu += bleu_score

    print(f"\nAverage Perplexity: {total_perplexity/df.shape[0]}")
    print(f"Average BLEU score: {total_bleu/df.shape[0]}")
