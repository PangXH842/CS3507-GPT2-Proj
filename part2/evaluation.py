import torch
import argparse
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from nltk.translate.bleu_score import sentence_bleu
import logging
import os

from attention import get_attention
from positionals import get_pos_encoder
from token_encodings import get_tokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define custom positional encodings
class SinusoidalPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class LearnablePositionalEmbedding(torch.nn.Module):
    def __init__(self, max_len, d_model):
        super(LearnablePositionalEmbedding, self).__init__()
        self.position_embeddings = torch.nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(0)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(1).expand_as(x[:, :, 0])
        return x + self.position_embeddings(position_ids)

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
    parser.add_argument('--tokenizer_type', type=str, choices=['gpt2', 'bert', 'albert', 'xlnet'], default='gpt2', help="Type of tokenizer to use")
    parser.add_argument('--attention_type', type=str, choices=['scaled_dot_product', 'multi_head', 'linear', 'nystrom'], default='scaled_dot_product', help="Type of attention mechanism to use")
    parser.add_argument('--positional_encoding', type=str, choices=['sinusoidal', 'learnable'], default='sinusoidal', help="Type of positional encoding to use")
    args = parser.parse_args()

    try:
        # Load tokenizer
        tokenizer = get_tokenizer(args.tokenizer_type)
        tokenizer.pad_token = tokenizer.eos_token

        # Load model configuration
        config = GPT2Config.from_pretrained(args.model_path)
        config.num_landmarks = 10  # Set number of landmarks for Nystrom attention

        # Load model
        model = GPT2LMHeadModel.from_pretrained(args.model_path, config=config)

        # Modify the model's attention mechanism
        if args.attention_type != "scaled_dot_product":
            for layer in model.transformer.h:
                layer.attn = get_attention(args.attention_type, config)

        # Add custom positional encoding if required
        if args.positional_encoding == 'sinusoidal':
            positional_encoding = SinusoidalPositionalEncoding(config.n_embd)
        elif args.positional_encoding == 'learnable':
            positional_encoding = LearnablePositionalEmbedding(5000, config.n_embd)

        for layer in model.transformer.h:
            layer.attn.register_buffer('positional_encoding', positional_encoding(torch.zeros(1, 1, config.n_embd)))

    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}")
        exit(1)

    try:
        # Load evaluation texts from CSV
        df = pd.read_csv(args.eval_file)
        if 'prompt' not in df.columns or 'references' not in df.columns:
            logger.error("CSV file must contain 'prompt' and 'references' columns.")
            exit(1)
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        exit(1)

    total_perplexity = 0
    total_bleu = 0
    for index, row in df.iterrows():
        prompt = row['prompt']
        references = row['references'].split('|')
        logger.info(f"\nEvaluating prompt {index+1} of {df.shape[0]}: {prompt}")

        # Generate text
        generated_text = generate_text(model, tokenizer, prompt)
        generated_text = str(generated_text).strip()
        logger.info(f"Generated Text: {generated_text}")

        # Calculate perplexity
        perplexity = calculate_perplexity(model, tokenizer, generated_text)
        logger.info(f"Perplexity: {perplexity}")
        total_perplexity += perplexity

        # Evaluate BLEU score
        bleu_score = evaluate_bleu(references, generated_text)
        logger.info(f"BLEU score: {bleu_score}")
        total_bleu += bleu_score

    logger.info(f"\nAverage Perplexity: {total_perplexity/df.shape[0]}")
    logger.info(f"Average BLEU score: {total_bleu/df.shape[0]}")
