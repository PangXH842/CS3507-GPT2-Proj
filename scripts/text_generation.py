from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, model=1):
    if model == 1:
        model = GPT2LMHeadModel.from_pretrained('./models/fine_tuned/')
        tokenizer = GPT2Tokenizer.from_pretrained('./models/fine_tuned/')
    else:
        model = GPT2LMHeadModel.from_pretrained('./models/gpt2/')
        tokenizer = GPT2Tokenizer.from_pretrained('./models/gpt2/')
    tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        temperature=0.9,
        max_length=200,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]

    return gen_text

if __name__ == "__main__":
    models = ["gpt2","fine_tuned"]
    model = 1
    while True:
        prompt = input(f"Enter your prompt here (Type 'switch' to switch model, current: {models[model]}): ")
        if prompt == "":
            break
        elif prompt == "switch":
            model = 1 - model
            continue
        output = generate_text(prompt)
        output = str(output).strip()
        print("="*40)
        print(output)
        print("="*40)
