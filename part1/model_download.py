from transformers import GPT2Tokenizer, GPT2LMHeadModel

def download_gpt2_model(model_name='gpt2', save_directory='./models/gpt2'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Save tokenizer and model
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)

if __name__ == "__main__":
    download_gpt2_model()
