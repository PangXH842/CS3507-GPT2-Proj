# CS3507-GPT2-Proj

This is a final project of the SJTU 2023-2024-2 CS3507 course based on OpenAI's GPT-2, which consists of two parts. 

Part 1 works on fine-tuning the GPT-2 model. The dataset we have chosen to finetune our model is wikitext(wikitext-2-raw-v1). Part 2 makes modifications on the GPT-2 model with various tokenizers, positional encoders, and attention mechanisms. This repository includes the Python scripts and other files needed to run the processes: 

```
- evaluation
    - evaluation_texts.csv
- part1
    - model_download.py
    - data_wikitext.py
    - fine_tuning.py
    - text_generation.py
    - evaluation.py
- part2
    - attention.py
    - positionals.py
    - token_encodings.py
    - evaluation.py
```

Required libraries:

```
pandas torch torchvision datasets transformers nltk
```

To successfully run this project, follow the steps below:

### Part 1:

1. Run `python model_download.py` to download and save the GPT-2 model
2. Run `python data_wikitext.py` to download and save the wikitext dataset
3. Run `python fine_tuning.py` with default parameters to fine-tune the GPT-2 model with wikitext dataset
4. Run 
    `python evaluation.py --model_path ./models/gpt2`
or 
    `python evaluation.py --model_path ./models/wikitext`
to evaluate the performance of the original GPT-2 model and the fine-tuned model on the selected `.csv` file containing prompts. The provided prompts are given in `./evaluation/evaluation_texts.csv` .

Optionally, `text_generation.py` can be run to test and compare the outputs of the original and fine-tuned model by interacting on a shell-like interface. To exit the interface, press enter without entering additional inputs.

### Part 2:

1. Run `python evaluation.py` without additional parameters to evaluate the GPT-2 with the evaluation file `./evaluation/evaluation_texts.csv` with default settings. 
2. To experiment with different tokenizers, positional encoders and attention mechanisms, run `python evaluation.py [--tokenizer] [--attention] [--positional]` with specified parameters. 
3. For further customization, you may define more parameters when running the evaluation file. The full list of parameters is as follows:
`evaluation.py [--model_path] [--eval_file] [--tokenizer] [--attention] [--positional] [--batch_size] [--max_len] [--d_model] [--num_heads] [--num_landmarks]`
4. Optional: 
    1. Run 
    `token_encodings.py [--tokenizer] [--text] [--text_path] [--output_path]` 
    to encode and decode a given text using different tokenizers
    2. Run 
    `positionals.py [--positionals] [--d_model] [--batch_size] [--seq_len] [--max_len] [--output_path]` 
    to demonstrate positional encoding to input tensors
    3. Run 
    `attention.py [--attention] [--d_model] [--text] [--text_path] [--batch_size] [--seq_len] [--num_heads] [--num_landmarks] [--output_path]` 
    to view the effect of applying different attention mechanisms to input tensors

### Conclusion
This project provides a framework for fine-tuning and modifying the GPT-2 model using various tokenizers, positional encodings, and attention mechanisms. 
