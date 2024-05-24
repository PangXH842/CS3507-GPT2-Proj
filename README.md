# CS3507-GPT2-Proj

This project works on fine-tuning OpenAI's GPT-2 model. The dataset we have chosen to finetune our model is wikitext(wikitext-2-raw-v1). This repository includes the Python scripts needed to run the processes: 

```
- scripts
    - model_download.py
    - data_wikitext.py
    - fine_tuning.py
    - text_generation.py
    - evaluation.py
```

To run our project, the recommended steps are as follows:

1. Run `model_download.py` to download and save the GPT-2 model
2. Run `data_wikitext.py` to download and save the wikitext dataset
3. Run `fine_tuning.py` with default parameters to fine-tune the GPT-2 model with wikitext dataset
4. Run 
    `evaluation.py --model_path ./models/gpt2`
or 
    `evaluation.py --model_path ./models/wikitext`
to evaluate the performance of the original GPT-2 model and the fine-tuned model on the selected `.csv` file containing prompts. The provided prompts are given in `./evaluation/evaluation_texts.csv` .

Optionally, `text_generation.py` can be run to test and compare the outputs of the original and fine-tuned model by interacting on a shell-like interface. To exit the interface, press enter without entering additional inputs.