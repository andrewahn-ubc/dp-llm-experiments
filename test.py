from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftModel
import pandas as pd

LLM_NAME = "/home/taegyoem/scratch/llama2_7b_chat_hf" 
TESTING_DATA = "./data/test.csv"
FINETUNED_LLM_PATH = "/home/taegyoem/scratch/finetuned_llm"

base_model = AutoModelForCausalLM.from_pretrained(LLM_NAME)
finetuned_model = PeftModel.from_pretrained(base_model, FINETUNED_LLM_PATH)

df = pd.read_csv(TESTING_DATA)

# generate 30 tokens for each perturbed prompt using the finetuned LLM

# save results in a new dataframe and save it as a new csv file

