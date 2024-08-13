from source.models.baseline_llm import BaselineLLM
from source.models.molx_llm import MolXLLM

load_model = {
    'baseline_llm': BaselineLLM,
    'molx_llm': MolXLLM,
}

# Replace the following with the model paths
get_llm_model_path = {
    'llama-7b'   : 'meta-llama/Llama-2-7b-chat-hf' ,
    'mistr-7b'   : 'mistralai/Mistral-7B-Instruct-v0.1' ,

    'chemllama-13b' : 'X-LANCE/ChemDFM-13B-v1.0' ,
}