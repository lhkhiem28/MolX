import os
import re
import tqdm
import torch
from torch.utils.data import DataLoader

from source.utils.help_funcs import seed_everything
from source.config import parse_args_llm
from source.utils.help_funcs import _save_checkpoint, _reload_model
from source.llasmol.generation import LlaSMolGeneration
from source.datasets import load_dataset
from source.utils.evaluation import eval_funcs
from source.utils.help_funcs import collate_fn

import warnings; warnings.filterwarnings("ignore")

def main(args):
    seed = args.seed
    seed_everything(seed=seed)

    # Step 1: Build Dataset
    test_dataset = load_dataset[args.dataset](data = args.data, split = 'test', task = args.task, use_smiles = True, use_llasmol=True)

    # Step 2: Build Model
    model = LlaSMolGeneration('osunlp/LlaSMol-Galactica-6.7B')

    # Step 3. Evaluating
    eval_output = []
    for i in tqdm.tqdm(range(len(test_dataset))):
        with torch.no_grad():
            question = test_dataset[i]["question"]
            try:
                output = model.generate(question, max_new_tokens=args.max_new_tokens)[0]['output'][0]
                match = re.search(r"<SMILES>(.*?)</SMILES>", output)
                if match:
                    output = match.group(1)
                eval_output.append({
                    'id': [i],
                    'pred': [output.replace("</s>", "").strip()],
                    'label': [test_dataset[i]['label']],
                })
            except:
                continue

    # Step 4. Post-processing & Evaluating
    os.makedirs(f'{args.output_dir}/downstream/{args.dataset}/{args.data}', exist_ok=True)
    path = f'{args.output_dir}/{args.dataset}/{args.data}/{args.model_name}_{args.llm_model_name}_llm_frozen{args.llm_frozen}_{args.graph_enc}_{args.smiles_enc}_{args.num_epochs}epochs_lr{args.lr}_{args.split}_best_{args.run_name}_{args.task.replace(" ", "-")}_{args.icl}examples.csv'
    if "prediction" in args.dataset:
        path = f'{args.output_dir}/downstream/{args.dataset}/{args.data}/{args.model_name}_{args.llm_model_name}_llm_frozen{args.llm_frozen}_{args.graph_enc}_{args.smiles_enc}_{args.num_epochs}epochs_lr{args.lr}_{args.split}_best_{args.run_name}_{args.task.replace(" ", "-")}_{args.icl}examples.csv'
        acc, f1, err = eval_funcs[args.dataset](eval_output, path)
        print('Test ACC (F1) | Err: {:.3f} ({:.3f}) | {:.3f}'.format(acc, f1, err))
    if "regression" in args.dataset:
        path = f'{args.output_dir}/downstream/{args.dataset}/{args.data}/{args.model_name}_{args.llm_model_name}_llm_frozen{args.llm_frozen}_{args.graph_enc}_{args.smiles_enc}_{args.num_epochs}epochs_lr{args.lr}_{args.split}_best_{args.run_name}_{args.task.replace(" ", "-")}_{args.icl}examples.csv'
        rmse, err = eval_funcs[args.dataset](eval_output, path)
        print('Test RMSE | Err: {:.3f} | {:.3f}'.format(rmse, err))
    if "generation" in args.dataset:
        path = f'{args.output_dir}/downstream/{args.dataset}/{args.data}/{args.model_name}_{args.llm_model_name}_llm_frozen{args.llm_frozen}_{args.graph_enc}_{args.smiles_enc}_{args.num_epochs}epochs_lr{args.lr}_{args.split}_best_{args.run_name}_{args.task.replace(" ", "-")}_{args.icl}examples.csv'
        scores = eval_funcs[args.dataset](eval_output, path, model.tokenizer, args.max_new_tokens, args.task)
        if args.task in ["Description", "IUPAC name"]:
            print("BLEU-2: {:.4f} BLEU-4: {:.4f} | ROUGE-1: {:.4f} ROUGE-2: {:.4f} ROUGE-L: {:.4f} | METEOR: {:.4f}".format(
                *scores
            ))
        else:
            print("ExactMatch: {:.4f} BLEU-2: {:.4f} METEOR: {:.4f} Levenshtein: {:.4f} | MACCS FTS: {:.4f} Morgan FTS: {:.4f} | Validity: {:.4f}".format(
                *scores
            ))

if __name__ == "__main__":
    args = parse_args_llm()
    main(args)