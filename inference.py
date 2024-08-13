import os
import re
import tqdm
import torch
from torch.utils.data import DataLoader

from source.utils.help_funcs import seed_everything
from source.config import parse_args_llm
from source.utils.help_funcs import _save_checkpoint, _reload_model
from source.models import load_model, get_llm_model_path
from source.datasets import load_dataset
from source.utils.evaluation import eval_funcs
from source.utils.help_funcs import collate_fn

import warnings; warnings.filterwarnings("ignore")

def main(args):
    seed = args.seed
    seed_everything(seed=seed)

    # Step 1: Build Dataset
    test_dataset = load_dataset[args.dataset](data = args.data, split = 'test', task = args.task, icl = args.icl, use_smiles = args.use_smiles, use_t5chem = args.use_t5chem)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)

    # Step 2: Build Model
    args.llm_model_path = get_llm_model_path[args.llm_model_name]
    model = load_model[args.model_name](args=args)
    if args.checkpoint_path is not None:
        model = _reload_model(model, args.checkpoint_path)

    # Step 3. Evaluating
    model.eval()
    eval_output = []
    progress_bar_test = tqdm.tqdm(range(len(test_loader)))
    for _, batch in enumerate(test_loader):
        with torch.no_grad():
            if "chem" not in args.llm_model_name:
                output = model.inference(batch)
                eval_output.append(output)
            else:
                if "t5" not in args.llm_model_name:
                    output = model.model.generate(
                        **model.tokenizer(batch["question"][0], return_tensors="pt").to("cuda"),
                        pad_token_id=model.tokenizer.eos_token_id,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        top_k=20,
                        top_p=0.9,
                        temperature=0.9,
                        repetition_penalty=1.05,
                    )
                    output = model.tokenizer.decode(output[0], skip_special_tokens=True)[len(batch["question"][0]):]
                else:
                    output = model.model.generate(
                        **model.tokenizer(batch["question"][0], return_tensors="pt").to("cuda"),
                        pad_token_id=model.tokenizer.eos_token_id,
                        max_new_tokens=args.max_new_tokens,
                    )
                    output = model.tokenizer.decode(output[0], skip_special_tokens=True).replace(' ', '').rstrip('.')
                eval_output.append({
                    'id': batch["id"],
                    'pred': [output.replace("</s>", "").strip()],
                    'label': batch["label"],
                })

        progress_bar_test.update(1)

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