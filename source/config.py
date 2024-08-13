import argparse

def parse_args_llm():
    parser = argparse.ArgumentParser(description="MolX-24")
    parser.add_argument("--project", type=str, default="MolX-24")

    # LLM related
    parser.add_argument("--model_name", type=str, default='baseline_llm')
    parser.add_argument("--llm_model_name", type=str, default='llama-7b')
    parser.add_argument("--llm_frozen", type=str, default='True')
    parser.add_argument("--graph_enc", type=str, default='graphcl')
    parser.add_argument("--smiles_enc", type=str, default='chemberta')
    parser.add_argument('--wo_init', action='store_true', default=False)
    parser.add_argument('--cl_init', action='store_true', default=False)
    parser.add_argument("--seed", type=int, default=0)

    # Model Training
    parser.add_argument("--dataset", type=str, default='baseline_generation')
    parser.add_argument("--data", type=str)
    parser.add_argument("--split", type=str, default="pretrain")
    parser.add_argument("--task", type=str, default="Label")
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--use_smiles', action='store_true', default=False)
    parser.add_argument('--use_llasmol', action='store_true', default=False)
    parser.add_argument('--use_t5chem', action='store_true', default=False)
    parser.add_argument('--xtokens', action='store_true', default=False)
    parser.add_argument("--xtokens_init", type=str, default='generation task')
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--grad_steps", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--warmup_epochs", type=float, default=1)

    # Inference
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--icl", type=int, default=0)

    # Checkpoint
    parser.add_argument("--run_name", type=str, default='')
    parser.add_argument("--output_dir", type=str, default='output')
    parser.add_argument("--checkpoint_path", type=str, default=None)

    args = parser.parse_args()
    return args