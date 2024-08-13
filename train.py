import os
import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import wandb

from source.utils.help_funcs import seed_everything
from source.config import parse_args_llm
from source.utils.help_funcs import _save_checkpoint, _reload_model
from source.models import load_model, get_llm_model_path
from source.datasets import load_dataset
from source.utils.evaluation import eval_funcs
from source.utils.help_funcs import collate_fn
from source.utils.help_funcs import adjust_learning_rate

import warnings; warnings.filterwarnings("ignore")

def main(args):
    seed = args.seed
    seed_everything(seed=seed)
    # if args.split == "pretrain":
    wandb.init(project=f"{args.project}",
            name=f"{args.model_name}_{args.run_name}",
            config=args,
            # mode="offline",
    )

    # Step 1: Build Dataset
    train_dataset = load_dataset[args.dataset](data = args.data, split = args.split, task = args.task, icl = args.icl, use_smiles = args.use_smiles)
    if args.split == "pretrain":
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, 
            load_dataset[args.dataset](data = args.data, split = args.split, task = "Canonical SMILES", icl = args.icl, use_smiles = True), 
            load_dataset[args.dataset.split("_")[0] + "_regression"](data = args.data, split = args.split, task = "Heavy Atom", icl = args.icl, use_smiles = args.use_smiles), 
            load_dataset[args.dataset.split("_")[0] + "_regression"](data = args.data, split = args.split, task = "Hydrogen Bond Acceptor", icl = args.icl, use_smiles = args.use_smiles), 
            load_dataset[args.dataset.split("_")[0] + "_regression"](data = args.data, split = args.split, task = "Hydrogen Bond Donor", icl = args.icl, use_smiles = args.use_smiles), 
            load_dataset[args.dataset.split("_")[0] + "_regression"](data = args.data, split = args.split, task = "Rotatable Bond", icl = args.icl, use_smiles = args.use_smiles), 
            load_dataset[args.dataset.split("_")[0] + "_regression"](data = args.data, split = args.split, task = "Aromatic Ring", icl = args.icl, use_smiles = args.use_smiles), 
            load_dataset[args.dataset.split("_")[0] + "_regression"](data = args.data, split = args.split, task = "Topological Polar Surface Area", icl = args.icl, use_smiles = args.use_smiles), 
            load_dataset[args.dataset.split("_")[0] + "_regression"](data = args.data, split = args.split, task = "Weight", icl = args.icl, use_smiles = args.use_smiles), 
            load_dataset[args.dataset.split("_")[0] + "_regression"](data = args.data, split = args.split, task = "LogP", icl = args.icl, use_smiles = args.use_smiles), 
            load_dataset[args.dataset.split("_")[0] + "_regression"](data = args.data, split = args.split, task = "Quantitative Estimate of Druglikeness", icl = args.icl, use_smiles = args.use_smiles), 
        ])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True, shuffle=True, collate_fn=collate_fn)

    # Step 2: Build Model
    args.llm_model_path = get_llm_model_path[args.llm_model_name]
    model = load_model[args.model_name](args=args)
    if args.checkpoint_path is not None:
        model = _reload_model(model, args.checkpoint_path)

    # Step 3 Set Optimizer
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': args.lr, 'weight_decay': args.wd}, ],
        betas=(0.9, 0.95)
    )
    trainable_params, all_param = model.print_trainable_params()
    print("-"*len(f"No. Trainable Params: {trainable_params} ({100 * trainable_params / all_param:.4f} %)"))
    print(f"No. Trainable Params: {trainable_params} ({100 * trainable_params / all_param:.4f} %)")

    # Step 4. Training
    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm.tqdm(range(num_training_steps))

    for epoch in range(1, args.num_epochs+1):
        model.train()
        epoch_loss, accum_loss = 0., 0.

        for step, batch in enumerate(train_loader):

            optimizer.zero_grad()
            loss = model(batch)
            if torch.isnan(loss):
                continue
            loss.backward()

            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)

            if (step + 1) % args.grad_steps == 0:
                adjust_learning_rate(optimizer.param_groups[0], args.lr, step / len(train_loader) + epoch, args)

            optimizer.step()
            epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()

            if (step + 1) % args.grad_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                # if args.split == "pretrain":
                wandb.log({'Lr': lr})
                wandb.log({'Loss': accum_loss / args.grad_steps})
                accum_loss = 0.

            progress_bar.update(1)

        print(f"Epoch {epoch}|{args.num_epochs}: Train Loss: {epoch_loss / len(train_loader):.4f}")
        if args.split == "pretrain":
            _save_checkpoint(model, optimizer, epoch, args, is_best=True)
        else:
            if epoch % 10 == 0:
                _save_checkpoint(model, optimizer, epoch, args, is_best=False)

if __name__ == "__main__":
    args = parse_args_llm()
    main(args)