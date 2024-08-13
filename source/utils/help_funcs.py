import random, os
import numpy as np
import torch
from torch_geometric.data import Batch

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def collate_fn(original_batch):
    batch = {}
    for k in original_batch[0].keys():
        batch[k] = [d[k] for d in original_batch]
    if 'graph' in batch:
        batch['graph'] = Batch.from_data_list(batch['graph'])
    return batch

def print_trainable_params(model):
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        num_params = param.numel()

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param

def _save_checkpoint(model, optimizer, cur_epoch, args, is_best=False):
    """
    Save the checkpoint at the current epoch.
    """
    os.makedirs(f'{args.output_dir}/training/{args.dataset}/{args.data}', exist_ok=True)

    param_grad_dic = {
        k: v.requires_grad for (k, v) in model.named_parameters()
    }
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]
    save_obj = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "config": args,
        "epoch": cur_epoch,
    }
    path = f'{args.output_dir}/training/{args.dataset}/{args.data}/{args.model_name}_{args.llm_model_name}_llm_frozen{args.llm_frozen}_{args.graph_enc}_{args.smiles_enc}_{args.num_epochs}epochs_lr{args.lr}_{args.split}_{"best" if is_best else cur_epoch}_{args.run_name}.pth'
    print("Saving checkpoint at epoch {} to {}".format(cur_epoch, path))
    torch.save(save_obj, path)

def _reload_model(model, checkpoint_path, strict=False):
    """
    Load the best checkpoint for evaluation.
    """
    print("Loading checkpoint from {}.".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=strict)

    return model

import math

def adjust_learning_rate(param_group, LR, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    min_lr = 5e-6
    if epoch < args.warmup_epochs:
        lr = LR * epoch / args.warmup_epochs
    else:
        lr = min_lr + (LR - min_lr) * 0.5 * (1.0 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs)))
    param_group["lr"] = lr
    return lr