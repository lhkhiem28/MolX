import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoModel
from source.models.gnn.gin import GINGraphCL, GINGraphSTM
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)

from torch.utils.data import Dataset
from torch_geometric.data import Data

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector

def smiles2data(smiles):
    """ used in MoleculeNetGraphDataset() class
    Converts rdkit mol objects to graph data object in pytorch geometric
    NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr """
    mol = Chem.MolFromSmiles(smiles)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = atom_to_feature_vector(atom)
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    if len(mol.GetBonds()) <= 0:  # mol has no bonds
        num_bond_features = 3  # bond type & direction
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    else:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)

            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

class MolXLLM(torch.nn.Module):
    def __init__(
        self,
        args,
        **kwargs
    ):
        super().__init__()
        self.max_new_tokens = args.max_new_tokens
        if "llama" in args.llm_model_name or "mistr" in args.llm_model_name:
            self.BOS = '<s>[INST]'
            self.EOS_USER = '[/INST]'
            self.EOS = '</s>'
            self.IGNORE_INDEX = -100
            if "7b"  in args.llm_model_name:
                llm_emb_dim = 4096
            if "13b" in args.llm_model_name:
                llm_emb_dim = 5120
            if "70b" in args.llm_model_name:
                llm_emb_dim = 8192

        print(f'Loading {args.llm_model_path}')
        kwargs = {
            "max_memory": {0: '80GiB', 1: '80GiB'},
            "device_map": "auto",
            "revision": "main",
        }
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **kwargs
        )

        # smiles encoder
        self.smiles_enc = args.smiles_enc
        if self.smiles_enc == "chemberta":
            config = AutoConfig.from_pretrained("DeepChem/ChemBERTa-10M-MTR")
            self.smiles_tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-10M-MTR")
            if not args.wo_init:
                self.smiles_encoder = AutoModel.from_pretrained("DeepChem/ChemBERTa-10M-MTR")
                print("ChemBERTa Initialization")
            else:
                self.smiles_encoder = AutoModel.from_config(config)
                print("No Initialization")
        if "llama" in args.llm_model_name or "mistr" in args.llm_model_name:
            self.smiles_projector = nn.Sequential(
                nn.Linear(384, 2048),
                nn.ReLU(inplace = True),
                nn.Linear(2048, llm_emb_dim),
            ).to(model.device)
        if args.cl_init:
            self.smiles_encoder.load_state_dict(torch.load(f'output/pretraining/pre-{args.graph_enc}/smiles_encoder_last.pth', map_location=torch.device('cpu')))
            self.smiles_projector.load_state_dict(torch.load(f'output/pretraining/pre-{args.graph_enc}/smiles_projector_last.pth', map_location=torch.device('cpu')))
            print("ChemBERTa Initialization and Pre-CL")
        # graph encoder
        if args.graph_enc == "graphcl":
            self.graph_encoder = GINGraphCL(
                num_layer=5,
                emb_dim=300,
                gnn_type='gin',
                drop_ratio=0.0,
                JK='last',
            ).to(model.device)
        if args.graph_enc == "graphstm":
            self.graph_encoder = GINGraphSTM(
                num_layer=5,
                emb_dim=300,
                gnn_type='gin',
                drop_ratio=0.0,
                JK='last',
            ).to(model.device)
        if "llama" in args.llm_model_name or "mistr" in args.llm_model_name:
            self.graph_projector = nn.Sequential(
                nn.Linear(300, 2048),
                nn.ReLU(inplace = True),
                nn.Linear(2048, llm_emb_dim),
            ).to(model.device)
        if not args.wo_init:
            self.graph_encoder.load_state_dict(
                torch.load(f'source/models/gnn/{args.graph_enc}.pth', map_location=torch.device('cpu')), 
                strict=False, 
            )
            print(f"Chem{args.graph_enc} Initialization")
            if args.cl_init:
                self.graph_encoder.load_state_dict(torch.load(f'output/pretraining/pre-{args.graph_enc}/graph_encoder_last.pth', map_location=torch.device('cpu')))
                self.graph_projector.load_state_dict(torch.load(f'output/pretraining/pre-{args.graph_enc}/graph_projector_last.pth', map_location=torch.device('cpu')))
                print(f"Chem{args.graph_enc} Initialization and Pre-CL")
        else:
            print("No Initialization")
        # fingerprint encoder
        if "llama" in args.llm_model_name or "mistr" in args.llm_model_name:
            self.fingerprint_projector = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.ReLU(inplace = True),
                nn.Linear(2048, llm_emb_dim),
            ).to(model.device)

        self.llm_frozen = args.llm_frozen
        if args.llm_frozen == 'True':
            print(f"{args.llm_model_path} has been frozen!")
            for name, param in model.named_parameters():
                param.requires_grad = False
        else:
            print(f"{args.llm_model_path} has been factorized for training!")
            model = prepare_model_for_int8_training(model)

            lora_r: int = 8
            lora_alpha: int = 32
            lora_dropout: float = 0.1
            lora_target_modules = ['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

        self.model = model
        self.word_embedding = self.model.model.get_input_embeddings()

        self.icl = args.icl
        self.use_smiles = args.use_smiles
        self.xtokens = args.xtokens
        if self.xtokens:
            init_token_ids = self.tokenizer(args.xtokens_init, add_special_tokens=False).input_ids
            init_token_ids = init_token_ids*2
            self.xprompt = torch.nn.Parameter(self.word_embedding.weight[torch.LongTensor(init_token_ids)].detach().clone().to(torch.float32)).to(self.model.device)

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def encode_mol(self, smiles, graph, fingerprint):
        smiles = self.smiles_tokenizer(smiles, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt", max_length=512)
        smiles = self.smiles_encoder(input_ids = smiles.input_ids, attention_mask=smiles.attention_mask)["last_hidden_state"]
        smiles = self.smiles_projector(torch.mean(smiles, dim=1).to(self.model.device))
        graph = self.graph_encoder(graph.to(self.model.device))
        graph = self.graph_projector(graph.to(self.model.device))
        fingerprint = self.fingerprint_projector(torch.tensor(fingerprint).to(self.model.device))
        return torch.mean(torch.cat((
            torch.mean(torch.cat((smiles.unsqueeze(0), graph.unsqueeze(0)), dim=0), dim=0).unsqueeze(0), 
            fingerprint.unsqueeze(0), 
        ), dim=0), dim=0)

    def forward(self, samples):
        # encode description, questions and labels
        questions_1 = self.tokenizer(samples["question_1"], add_special_tokens=False)
        questions_2 = self.tokenizer(samples["question_2"], add_special_tokens=False)
        smiles = self.tokenizer(samples["smiles"], add_special_tokens=False)
        post_questions = self.tokenizer(samples["post_question"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)

        # encode mol
        mol = self.encode_mol(samples["smiles"], samples["graph"], samples["fingerprint"])

        # encode special tokens
        eos_tokens = self.tokenizer(self.EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(self.EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(self.BOS, add_special_tokens=False, return_tensors='pt').input_ids[0])
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):
            # Add bos & eos token
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
            question_embeds_1, question_embeds_2, smiles_embeds, post_question_embeds, eos_user_tokens_embeds, label_input_ids_embeds = self.word_embedding(torch.tensor(questions_1.input_ids[i]).to(self.model.device)), self.word_embedding(torch.tensor(questions_2.input_ids[i]).to(self.model.device)), self.word_embedding(torch.tensor(smiles.input_ids[i]).to(self.model.device)), self.word_embedding(torch.tensor(post_questions.input_ids[i]).to(self.model.device)), self.word_embedding(torch.tensor(eos_user_tokens.input_ids).to(self.model.device)), self.word_embedding(torch.tensor(label_input_ids).to(self.model.device))
            # smiles_embeds = torch.mean(smiles_embeds, dim=0)
            if "<EXAMPLE>" in samples["question_e"][i]:
                question_e = samples["question_e"][i]
                question_embeds_e = self.encode_e(question_e)
                if self.xtokens:
                    inputs_embeds = torch.cat([bos_embeds, question_embeds_1, question_embeds_e, question_embeds_2, mol[i].unsqueeze(0), self.xprompt, post_question_embeds, eos_user_tokens_embeds, label_input_ids_embeds], dim=0)
                else:
                    inputs_embeds = torch.cat([bos_embeds, question_embeds_1, question_embeds_e, question_embeds_2, mol[i].unsqueeze(0), post_question_embeds, eos_user_tokens_embeds, label_input_ids_embeds], dim=0)
            else:
                if self.xtokens:
                    inputs_embeds = torch.cat([bos_embeds, question_embeds_1, question_embeds_2, mol[i].unsqueeze(0), self.xprompt, post_question_embeds, eos_user_tokens_embeds, label_input_ids_embeds], dim=0)
                else:
                    inputs_embeds = torch.cat([bos_embeds, question_embeds_1, question_embeds_2, mol[i].unsqueeze(0), post_question_embeds, eos_user_tokens_embeds, label_input_ids_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [self.IGNORE_INDEX] * (inputs_embeds.shape[0]-len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(len(batch_inputs_embeds)):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length + batch_attention_mask[i]
            batch_label_input_ids[i] = [self.IGNORE_INDEX] * pad_length+batch_label_input_ids[i]

        if self.llm_frozen == 'True':
            inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device).to(torch.float16)
        else:
            inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=label_input_ids,
                return_dict=True,
            )

        return outputs.loss

    def inference(self, samples):
        self.model = self.model.bfloat16()
        # encode description and questions
        questions_1 = self.tokenizer(samples["question_1"], add_special_tokens=False)
        questions_2 = self.tokenizer(samples["question_2"], add_special_tokens=False)
        smiles = self.tokenizer(samples["smiles"], add_special_tokens=False)
        post_questions = self.tokenizer(samples["post_question"], add_special_tokens=False)

        # encode mol
        mol = self.encode_mol(samples["smiles"], samples["graph"], samples["fingerprint"])

        # encode special tokens
        eos_user_tokens = self.tokenizer(self.EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(self.BOS, add_special_tokens=False, return_tensors='pt').input_ids[0])
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            # Add bos & eos token
            question_embeds_1, question_embeds_2, smiles_embeds, post_question_embeds, eos_user_tokens_embeds = self.word_embedding(torch.tensor(questions_1.input_ids[i]).to(self.model.device)), self.word_embedding(torch.tensor(questions_2.input_ids[i]).to(self.model.device)), self.word_embedding(torch.tensor(smiles.input_ids[i]).to(self.model.device)), self.word_embedding(torch.tensor(post_questions.input_ids[i]).to(self.model.device)), self.word_embedding(torch.tensor(eos_user_tokens.input_ids).to(self.model.device))
            # smiles_embeds = torch.mean(smiles_embeds, dim=0)
            if "<EXAMPLE>" in samples["question_e"][i]:
                question_e = samples["question_e"][i]
                question_embeds_e = self.encode_e(question_e)
                if self.xtokens:
                    inputs_embeds = torch.cat([bos_embeds, question_embeds_1, question_embeds_e, question_embeds_2, mol[i].unsqueeze(0), self.xprompt, post_question_embeds, eos_user_tokens_embeds], dim=0)
                else:
                    inputs_embeds = torch.cat([bos_embeds, question_embeds_1, question_embeds_e, question_embeds_2, mol[i].unsqueeze(0), post_question_embeds, eos_user_tokens_embeds], dim=0)
            else:
                if self.xtokens:
                    inputs_embeds = torch.cat([bos_embeds, question_embeds_1, question_embeds_2, mol[i].unsqueeze(0), self.xprompt, post_question_embeds, eos_user_tokens_embeds], dim=0)
                else:
                    inputs_embeds = torch.cat([bos_embeds, question_embeds_1, question_embeds_2, mol[i].unsqueeze(0), post_question_embeds, eos_user_tokens_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device).to(torch.bfloat16)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.max_new_tokens,
                # do_sample=True,
                use_cache=True  # IMPORTANT!
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {'id': samples['id'],
                'pred': [p.strip() for p in pred],
                'label': samples['label'],
                # 'question': samples['question'],
        }

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param