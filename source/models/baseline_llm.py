import contextlib
import torch
from torch.cuda.amp import autocast as autocast
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)

class BaselineLLM(torch.nn.Module):
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

        print(f'Loading {args.llm_model_path}')
        kwargs = {
            "max_memory": {0: '80GiB', 1: '80GiB'},
            "device_map": "auto",
            "revision": "main",
            "trust_remote_code": True,
        }
        if "t5" not in args.llm_model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"], trust_remote_code=kwargs["trust_remote_code"])
            self.tokenizer.pad_token_id = 0
            self.tokenizer.padding_side = 'left'
            model = AutoModelForCausalLM.from_pretrained(
                args.llm_model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                **kwargs
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.llm_model_path,
                # torch_dtype=torch.float16,
                # low_cpu_mem_usage=True,
                **kwargs
            )

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

        if "t5" not in args.llm_model_name:
            self.word_embedding = self.model.model.get_input_embeddings()

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

    def forward(self, samples):
        # encode description, questions and labels
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)

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
            input_ids = questions.input_ids[i] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [self.IGNORE_INDEX] * (inputs_embeds.shape[0]-len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length + batch_attention_mask[i]
            batch_label_input_ids[i] = [self.IGNORE_INDEX] * pad_length+batch_label_input_ids[i]

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
        questions = self.tokenizer(samples["question"], add_special_tokens=False)

        # encode special tokens
        eos_user_tokens = self.tokenizer(self.EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(self.BOS, add_special_tokens=False, return_tensors='pt').input_ids[0])
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            # Add bos & eos token
            input_ids = questions.input_ids[i] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=0)

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