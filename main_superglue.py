from __future__ import absolute_import, division, print_function
import warnings
warnings.filterwarnings('ignore')

import argparse
import logging
import os
import random
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from torch.utils.data.distributed import DistributedSampler

from copy import deepcopy as cp

from utils_superglue import (convert_examples_to_features,
                        output_modes, processors)
from models import FineTunedModel, ActionPredictor, FineTunedTransformerModel
from trainer_superglue import trainer

import transformers

transformers.logging.set_verbosity_error()

from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

import wandb

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}

all_task_names = ['RTE','BOOLQ','CB','COPA','WIC','WSC']


class SimpleDataset(Dataset):
    def __init__(self, x1, x2, x3, x4):
        self.__iter = None
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, key):
        val = self.x1[key], self.x2[key], self.x3[key], self.x4[key]
        return val
    
class SimpleDataset2(Dataset):
    def __init__(self, x1, x2, x3):
        self.__iter = None
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, key):
        val = self.x1[key], self.x2[key], self.x3[key]
        return val
    

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def load_and_cache_examples(args, task, tokenizer, evaluate=False, held=False, test=False):
    processor = processors[task.lower()]()
    output_mode = output_modes[task.lower()]
    if task.lower() == 'mnli-mm':
        task = 'MNLI'
        data_dir = os.path.join(args.data_dir,task)
    else:
        data_dir = os.path.join(args.data_dir,task)

    logger.info("Creating features from dataset file at %s", data_dir)
    label_list = processor.get_labels()
    if evaluate:
        if test:
            examples = processor.get_test_examples(data_dir)
        else:
            if held:
                examples = processor.get_held_examples(data_dir)
            else:
                examples = processor.get_dev_examples(data_dir)
    else:
        examples = processor.get_train_examples(data_dir)
    features = convert_examples_to_features(examples, label_list, args.max_seq_length,
                                            tokenizer, output_mode,
                                            cls_token_at_end=False,
                                            cls_token=tokenizer.cls_token,
                                            sep_token=tokenizer.sep_token,
                                            cls_token_segment_id=1,
                                            pad_on_left=False,
                                            pad_token_segment_id=0)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    if test == False:
        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

        dataset = SimpleDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    else:
        dataset = SimpleDataset2(all_input_ids, all_input_mask, all_segment_ids)

    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task", default='RTE', type=str,
                        help="Model type selected in the list: " + ", ".join(all_task_names))
    
    parser.add_argument("--data_dir", default='./superglue_data/', type=str, required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    
    parser.add_argument("--model_type", default='bert', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--teacher_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--student_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument('--teacher_tf_checkpoint', default=None, type=str,
                        help='Teacher TF Checkpoint')
    parser.add_argument('--student_tf_checkpoint', default=None, type=str,
                        help='Teacher TF Checkpoint')
    parser.add_argument("--not_train_teacher", action='store_true',
                        help="Whether not to train teacher.")
    parser.add_argument('--teacher_finetuned_checkpoint', default=None, type=str,
                        help='Teacher finetuned Checkpoint')
    
    parser.add_argument("--nlayers", default=6, type=int,
                        help="Number of encoder layers, in case the student model is not pretrained")
    parser.add_argument("--emsize", default=768, type=int,
                        help="Hidden size of embedding layer, in case the student model is not pretrained")
    parser.add_argument("--nhid", default=768, type=int,
                        help="Hidden size of encoder, in case the student model is not pretrained")
    parser.add_argument("--pooling_method", default="cls", type=str,
                        help="Pooling method for Transformer student model")
    parser.add_argument("--encoder_version", default="post", type=str,
                        help="Encoder version for Transformer, either post or ReZero")

    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--log_dir", default='logs', type=str, help="The log data dir.")
    parser.add_argument("--output_dir", default='tmp/', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    
    parser.add_argument("--temperature", default=5.0, type=float,
                        help="Distillation temperature for soft target.")
    parser.add_argument("--max_seq_length", default=128, type=int,   ## initially 128
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--alpha", default=0.5, type=float,
                        help="Train loss ratio.")
    parser.add_argument("--lambda_", default=0.5, type=float,
                        help="Meta Train loss ratio.")
    parser.add_argument("--beta", default=100.0, type=float,
                        help="Distillation loss ratio.")
    
    parser.add_argument("--teacher_learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--meta_learning_rate", default=0.001, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--train_dataloader_size", default=8, type=int,  ## 8 initial
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--val_dataloader_size", default=8, type=int,    ## 8 initial
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--teacher_epochs", default=10, type=int,
                        help="Number of epochs.")
    parser.add_argument("--epochs", default=10, type=int,                
                        help="Number of epochs.")
    parser.add_argument("--num_episodes", default=200, type=int,
                        help="Number of steps for meta update.")
    
    parser.add_argument("--max_grad_norm", default=1, type=float,
                        help="Max gradient norm.")
    
    parser.add_argument("--use_comp_loss", action='store_true',
                        help="Use competitive loss or not.")
    
    parser.add_argument("--reward_function", default='binary', type=str,
                        help="Type of reward function - binary or real.")
    
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument('--wandb_logging', action='store_true',
                        help='wandb logging needed')
    parser.add_argument('--wandb_project_name', type=str, default='Superglue V4', required=False,
                        help='wandb project name')
    
    args = parser.parse_args()

    args.teacher_save_path = os.path.join("output", args.task, "pytorch_model.bin")

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.teacher_model, do_lower_case=args.do_lower_case)

    args.n_gpu = 1
    
    # Set seed
    set_seed(args)

    task_names = ['RTE','BOOLQ','CB','COPA','WIC','WSC']
    now = int(datetime.now().timestamp())
    args.timestamp = now

    task = args.task

    task_loaders = {}
    label_nums = {}

    for task_name in task_names:
        processor = processors[task_name.lower()]()
        output_mode = output_modes[task_name.lower()]
        label_list = processor.get_labels()
        num_labels = len(label_list)
        print ("Loaded %s dataset" % (task_name))
        
        if task_name != args.task:
            held_dataset = load_and_cache_examples(args, task_name, tokenizer, evaluate=True, held=True)
            held_sampler = RandomSampler(held_dataset) if args.local_rank == -1 else DistributedSampler(held_dataset)
            held_loader = DataLoader(held_dataset, sampler=held_sampler,batch_size=args.val_dataloader_size)

            task_loaders[task_name] = {'held': {'loader': held_loader, 'dataset': held_dataset},
                                    'num_labels': num_labels}
        
        else:
            train_dataset = load_and_cache_examples(args, task_name, tokenizer, evaluate=False)
            held_dataset = load_and_cache_examples(args, task_name, tokenizer, evaluate=True, held=True)
            eval_dataset = load_and_cache_examples(args, task_name, tokenizer, evaluate=True)
            test_dataset = load_and_cache_examples(args, task_name, tokenizer, evaluate=True, test=True)
            
            print ("Number of test samples - {}".format(len(test_dataset)))

            train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
            held_sampler = RandomSampler(held_dataset) if args.local_rank == -1 else DistributedSampler(held_dataset)
            eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
            train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_dataloader_size)
            held_loader = DataLoader(held_dataset, sampler=held_sampler, batch_size=args.val_dataloader_size)
            test_loader = DataLoader(test_dataset, batch_size=args.val_dataloader_size, shuffle=False)
            eval_loader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.val_dataloader_size)
            task_loaders[task_name] = {'train': {'loader': train_loader, 'dataset': train_dataset}, \
                                        'held': {'loader': held_loader, 'dataset': held_dataset}, \
                                        'eval': {'loader': eval_loader, 'dataset': eval_dataset}, \
                                        'test': {'loader': test_loader, 'dataset': test_dataset}, \
                                        'num_labels': num_labels}
        label_nums[task_name.lower()] = num_labels

        #args.num_meta_steps += int(len(held_dataset)*args.meta_fraction//args.val_dataloader_size)

    t_config = config_class.from_pretrained(args.teacher_model,
                                          num_labels=label_nums[args.task.lower()], finetuning_task=args.task)
    s_config = config_class.from_pretrained(args.teacher_model,
                                          num_labels=label_nums[args.task.lower()], finetuning_task=args.task)
    
    s_config.num_hidden_layers = 6

    teacher_model = FineTunedModel(task_names, label_nums, t_config, pretrained_model_name=args.teacher_model, \
                                   tf_checkpoint=args.teacher_tf_checkpoint)
    
    try:
        teacher_model.load_state_dict(torch.load("./models_new/teacher_{}_{}.ckpt".format(args.task, args.seed)))
        args.teacher_epochs = 0
        print ("Loaded teacher from previous checkpoint")
    except:
        pass

    if args.not_train_teacher == True:
        args.teacher_epochs = 0

    if args.student_model != 'transformer':
        student_model = FineTunedModel(task_names, label_nums, s_config, pretrained_model_name=args.teacher_model, \
                                   tf_checkpoint=args.student_tf_checkpoint)
    
    else:
        student_model = FineTunedTransformerModel(ntokens=t_config.vocab_size, emsize=args.emsize,\
                                               nhid=args.nhid, nlayers=args.nlayers, tasks=task_names, \
                                               label_nums=label_nums, pooling_method=args.pooling_method, \
                                               encoder_version=args.encoder_version, nhead=8, dropout=0.1)
        
    action_model = ActionPredictor(d_model=768, num_actions=len(task_names))

    print ("Number of parameters for student {}".format(sum(p.numel() for p in student_model.parameters() if p.requires_grad)))
    print ("Number of parameters for teacher {}".format(sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)))
    print ("Number of parameters for action model {}".format(sum(p.numel() for p in action_model.parameters() if p.requires_grad)))
    
    if args.wandb_logging == True:
        config = vars(args)
        wandb.login()
        wandb.init(project=args.wandb_project_name,config=config)

    trajectories, all_rewards = trainer(args, teacher_model, student_model, action_model, task_loaders, label_nums, task)

if __name__ == "__main__":
    main()

