import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from torch.distributions import Categorical

import random
from collections import Counter
from copy import deepcopy as cp

import wandb

def simple_accuracy(preds, labels):
    return accuracy_score(preds, labels)

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average="macro")  ## for cb task  average = "weighted"
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }
def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i] 
        for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name in ["boolq", "copa", "rte", "wic", "wsc"]:
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "cb":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)

def evaluate_model(task_loaders, task_name, model, label_nums, device, split='eval'):    
    assert split in ['eval', 'test']

    model.eval()   

    if task_name.lower() in ['sts-b']:
        loss_fn = nn.MSELoss(reduction='mean') #F.mse_loss(labels.view(-1), student_out.view(-1))
    else:
        loss_fn = nn.CrossEntropyLoss()
    
    task_loader = task_loaders[task_name]
    val_dataloader = task_loader[split]['loader']   
    val_data = task_loader[split]['dataset']  
    num_labels = label_nums[task_name.lower()]

    all_labels = []
    all_preds = []

    total_val_loss = 0
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            if split == 'eval':
                input_ids, attention_mask, token_type_ids, labels = batch[0], batch[1], batch[2], batch[3]
                labels = labels.to(device)
            else:
                input_ids, attention_mask, token_type_ids = batch[0], batch[1], batch[2]

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)

            out = model(src=input_ids, task_name=task_name.lower(), mask=attention_mask, token_type_ids=token_type_ids)
            
            student_out = out[0]

            if task_name.lower() != 'sts-b':
                student_out_ = student_out.argmax(-1)
            else:
                student_out_ = student_out[:,0].clip(0,5)

            if task_name.lower() != 'sts-b':
                all_preds += student_out_.cpu().numpy().astype(int).tolist()
                if split == 'eval':
                    all_labels += labels.cpu().numpy().astype(int).tolist()
            else:
                all_preds += student_out_.cpu().numpy().tolist()
                if split == 'eval':
                    all_labels += labels.cpu().numpy().tolist()

            if split == 'eval':
                if task_name.lower() in ['sts-b']:
                    loss = loss_fn(student_out.view(-1), labels.view(-1)) 
                else:
                    loss = loss_fn(student_out.view(-1, num_labels), labels.view(-1)) 

                total_val_loss += loss.item() * labels.shape[0]

    total_val_loss = total_val_loss/len(val_data)

    if split == 'eval':
        out = compute_metrics(task_name.lower(), all_preds, all_labels)
    else:
        out = {}

    out['task'] = task_name
    if split == 'eval':
        out['val_loss'] = total_val_loss
    
    return all_preds, out


#### TRAINER METHOD #####
def trainer(args, teacher_model, student_model, action_model, task_loaders, label_nums, task_name):
    rewards = []
    trajectories = []
    reward_tensor = None

    best_score = 0

    task_loader = task_loaders[task_name]
    train_dataloader = task_loader['train']['loader']
    train_data = task_loader['train']['dataset']
    
    if task_name.lower() in ['sts-b']:
        loss_fn = nn.MSELoss(reduction='mean') 
    else:
        loss_fn = nn.CrossEntropyLoss()

    all_tasks = list(task_loaders.keys())

    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    
        
    teacher_model.to(device)
    student_model.to(device)
    action_model.to(device)

    t_total = args.train_dataloader_size * args.epochs

    t_optimizer = torch.optim.AdamW(teacher_model.parameters(), lr=args.teacher_learning_rate)
    s_optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.learning_rate)
    s_optimizer2 = torch.optim.AdamW(action_model.parameters(), lr=args.meta_learning_rate)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        teacher_model, t_optimizer = amp.initialize(teacher_model, t_optimizer, opt_level=args.fp16_opt_level)
        student_model, s_optimizer = amp.initialize(student_model, s_optimizer, opt_level=args.fp16_opt_level)
        action_model, s_optimizer2 = amp.initialize(action_model, s_optimizer2, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        teacher_model = torch.nn.DataParallel(teacher_model)
        student_model = torch.nn.DataParallel(student_model)
        action_model = torch.nn.DataParallel(action_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
        student_model = torch.nn.parallel.DistributedDataParallel(student_model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
        action_model = torch.nn.parallel.DistributedDataParallel(action_model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
        
    ###################################
    ###### TEACHER FINE-TUNING ########
    ###################################

    print ("Fine-tuning teacher model on {}".format(task_name))


    num_labels = label_nums[task_name.lower()]

    all_held_batches = {}
    for action in task_loaders:
        all_held_batches[action] = {}
        for id, batch in enumerate(task_loaders[action]['held']['loader']):
            all_held_batches[action][id] = batch

    for epoch in range(args.epochs):

        if epoch < args.teacher_epochs:
            t_optimizer.zero_grad()
            total_training_task_loss = 0

            # Fine tune teacher first
            for step, batch in enumerate(train_dataloader):

                input_ids, attention_mask, token_type_ids, labels = batch[0], batch[1], batch[2], batch[3]
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)
                labels = labels.to(device)

                out = teacher_model.forward(src=input_ids, task_name=task_name.lower(), mask=attention_mask, token_type_ids=token_type_ids)
                teacher_out = out[0]

                if task_name.lower() in ['sts-b']:
                    loss = loss_fn(teacher_out.view(-1), labels.view(-1)) 
                else:
                    loss = loss_fn(teacher_out.view(-1, num_labels), labels.view(-1)) 
                
                if args.fp16:
                    with amp.scale_loss(loss, t_optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(t_optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), args.max_grad_norm)

                total_training_task_loss += loss.item() * labels.shape[0]

                t_optimizer.step()
                t_optimizer.zero_grad()

            total_training_task_loss = total_training_task_loss/len(train_data)
            print('| task {} | epoch {:3d} | training loss {:5.3f}'.format(
                    task_name, epoch, total_training_task_loss))
            
            preds, out = evaluate_model(task_loaders, task_name, teacher_model, label_nums, device, split='eval')
            print ("Epoch {}, Teacher Result: {}".format(epoch, out))

            if 'mcc' in out:
                score = out['mcc']
            elif 'f1' in out:
                score = out['f1']
            elif 'acc' in out:
                score = out['acc']
            elif 'spearmanr' in out:
                score = out['spearmanr']

            models_dir = "./models_new"
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)

            if score > best_score:
                torch.save(teacher_model.state_dict(), "./models_new/teacher_{}_{}.ckpt".format(args.task, args.seed))
                best_score = score

            if args.wandb_logging == True:
                for key in out:
                    if key != 'task':
                        wandb.log({"Teacher " + key: out[key]})

    teacher_model.load_state_dict(torch.load("./models_new/teacher_{}_{}.ckpt".format(args.task, args.seed)))

    preds, out = evaluate_model(task_loaders, task_name, teacher_model, label_nums, device, split='eval')
    print ("Teacher Final Result: {}".format(out))

    if args.wandb_logging == True:
        for key in out:
            if key != 'task':
                wandb.log({"Teacher Final " + key: out[key]})

    # Test prediction
    all_preds, _ = evaluate_model(task_loaders, task_name, teacher_model, label_nums, device, split='test')
    test_pred = pd.DataFrame({'index': range(len(all_preds)), 'prediction': all_preds})

    if task_name.lower() in ['rte']:
        mapping = ["entailment", "not_entailment"]
        test_pred['prediction'] = test_pred['prediction'].apply(lambda x: mapping[x])
    elif task_name.lower() in ['cb']:
        mapping = ["entailment","contradiction", "neutral"]
        test_pred['prediction'] = test_pred['prediction'].apply(lambda x: mapping[x])

    try:
        os.makedirs("test_outputs_sg_new")
    except:
        pass

    test_pred.to_csv("./test_outputs_sg_new/"+"teacher_" + task_name + '.tsv', sep='\t', index=False)
    
    ### CLONING THE TEACHER TO META TEACHER ###
    teacher_model2 = cp(teacher_model)

    if args.n_gpu > 1:
        for param in teacher_model2.module.encoder.parameters():
            param.requires_grad = False
    else:
        for param in teacher_model2.encoder.parameters():
            param.requires_grad = False


    t_optimizer2 = torch.optim.AdamW(teacher_model2.parameters(), lr=args.meta_learning_rate)
 
    student_model2 = cp(student_model)

    s_optimizer3 = torch.optim.AdamW(student_model2.parameters(), lr=args.learning_rate)
    
    best_score = 0
    best_score2 = 0
    best_score3 = 0

    loss_type = 'comp' if args.use_comp_loss else 'col'

    reward_df = pd.DataFrame()
    reward_df['Episode'] = np.arange(args.num_episodes)


    ###################################
    ###### STUDENT DISTILLATION #######
    ###################################

    print ("Fine-tuning student model on {}".format(task_name))
    print ("Updating teacher model quiz dataset")


    for epoch in range(args.epochs):  
        num_labels = label_nums[task_name.lower()]
        total_training_loss = 0
        total_training_task_loss = 0

        s_optimizer.zero_grad()

        for step, batch in enumerate(train_dataloader):

            input_ids, attention_mask, token_type_ids, labels = batch[0], batch[1], batch[2], batch[3]
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)

            out = student_model.forward(src=input_ids, task_name=task_name.lower(), mask=attention_mask, token_type_ids=token_type_ids)
            student_out, student_pooler = out[0], out[1]

            with torch.no_grad():
                out = teacher_model2.forward(src=input_ids, task_name=task_name.lower(), mask=attention_mask, token_type_ids=token_type_ids)
                teacher_out = out[0]
                teacher_pooler = out[1][:student_pooler.shape[0]]

            if task_name.lower() in ['sts-b']:
                loss = loss_fn(student_out.view(-1), labels.view(-1)) 
            else:
                loss = loss_fn(student_out.view(-1, num_labels), labels.view(-1)) 
            
            if task_name.lower() in ['sts-b']:
                soft_loss = F.mse_loss(teacher_out, student_out)
            else:
                T = args.temperature
                soft_targets = F.softmax(teacher_out / T, dim=-1)

                probs = F.softmax(student_out / T, dim=-1)
                soft_loss = F.mse_loss(soft_targets, probs) * T * T

            if args.beta == 0: 
                pkd_loss = torch.zeros_like(soft_loss)
                
            else:
                t_features = teacher_pooler / teacher_pooler.norm(dim=-1).unsqueeze(-1)
                s_features = student_pooler / student_pooler.norm(dim=-1).unsqueeze(-1)
                pkd_loss = F.mse_loss(s_features, t_features, reduction="mean")

            total_loss = args.alpha * soft_loss + (
                    1 - args.alpha) * loss + args.beta * pkd_loss
            
            if args.fp16:
                with amp.scale_loss(total_loss, s_optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(s_optimizer), args.max_grad_norm)
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)

            s_optimizer.step()
            s_optimizer.zero_grad()

            total_training_task_loss += loss.item() * labels.shape[0]
            total_training_loss += total_loss.item() * labels.shape[0]

        total_training_loss = total_training_loss/len(train_data)
        total_training_task_loss = total_training_task_loss/len(train_data)

        print('| task {} | epoch {:3d} | training loss {:5.3f} | training task loss {:5.3f}'.format(
                    task_name, epoch, total_training_loss, total_training_task_loss))

        if args.wandb_logging == True:
            wandb.log({"Training loss": total_training_loss})
            wandb.log({"Training task only loss": total_training_task_loss})

        if args.local_rank in [-1, 0]:
            preds, out3 = evaluate_model(task_loaders, task_name, student_model, label_nums, device, split='eval')
            print ("Epoch {}, Student Result before Meta Step: {}".format(epoch, out3))

            if args.wandb_logging == True:
                for key in out3:
                    if key != 'task':
                        wandb.log({"Student PKD " + key: out3[key]})

            if 'mcc' in out3:
                score = out3['mcc']
            elif 'f1' in out3:
                score = out3['f1']
            elif 'acc' in out3:
                score = out3['acc']
            elif 'spearmanr' in out3:
                score = out3['spearmanr']

            if score > best_score:
                torch.save(student_model.state_dict(), "./models_new/student_pkd_{}_{}.ckpt".format(args.task, args.seed))
                best_score = score

        

        #######################################
        ######## TEACHER META LEARNING ########
        #######################################

        total_teacher_meta_loss = 0
        total_student_meta_loss = 0
        action = task_name
        task_loader = task_loaders[action]
        held_dataloader = task_loader['held']['loader']

        if args.local_rank in [-1, 0]: 
            for batch in held_dataloader:
                input_ids, attention_mask, token_type_ids, labels = batch[0], batch[1], batch[2], batch[3]
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)

                t_optimizer2.zero_grad()
                
                ## extracting the pooled output from teacher and student
                ## and feeding it to the feed forward meta teacher

                with torch.no_grad():
                    _, _, _, teacher_pooler_output = teacher_model(src=input_ids, \
                                                                task_name=action.lower(),mask=attention_mask, token_type_ids=token_type_ids)
                    
                    _, _, _, student_pooler_output = student_model(src=input_ids, \
                                                                task_name=action.lower(),mask=attention_mask, token_type_ids=token_type_ids)

                out = teacher_model2.forward(pooled_output=teacher_pooler_output, task_name=action.lower(), discriminator=True)
                teacher_out = out[0]

                out = teacher_model2.forward(pooled_output=student_pooler_output, task_name=action.lower(), discriminator=True)
                teacher_out2 = out[0]

                if args.use_comp_loss == False:
                    if action.lower() in ['sts-b']:
                        teacher_loss = 0.5*nn.MSELoss()(teacher_out.view(-1), labels.view(-1)) + 0.5*nn.MSELoss()(teacher_out2.view(-1), labels.view(-1))  #F.mse_loss(labels.view(-1), student_out.view(-1))
                    else:
                        teacher_loss = 0.5*nn.CrossEntropyLoss()(teacher_out.view(-1, num_labels), labels.view(-1)) + 0.5*nn.CrossEntropyLoss()(teacher_out2.view(-1, num_labels), labels.view(-1)) #F.cross_entropy(student_out.view(-1, num_labels), labels.view(-1))
                else:
                    if action.lower() in ['sts-b']:
                        teacher_loss2 = nn.MSELoss()(teacher_out.view(-1), labels.view(-1)) #+ nn.MSELoss()(teacher_out2.view(-1), labels.view(-1))  #F.mse_loss(labels.view(-1), student_out.view(-1))
                    else:
                        teacher_loss2 = nn.CrossEntropyLoss()(teacher_out.view(-1, num_labels), labels.view(-1)) #+ nn.CrossEntropyLoss()(teacher_out2.view(-1, num_labels), labels.view(-1)) #F.cross_entropy(student_out.view(-1, num_labels), labels.view(-1))

                    if action.lower() not in ['sts-b']:
                        teacher_out = nn.Softmax(-1)(teacher_out).gather(dim=1,index=labels.long().view(-1,1)).squeeze()
                        teacher_out2 = nn.Softmax(-1)(teacher_out2).gather(dim=1,index=labels.long().view(-1,1)).squeeze()                
                    else:
                        teacher_out = -1*torch.abs(labels - teacher_out)  
                        teacher_out2 = -1*torch.abs(labels - teacher_out2)   

                    teacher_loss = -torch.mean(teacher_out) + torch.mean(teacher_out2) + teacher_loss2

                teacher_loss.backward()
                torch.nn.utils.clip_grad_norm_(teacher_model2.parameters(), args.max_grad_norm)

                total_teacher_meta_loss += teacher_loss.item()

                t_optimizer2.step()
                t_optimizer2.zero_grad()
                
                total_teacher_meta_loss = total_teacher_meta_loss/len(held_dataloader)

            print('| epoch {:3d} | teacher meta loss {:5.3f}'.format(
                        epoch, total_teacher_meta_loss))
            
            preds, out = evaluate_model(task_loaders, task_name, teacher_model2, label_nums, device, split='eval')
            print ("Epoch {}, Teacher Meta Result: {}".format(epoch, out))

            if args.wandb_logging == True:
                for key in out:
                    if key != 'task':
                        wandb.log({"Teacher Meta " + key: out[key]})
                wandb.log({"Teacher Meta loss": total_teacher_meta_loss})

            if 'mcc' in out:
                score = out['mcc']
            elif 'f1' in out:
                score = out['f1']
            elif 'acc' in out:
                score = out['acc']
            elif 'spearmanr' in out:
                score = out['spearmanr']

            if score > best_score2:
                torch.save(teacher_model2.state_dict(), "./models_new/teacher_meta_{}_{}_{}.ckpt".format(loss_type, args.task, args.seed))
                best_score2 = score

    student_model.load_state_dict(torch.load("./models_new/student_pkd_{}_{}.ckpt".format(args.task, args.seed)))
    teacher_model2.load_state_dict(torch.load("./models_new/teacher_meta_{}_{}_{}.ckpt".format(loss_type,args.task,args.seed)))
    student_model2.load_state_dict(torch.load("./models_new/student_pkd_{}_{}.ckpt".format(args.task, args.seed)))

    print ("Fine-tuning student model on quiz data for increasing reward")
    

    ################################################
    ########### STUDENT REWARD LEARNING ############
    ################################################

    all_rewards = []

    if args.local_rank in [-1, 0]: 
        for episode in range(args.num_episodes):
            trajectory = []
            batch_rewards = []
            batch_states = []
            batch_actions = []
            held_dataloader_main = task_loaders[task_name]['held']['loader']
            held_dataset_main = task_loaders[task_name]['held']['dataset']

            for step, batch_main in enumerate(held_dataloader_main):
                if step == 0:
                    action = random.choice(list(task_loaders.keys()))
                    trajectory.append(action)

                idx = random.choice(np.arange(len(all_held_batches[action])))

                batch  = all_held_batches[action][idx] 
                input_ids, attention_mask, token_type_ids, labels = batch[0], batch[1], batch[2], batch[3]

                input_ids = input_ids.to(device)
                labels = labels.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)

                student_out, _, state_space, student_pooler_output2 = student_model2.forward(src=input_ids, \
                                                                                task_name=action.lower(),mask=attention_mask, token_type_ids=token_type_ids)

                num_labels = label_nums[action.lower()]

                with torch.no_grad():
                    teacher_out2, _, _, _ = teacher_model2.forward(pooled_output=student_pooler_output2, \
                                                    task_name=action.lower(), discriminator=True)

                if action.lower() in ['sts-b']:
                    loss = nn.MSELoss()(student_out.view(-1), labels.view(-1)) 
                else:
                    loss = nn.CrossEntropyLoss()(student_out.view(-1, num_labels), labels.view(-1))

                if action.lower() not in ['sts-b']:
                    student_out = nn.Softmax(-1)(student_out).gather(dim=1,index=labels.long().view(-1,1)).squeeze()
                    teacher_out2 = nn.Softmax(-1)(teacher_out2).gather(dim=1,index=labels.long().view(-1,1)).squeeze()                
                else:
                    student_out = -1*torch.abs(labels - student_out)  
                    teacher_out2 = -1*torch.abs(labels - teacher_out2)    
                

                loss = loss - torch.mean(student_out) + torch.mean(teacher_out2)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model2.parameters(), args.max_grad_norm)
                
                 
                actions = action_model(state_space)[0]
                action_ = Categorical(actions).sample()  
                action = all_tasks[action_.cpu().numpy()] 

                prev_action = trajectory[-1]
                
                input_ids_main, attention_mask_main, token_type_ids_main, labels_main = batch_main[0], batch_main[1], batch_main[2], batch_main[3]
                input_ids_main = input_ids_main.to(device)
                labels_main = labels_main.to(device)
                attention_mask_main = attention_mask_main.to(device)
                token_type_ids_main = token_type_ids_main.to(device)

                
                with torch.no_grad():
                    student_out3, _, _, _ = student_model2.forward(src=input_ids_main, \
                                                    task_name=task_name.lower(),mask=attention_mask_main, token_type_ids=token_type_ids_main)

                    teacher_out, _, _, teacher_pooler_output = teacher_model2(src=input_ids_main, \
                                                                task_name=task_name.lower(),mask=attention_mask_main, token_type_ids=token_type_ids_main)
                    
                    #out = teacher_model2.forward(pooled_output=teacher_pooler_output, task_name=task_name.lower(), discriminator=True)
                    #teacher_out = out[0]

                if task_name.lower() not in ['sts-b']:
                    teacher_out = nn.Softmax(-1)(teacher_out)
                    student_out3 = nn.Softmax(-1)(student_out3)
                    teacher_out = teacher_out.gather(dim=1,index=labels_main.long().view(-1,1)).squeeze()
                    student_out3 = student_out3.gather(dim=1,index=labels_main.long().view(-1,1)).squeeze()                

                    if args.reward_function == 'real':
                        reward = (student_out3 - teacher_out).float().sum()
                    elif args.reward_function == 'binary':
                        reward = (student_out3 > teacher_out).float().sum()
                    else:
                        raise ValueError("reward function should be either 'binary' or 'real'")
                else:
                    teacher_out = teacher_out[:,0]
                    student_out3 = student_out3[:,0]

                    if args.reward_function == 'real':
                        reward = (torch.abs(labels_main - teacher_out) - torch.abs(labels_main - student_out3)).float().sum()
                    elif args.reward_function == 'binary':
                        reward = (torch.abs(labels_main - teacher_out) > torch.abs(labels_main - student_out3)).float().sum()
                    else:
                        raise ValueError("reward function should be either 'binary' or 'real'")
                    
                #print (input_ids_main, reward)
                batch_rewards.append(reward)
                
                batch_states.append(state_space)
                batch_actions.append(action_.unsqueeze(0))

                trajectory.append(action)

            batch_states = torch.cat(batch_states,0)
            total_rewards = torch.sum(torch.FloatTensor(batch_rewards)).item()/len(held_dataset_main)
            all_rewards.append(total_rewards)
            reward_tensor = torch.FloatTensor(discount_rewards(torch.FloatTensor(batch_rewards)/len(held_dataset_main))).to(batch_states.device)
            #reward_tensor = torch.FloatTensor(batch_rewards).to(batch_states.device)
            action_tensor = torch.cat(batch_actions,0).unsqueeze(1)
            logprob = torch.log(
                action_model.forward(batch_states))
            

            selected_logprobs = reward_tensor * torch.gather(logprob, 1, action_tensor).squeeze()
            loss = -selected_logprobs.mean()

            if args.fp16:
                with amp.scale_loss(loss, s_optimizer2) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(s_optimizer2), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(action_model.parameters(), args.max_grad_norm)

            s_optimizer2.step()
            s_optimizer3.step()

            s_optimizer3.zero_grad()
            s_optimizer2.zero_grad()

            total_student_meta_loss = loss.item()
        
            if args.wandb_logging == True:
                wandb.log({"Student Meta loss": total_student_meta_loss})
                wandb.log({"Student Reward": total_rewards})

            preds, out2 = evaluate_model(task_loaders, task_name, student_model2, label_nums, device, split='eval')
            #print ("Student Result: {}, Reward: {}".format(out2, total_rewards))

            if args.wandb_logging == True:
                wandb.log({"Trajectory": trajectory})

            if args.wandb_logging == True:
                for key in out2:
                    if key != 'task':
                        wandb.log({"Student " + key: out2[key]})

            if 'mcc' in out2:
                score = out2['mcc']
            elif 'f1' in out2:
                score = out2['f1']
            elif 'acc' in out2:
                score = out2['acc']
            elif 'spearmanr' in out2:
                score = out2['spearmanr']

            if score > best_score3:
                torch.save(student_model2.state_dict(), "./models_new/best_model_{}.ckpt".format(args.timestamp))
                best_score3 = score

                print (Counter(trajectory))
                print ("The full trajectory is - {}".format(" -> ".join(trajectory)))
                trajectories.append(trajectory)

        #reward_df['Epoch {}'.format(epoch+1)] = all_rewards

        #if epoch == args.epochs-1:
        plt.plot(np.arange(args.num_episodes), all_rewards)
        plt.savefig("rewards_{}.png".format(task_name), dpi=300, bbox_inches='tight')
        wandb.log({"reward_graph": wandb.Image("rewards_{}.png".format(task_name))})
        
        # Validation

        preds, out = evaluate_model(task_loaders, task_name, student_model, label_nums, device, split='eval')
        print ("Student PKD Final Result: {}".format(out))
        
        if args.wandb_logging == True:
            for key in out:
                if key != 'task':
                    wandb.log({"Student PKD Final " + key: out[key]})

        preds, out = evaluate_model(task_loaders, task_name, teacher_model2, label_nums, device, split='eval')
        print ("Teacher Meta Final Result: {}".format(out))
        
        if args.wandb_logging == True:
            for key in out:
                if key != 'task':
                    wandb.log({"Teacher Meta Final " + key: out[key]})

        # Test prediction
        all_preds, _ = evaluate_model(task_loaders, task_name, student_model, label_nums, device, split='test')
        test_pred = pd.DataFrame({'index': range(len(all_preds)), 'prediction': all_preds})


        if task_name.lower() in ['rte']:
            mapping = ["entailment", "not_entailment"]
            test_pred['prediction'] = test_pred['prediction'].apply(lambda x: mapping[x])
        elif task_name.lower() in ['cb']:
            mapping = ["entailment","contradiction", "neutral"]
            test_pred['prediction'] = test_pred['prediction'].apply(lambda x: mapping[x])

        try:
            os.makedirs("test_outputs_sg_new")
        except:
            pass

        test_pred.to_csv("./test_outputs_sg_new/"+"student_PKD_" + task_name + '.tsv', sep='\t', index=False)

        # Test prediction
        all_preds, _ = evaluate_model(task_loaders, task_name, teacher_model2, label_nums, device, split='test')
        test_pred = pd.DataFrame({'index': range(len(all_preds)), 'prediction': all_preds})

        if task_name.lower() in ['rte']:
            mapping = ["entailment", "not_entailment"]
            test_pred['prediction'] = test_pred['prediction'].apply(lambda x: mapping[x])
        elif task_name.lower() in ['cb']:
            mapping = ["entailment","contradiction",  "neutral"]
            test_pred['prediction'] = test_pred['prediction'].apply(lambda x: mapping[x])

        try:
            os.makedirs("test_outputs_sg_new")
        except:
            pass

        test_pred.to_csv("./test_outputs_sg_new/"+"teacher_meta_" + task_name + "_" + loss_type + '.tsv', sep='\t', index=False)

        student_model.load_state_dict(torch.load("./models_new/best_model_{}.ckpt".format(args.timestamp)))

        preds, out2 = evaluate_model(task_loaders, task_name, student_model, label_nums, device, split='eval')
        print ("Student Final Result: {}".format(out2))

        if args.wandb_logging == True:
            for key in out2:
                if key != 'task':
                    wandb.log({"Student Final " + key: out2[key]})

        # Test prediction
        all_preds, _ = evaluate_model(task_loaders, task_name, student_model, label_nums, device, split='test')
        test_pred = pd.DataFrame({'index': range(len(all_preds)), 'prediction': all_preds})

        if task_name.lower() in ['rte']:
            mapping = ["entailment", "not_entailment"]
            test_pred['prediction'] = test_pred['prediction'].apply(lambda x: mapping[x])
        elif task_name.lower() in ['cb']:
            mapping = ["entailment","contradiction", "neutral"]
            test_pred['prediction'] = test_pred['prediction'].apply(lambda x: mapping[x])

        try:
            os.makedirs("test_outputs_sg_new")
        except:
            pass

        test_pred.to_csv("./test_outputs_sg_new/"+task_name+"_" + str(args.timestamp) + '.tsv', sep='\t', index=False)

    return trajectories, reward_tensor
