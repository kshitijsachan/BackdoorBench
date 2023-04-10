"""
This file implements redwood's finetuning + regularization defense
"""

import argparse
import logging
import os
import random
import sys

from matplotlib import pyplot as plt

sys.path.append("../")
sys.path.append(os.getcwd())
import copy
import time
from pprint import pformat, pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from utils.aggregate_block.dataset_and_transform_generate import (
    get_input_shape,
    get_num_classes,
    get_transform,
)
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.bd_dataset import prepro_cls_DatasetBD

# from utils import args
from utils.choose_index import choose_index

# from utils.input_aware_utils import progress_bar
from utils.nCHW_nHWC import nCHW_to_nHWC
from utils.save_load_attack import load_attack_result


def get_args():
    # set the basic parameter
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, help="cuda, cpu")
    parser.add_argument("--checkpoint_load", type=str)
    parser.add_argument("--checkpoint_save", type=str)
    parser.add_argument("--experiment_name", type=str, help="experiment identifier")
    parser.add_argument("--log", type=str)
    parser.add_argument("--data_root", type=str)

    parser.add_argument("--dataset", type=str, help="mnist, cifar10, gtsrb, celeba, tiny")
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--input_height", type=int)
    parser.add_argument("--input_width", type=int)
    parser.add_argument("--input_channel", type=int)

    # parser.add_argument("--epochs", type=int)
    # parser.add_argument("--lr", type=float)
    # parser.add_argument("--lr_scheduler", type=str, help="the scheduler of lr")
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--num_workers", type=float)

    parser.add_argument("--poison_rate", type=float)
    parser.add_argument("--target_type", type=str, help="all2one, all2all, cleanLabel")
    parser.add_argument("--target_label", type=int)

    parser.add_argument("--model", type=str, help="resnet18")
    parser.add_argument("--random_seed", type=int, help="random seed")
    parser.add_argument("--index", type=str, help="index of clean data")
    parser.add_argument("--result_file", type=str, help="the location of result")

    parser.add_argument("--yaml_path", type=str, default="./config/defense/rrft/config.yaml", help="the path of yaml")

    parser.add_argument("--rrfs", action="store_true", help="load data and save files to rrfs instead of locally")

    # set the parameter for the ft defense
    parser.add_argument("--ft_n_clean", type=int, help="size of the clean set")
    parser.add_argument("--ft_lr", type=float, help="fine tuning learning rate")
    parser.add_argument("--ft_reg_coeff", type=float, help="fine tuning regularization coefficient")
    parser.add_argument("--ft_temp", type=float, help="temperature for fine tuning distillation loss", default=2.0)
    parser.add_argument(
        "--ft_lr_schedule",
        type=str,
        choices=["constant", "linear"],
        help="lr scheduler for fine tuning",
        default="linear",
    )
    parser.add_argument(
        "--ft_epochs",
        type=int,
        help="num epochs to train finetuning model",
    )
    parser.add_argument(
        "--ft_batch_size",
        type=int,
        help="batch size for fine tuning model",
    )

    arg = parser.parse_args()

    print(arg)
    return arg


def kl_div(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p_logprobs = F.log_softmax(p_logits, dim=-1)
    q_logprobs = F.log_softmax(q_logits, dim=-1)
    return torch.sum(torch.exp(p_logprobs) * (p_logprobs - q_logprobs), dim=-1)


def fine_tuning(arg, model, ft_model, ft_optimizer, ft_scheduler, epoch, trainloader):
    train_loss = 0
    batch_loss = []
    model.eval()
    ft_model.train()
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(arg.device), labels.to(arg.device)
        with torch.no_grad():
            orig_logits = model(inputs)
        ft_logits = ft_model(inputs)
        loss = torch.mean(kl_div(orig_logits / arg.ft_temp, ft_logits / arg.ft_temp))
        batch_loss.append(loss.item())
        train_loss += loss.item()
        loss.backward()
        ft_optimizer.step()
        ft_optimizer.zero_grad()

    with torch.no_grad():
        l2norm = torch.stack([torch.sum(p**2) for p in ft_model.parameters()], dim=0).sum().sqrt()
        logging.info(
            f"Epoch {epoch + 1}/{arg.ft_epochs}\tLoss: {sum(batch_loss) / len(batch_loss):.3f}\tL2 norm: {l2norm.item():.3f}"
        )
    ft_scheduler.step()
    return ft_model


def get_no_bd_process_dataloader(train, result, dataname: str, batch_size: int, num_workers: int):
    x = result[dataname]["x"]
    y = result[dataname]["y"]
    data = list(zip(x, y))
    process_data = prepro_cls_DatasetBD(
        full_dataset_without_transform=data,
        poison_idx=np.zeros(len(data)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    dataloader = torch.utils.data.DataLoader(
        process_data,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=True,
        pin_memory=True,
    )
    return dataloader


def rrft(args, result, config):
    logFormatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
    )
    logger = logging.getLogger()
    fileHandler = logging.FileHandler(f"{args.log}/{time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())}.log")
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(logging.INFO)
    logging.info(pformat(args.__dict__))

    fix_random(args.random_seed)

    # Prepare model, optimizer, scheduler, loss_fn
    model = generate_cls_model(args.model, args.num_classes)
    model.load_state_dict(result["model"])
    ft_model = copy.deepcopy(model)
    for p in ft_model.parameters():
        p.requires_grad_(True)

    model.to(args.device)
    ft_model.to(args.device)
    optimizer = torch.optim.AdamW(ft_model.parameters(), lr=args.ft_lr, weight_decay=args.ft_reg_coeff)
    if args.ft_lr_scheduler == "constant":
        optim_fn = lambda _epoch: 1
    elif args.ft_lr_scheduler == "linear":
        optim_fn = lambda epoch: (args.ft_epochs - epoch) / args.ft_epochs
    else:
        raise ValueError("Invalid scheduler: ", args.ft_lr_scheduler)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, optim_fn)

    # get data
    tran = get_transform(args.dataset, *([args.input_height, args.input_width]), train=True)
    x = result["clean_train"]["x"]
    y = result["clean_train"]["y"]
    ran_idx = random.sample(range(len(y)), args.ft_n_clean)

    np.savetxt(f"{args.checkpoint_save}/clean_data_indices.txt", ran_idx, fmt="%d")
    data_set = list(zip([x[ii] for ii in ran_idx], [y[ii] for ii in ran_idx]))
    data_set_o = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_set,
        poison_idx=np.zeros(len(data_set)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    trainloader = torch.utils.data.DataLoader(
        data_set_o, batch_size=args.ft_batch_size, num_workers=args.num_workers, shuffle=True
    )

    for i in range(args.ft_epochs):
        ft_model = fine_tuning(
            args,
            model,
            ft_model,
            optimizer,
            scheduler,
            i,
            trainloader,
        )

    return model, ft_model


@torch.no_grad()
def anomaly_score(model, ft_model, data: torch.Tensor) -> torch.Tensor:
    orig_logits = model(data)
    finetuned_logits = ft_model(data)
    return kl_div(orig_logits, finetuned_logits)


def compute_roc_auc(small_scores, large_scores):
    nsmall = len(small_scores)
    nlarge = len(large_scores)
    scores = np.concatenate((small_scores, large_scores))
    labels = np.concatenate((np.full(nsmall, True), np.full(nlarge, False)))
    order = np.argsort(scores)
    scores = scores[order]
    labels = labels[order]

    small_upto = np.cumsum(labels)
    large_upto = np.cumsum(1 - labels)
    tpr = small_upto / nsmall
    fpr = large_upto / nlarge
    # TODO: more numerically precise way?
    fpr_delta = np.diff(fpr, prepend=0.0)
    roc_auc = np.sum(tpr * fpr_delta)
    return roc_auc


if __name__ == "__main__":
    import ray

    ray.init(_node_ip_address="10.8.0.1")
    ### 1. basic setting: args
    args = get_args()
    with open(args.yaml_path, "r") as stream:
        config = yaml.safe_load(stream)
    config.update({k: v for k, v in args.__dict__.items() if v is not None})
    args.__dict__ = config
    args.num_classes = get_num_classes(args.dataset)
    args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
    args.img_size = (args.input_height, args.input_width, args.input_channel)

    bdbench_root_dir = os.path.expanduser("~/rrfs/ksachan/backdoor_bench") if args.rrfs else "."
    attack_record_dir = f"{bdbench_root_dir}/record/{args.result_file}"
    if args.checkpoint_save is None:
        args.checkpoint_save = f"{attack_record_dir}/defense/rrft/{args.experiment_name}"
        if not (os.path.exists(args.checkpoint_save)):
            os.makedirs(args.checkpoint_save)
    if args.log is None:
        args.log = f"{args.checkpoint_save}/log"
        if not (os.path.exists(args.log)):
            os.makedirs(args.log)

    ### 2. attack result(model, train data, test data)
    result = load_attack_result(attack_record_dir + "/attack_result.pt")

    print("Continue training...")
    ### 3. ft defense:
    model, ft_model = rrft(args, result, config)

    ### 4. test the result and get ASR, ACC, RC
    model.eval()
    ft_model.eval()

    tran = get_transform(args.dataset, *([args.input_height, args.input_width]), train=False)
    data_bd_loader = get_no_bd_process_dataloader(tran, result, "bd_test", args.eval_batch_size, args.num_workers)
    data_clean_loader = get_no_bd_process_dataloader(tran, result, "clean_test", args.eval_batch_size, args.num_workers)

    bd_scores = []
    for i, (inputs, _labels) in enumerate(data_bd_loader):  # type: ignore
        inputs = inputs.to(args.device)
        bd_scores.extend(anomaly_score(model, ft_model, inputs).tolist())

    clean_scores = []
    clean_acc = []
    for i, (inputs, labels) in enumerate(data_clean_loader):  # type: ignore
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        clean_scores.extend(anomaly_score(model, ft_model, inputs).tolist())
        with torch.no_grad():
            orig_model_logits = model(inputs)
            orig_model_pred = torch.max(orig_model_logits, dim=1)[1]
            clean_acc.extend((orig_model_pred == labels).tolist())

    roc_auc = compute_roc_auc(clean_scores, bd_scores)
    print("Num parameters in model: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"ROC AUC: {roc_auc:.4f}")

    torch.save(
        {
            "model_name": args.model,
            # not saving original model because it didn't change after finetuning
            # "model": model.cpu().state_dict(),
            "ft_model": ft_model.cpu().state_dict(),
        },
        f"{args.checkpoint_save}/model.pt",
    )
    torch.save(
        {
            "roc_auc": roc_auc,
            "clean_scores": clean_scores,
            "bd_scores": bd_scores,
            "clean_is_correct": clean_acc,
        },
        f"{args.checkpoint_save}/result.pt",
    )
