"""
This file implements the defense method called finetuning (ft), which is a standard fine-tuning that uses clean data to finetune the model.

basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. ft defense:
        a. get some clean data
        b. retrain the backdoor model
    4. test the result and get ASR, ACC, RC 
"""

import argparse
import copy
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("../")
sys.path.append(os.getcwd())

import logging
import time
from pprint import pformat

import altair as alt
import polars as pl
import yaml

from defense.base import defense
from utils.aggregate_block.dataset_and_transform_generate import (
    get_input_shape,
    get_num_classes,
    get_transform,
)
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.train_settings_generate import (
    argparser_criterion,
    argparser_opt_scheduler,
)
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2
from utils.choose_index import choose_index
from utils.log_assist import get_git_info
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.trainer_cls import PureCleanModelTrainer


class KLDiv:
    def __init__(self, temperature):
        self.temperature = temperature

    def __call__(self, p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
        p_logprobs = F.log_softmax(p_logits / self.temperature, dim=-1)
        q_logprobs = F.log_softmax(q_logits / self.temperature, dim=-1)
        return torch.sum(torch.exp(p_logprobs) * (p_logprobs - q_logprobs), dim=-1)


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


class rrft(defense):
    def __init__(self, args):
        with open(args.yaml_path, "r") as f:
            defaults = yaml.safe_load(f)

        defaults.update({k: v for k, v in args.__dict__.items() if v is not None})

        args.__dict__ = defaults

        args.terminal_info = sys.argv

        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        self.args = args

        if "result_file" in args.__dict__:
            if args.result_file is not None:
                self.set_result(args.result_file)

    def add_arguments(parser):
        parser.add_argument("--device", type=str, help="cuda, cpu")
        parser.add_argument(
            "-pm", "--pin_memory", type=lambda x: str(x) in ["True", "true", "1"], help="dataloader pin_memory"
        )
        parser.add_argument(
            "-nb",
            "--non_blocking",
            type=lambda x: str(x) in ["True", "true", "1"],
            help=".to(), set the non_blocking = ?",
        )
        parser.add_argument("-pf", "--prefetch", type=lambda x: str(x) in ["True", "true", "1"], help="use prefetch")
        parser.add_argument("--amp", type=lambda x: str(x) in ["True", "true", "1"])

        parser.add_argument("--checkpoint_load", type=str, help="the location of load model")
        parser.add_argument("--checkpoint_save", type=str, help="the location of checkpoint where model is saved")
        parser.add_argument("--log", type=str, help="the location of log")
        parser.add_argument("--dataset_path", type=str, help="the location of data")
        parser.add_argument("--dataset", type=str, help="mnist, cifar10, cifar100, gtrsb, tiny")
        parser.add_argument("--result_file", type=str, help="the location of result")

        parser.add_argument("--epochs", type=int)
        parser.add_argument("--batch_size", type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument("--lr", type=float)
        parser.add_argument("--lr_scheduler", type=str, help="the scheduler of lr")
        parser.add_argument("--steplr_stepsize", type=int)
        parser.add_argument("--steplr_gamma", type=float)
        parser.add_argument("--steplr_milestones", type=list)
        parser.add_argument("--model", type=str, help="resnet18")

        parser.add_argument("--client_optimizer", type=int)
        parser.add_argument("--sgd_momentum", type=float)
        parser.add_argument("--wd", type=float, help="weight decay of sgd")
        parser.add_argument("--frequency_save", type=int, help=" frequency_save, 0 is never")

        parser.add_argument("--random_seed", type=int, help="random seed")
        parser.add_argument(
            "--yaml_path", type=str, default="./config/defense/rrft/cifar10.yaml", help="the path of yaml"
        )

        parser.add_argument("--experiment_name", type=str, help="the name of experiment", required=True)
        # rrfs
        parser.add_argument("--rrfs", action="store_true", help="load data and save files to rrfs instead of locally")

        # set the parameter for the ft defense
        parser.add_argument("--ft_n_clean", type=int, help="size of the clean set")
        parser.add_argument("--ft_eval_during_train", action="store_true", help="ft_model.eval() when finetuning")
        parser.add_argument("--ft_reg_coeff", type=float, help="fine tuning regularization coefficient")
        parser.add_argument("--kl_temp", type=float, help="temperature for fine tuning distillation loss", default=2.0)

    def set_result(self, result_file):
        record_path = os.path.expanduser("~/rrfs/ksachan/backdoor_bench/record") if args.rrfs else "record"
        attack_file = os.path.join(record_path, result_file)
        save_path = os.path.join(attack_file, "defense", "rrft", self.args.experiment_name)
        if not (os.path.exists(save_path)):
            os.makedirs(save_path)
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = os.path.join(save_path, "checkpoint")
            if not (os.path.exists(self.args.checkpoint_save)):
                os.makedirs(self.args.checkpoint_save)
        if self.args.log is None:
            self.args.log = os.path.join(save_path, "log")
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)
        self.result = load_attack_result(attack_file + "/attack_result.pt")

    def set_trainer(self, model):
        self.trainer = PureCleanModelTrainer(
            model,
        )

    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d:%H:%M:%S",
        )
        logger = logging.getLogger()

        fileHandler = logging.FileHandler(
            args.log + "/" + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + ".log"
        )
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

        logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))

        try:
            logging.info(pformat(get_git_info()))
        except:
            logging.info("Getting git info fails.")

        # stop printing log to console twice
        logger.propagate = False

    def set_devices(self):
        # self.device = torch.device(
        #     (
        #         f"cuda:{[int(i) for i in self.args.device[5:].split(',')][0]}" if "," in self.args.device else self.args.device
        #         # since DataParallel only allow .to("cuda")
        #     ) if torch.cuda.is_available() else "cpu"
        # )
        self.device = self.args.device

    @torch.no_grad()
    def get_scores_and_is_correct(self, model, ft_model, data_loader, is_bd_loader: bool, kl_div):
        scores = []
        is_correct = []
        for x in data_loader:
            if is_bd_loader:
                images, labels, _original_index, _poison_indicator, _original_targets = x
            else:
                images, labels = x
            images, labels = images.to(self.device), labels.to(self.device)
            orig_logits = model(images)
            orig_pred = torch.argmax(orig_logits, dim=1)
            ft_logits = ft_model(images)
            scores.extend(kl_div(orig_logits, ft_logits).tolist())
            is_correct.extend((orig_pred == labels).tolist())
        return scores, is_correct

    def mitigation(self):
        self.set_devices()
        fix_random(self.args.random_seed)

        # Prepare model, optimizer, scheduler
        model = generate_cls_model(self.args.model, self.args.num_classes)
        model.load_state_dict(self.result["model"])
        ft_model = copy.deepcopy(model)
        for p in ft_model.parameters():
            p.requires_grad_(True)
        for m in [model, ft_model]:
            if "," in self.device:
                m = torch.nn.DataParallel(
                    m, device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
                )
                self.args.device = f"cuda:{m.device_ids[0]}"
                m.to(self.args.device)
            else:
                m.to(self.args.device)

        optimizer, scheduler = argparser_opt_scheduler(ft_model, self.args)
        kl_div = KLDiv(args.kl_temp)

        # get clean train, clean test, and backdoor test
        train_tran = get_transform(self.args.dataset, *([self.args.input_height, self.args.input_width]), train=True)
        clean_dataset = prepro_cls_DatasetBD_v2(self.result["clean_train"].wrapped_dataset)
        if args.index is None:
            ran_idx = random.sample(range(len(clean_dataset)), self.args.ft_n_clean)
        else:
            ran_idx = np.loadtxt(args.index, dtype=int)
        log_index = os.path.join(self.args.log, "index.txt")
        np.savetxt(log_index, ran_idx, fmt="%d")
        clean_dataset.subset(ran_idx)
        data_set_without_tran = clean_dataset
        data_set_o = self.result["clean_train"]
        data_set_o.wrapped_dataset = data_set_without_tran
        data_set_o.wrap_img_transform = train_tran
        data_loader = torch.utils.data.DataLoader(
            data_set_o,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            pin_memory=args.pin_memory,
        )
        trainloader = data_loader

        test_tran = get_transform(self.args.dataset, *([self.args.input_height, self.args.input_width]), train=False)
        data_bd_testset = self.result["bd_test"]
        data_bd_testset.wrap_img_transform = test_tran
        data_bd_loader = torch.utils.data.DataLoader(
            data_bd_testset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            drop_last=False,
            shuffle=True,
            pin_memory=args.pin_memory,
        )

        data_clean_testset = self.result["clean_test"]
        data_clean_testset.wrap_img_transform = test_tran
        data_clean_loader = torch.utils.data.DataLoader(
            data_clean_testset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            drop_last=False,
            shuffle=True,
            pin_memory=args.pin_memory,
        )

        # Train fine tuned model
        model.eval()
        if args.ft_eval_during_train:
            ft_model.eval()
        else:
            ft_model.train()
        epoch_metrics = []
        batch_metrics = []
        for epoch in range(1, args.epochs + 1):
            running_loss = 0.0
            for images, labels, original_index, poison_indicator, original_targets in trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                with torch.no_grad():
                    orig_logits = model(images)
                ft_logits = ft_model(images)
                loss = torch.mean(kl_div(orig_logits, ft_logits))
                loss.backward()
                optimizer.step()
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(loss.item())
                elif scheduler is not None:
                    scheduler.step()
                running_loss += loss.item() * len(images)
                batch_metrics.append({"learning_rate": optimizer.param_groups[0]["lr"]})

            with torch.no_grad():
                running_loss /= len(trainloader.dataset)
                l2_norm = torch.stack([torch.sum(p**2) for p in ft_model.parameters()], dim=0).sum().sqrt().item()
                epoch_metrics.append(
                    {
                        "epoch": epoch,
                        "loss": running_loss,
                        "l2norm": l2_norm,
                    }
                )
                logging.info(f"Epoch: {epoch}, Loss: {running_loss}, L2 norm: {l2_norm}")

        # evaluate fine tuned model
        model.eval()
        ft_model.eval()
        clean_scores, clean_orig_correct = self.get_scores_and_is_correct(
            model, ft_model, data_clean_loader, False, kl_div
        )
        bd_scores, bd_orig_correct = self.get_scores_and_is_correct(model, ft_model, data_bd_loader, True, kl_div)
        roc_auc = compute_roc_auc(clean_scores, bd_scores)
        logging.info(f"ROC AUC: {roc_auc:.3f}")

        # save results
        batch_metrics_df = pl.DataFrame(batch_metrics).with_row_count("batch")
        epoch_metrics_df = pl.DataFrame(epoch_metrics)
        epoch_metrics_df.write_csv(os.path.join(args.save_path, "epoch_metrics.csv"))
        batch_metrics_df.write_csv(os.path.join(args.save_path, "batch_metrics.csv"))

        base = alt.Chart(epoch_metrics_df.to_pandas()).mark_line().encode(x=alt.X("epoch:O", title="Epoch"))
        color1, color2 = "#57A44C", "#4C78A8"
        loss_line = base.mark_line(stroke=color1).encode(
            y=alt.Y("loss:Q", axis=alt.Axis(title="Loss", titleColor=color1))
        )
        l2_norm_line = base.mark_line(stroke=color2).encode(
            y=alt.Y("l2norm:Q", axis=alt.Axis(title="L2 norm", titleColor=color2))
        )
        chart = alt.layer(loss_line, l2_norm_line).resolve_scale(y="independent")
        chart.save(os.path.join(args.save_path, "loss_l2norm.png"))
        print("Saved loss curves ", os.path.join(args.save_path, "loss_l2norm.png"))

        alt.Chart(batch_metrics_df.to_pandas()).mark_line().encode(
            x=alt.X("batch:Q", title="Batch"),
            y=alt.Y("learning_rate:Q", title="Learning rate"),
        ).save(os.path.join(args.save_path, "learning_rate.png"))
        print("Saved learning rate curve ", os.path.join(args.save_path, "learning_rate.png"))

        torch.save(
            {
                "model_name": args.model,
                "num_classes": args.num_classes,
                "ft_model": ft_model.cpu().state_dict(),
            },
            os.path.join(args.save_path, "ft_model.pth"),
        )
        torch.save(
            {
                "roc_auc": roc_auc,
                "clean_scores": clean_scores,
                "bd_scores": bd_scores,
                "clean_orig_correct": clean_orig_correct,
                "bd_orig_correct": bd_orig_correct,
            },
            os.path.join(args.save_path, "result.pth"),
        )
        return locals()

    def defense(self, result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result


if __name__ == "__main__":
    import ray

    # ray.init(_node_ip_address="10.8.0.1")
    parser = argparse.ArgumentParser(description=sys.argv[0])
    rrft.add_arguments(parser)
    args = parser.parse_args()
    ft_method = rrft(args)
    if "result_file" not in args.__dict__ or args.result_file is None:
        args.result_file = "defense_test_badnet"
    result = ft_method.defense(args.result_file)
