import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
    AdamW,
)
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from metric import SpanEntityScore, print_table
from loss import FocalLoss, contrastive_loss, KL_loss, sup_contrastive_loss
from tqdm import tqdm
from collections import defaultdict
from common import seed_everything, rope, get_linear_schedule_with_warmup_two_stage
from init_flat_mlp import NerModel, NerDataLoader, MyIter, Evaluater, KMean
from copy import deepcopy

import warnings
import numpy as np

np.warnings = warnings  # 兼容新版 NumPy 与 pyclustering

import math
import json
import random
import os
import time
import dill as pickle
import random

seed = 42
seed_everything(seed=seed)

# device_ids = [0, 1, 2]
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, device_ids))


def source_trainer():
    """
    source训练
    """
    epochs = 3
    lr = 2e-5
    max_length = 345
    max_grad_norm = 1.0
    data_size = 1
    batch_size = 12
    dropout_locate = 0.2
    dropout_type = 0.2
    weight_decay = 0
    warmup_proportion = 1e-1
    max_k = 10
    tau = 0.1
    model_type = "bert"
    data_name = "en_conll03"  # target data
    is_nest = data_name in ["ace04", "ace05", "ace2004", "genia"]
    assert model_type in ["bert", "roberta", "chinesebert-base", "chinesebert-large"]

    vec_dir = os.path.join(
        "..", "Nest_KNN", "models", "pretrained_models", "bert-base-uncased"
    )
    save_dir = os.path.join(".", "models", data_name)

    data_dir = os.path.join(".", "data_flat", "en_conll03", "train.ner.bio")
    label_dir = os.path.join(".", "data_flat", "en_conll03", "ner_labels.txt")
    # coarse to fine时用
    # data_dir = os.path.join(
    #     '.', 'data_flat', 'few-nerd-c2f', 'coarse', 'train.ner.bio')
    # label_dir = os.path.join(
    #     '.', 'data_flat', 'few-nerd-c2f', 'coarse', 'ner_labels.txt')

    # data_dir = os.path.join(
    #     ".", "data_flat", "few-nerd-c2f", "ontonotes", "train.ner.bio"
    # )
    # label_dir = os.path.join(
    #     ".", "data_flat", "few-nerd-c2f", "ontonotes", "ner_labels.txt"
    # )

    if model_type == "chinesebert-base":
        tokenizer = AutoTokenizer.from_pretrained(
            "iioSnail/ChineseBERT-base", trust_remote_code=True, cache_dir=vec_dir
        )
    elif model_type == "chinesebert-large":
        tokenizer = AutoTokenizer.from_pretrained(
            "iioSnail/ChineseBERT-large", trust_remote_code=True, cache_dir=vec_dir
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(vec_dir)

    dataset = NerDataLoader(
        tokenizer,
        data_dir,
        label_dir,
        max_length=max_length,
        data_size=data_size,
        model_type=model_type,
    )
    label2idx, idx2label = dataset.get_soups()
    print(label2idx, len(idx2label), "max_len: {}".format(dataset.max_len))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = NerModel(
        vec_dir=vec_dir,
        num_labels=len(idx2label),
        dropout_locate=dropout_locate,
        dropout_type=dropout_type,
        model_type=model_type,
    ).cuda()

    km = KMean(
        dataset=dataset, dataloader=dataloader, max_length=max_length, max_k=max_k
    )
    km.update(model)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = DataParallel(model)
    model = model.cuda()

    opt = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        no_deprecation_warning=True,
    )
    epoch_steps = len(dataloader)
    # print(len(dataset), len(dataloader))
    updates_total_stage = epoch_steps * epochs
    scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=int(warmup_proportion * updates_total_stage),
        num_training_steps=updates_total_stage,
    )

    locate_loss = nn.CrossEntropyLoss()
    type_loss = nn.CrossEntropyLoss()

    model.train()
    for epoch in tqdm(range(epochs), ncols=50):
        print("Epoch: {}".format(epoch))
        Loss, start = 0, time.time()
        if epoch:
            km.update(model)  # 更新聚类点
        for idx_d, (text, pinyin, label, label_loc, mask, idx) in enumerate(dataloader):
            # print(text, pinyin, label, mask, idx)
            text, pinyin, label, label_loc, mask = (
                text.cuda(),
                pinyin.cuda(),
                label.cuda(),
                label_loc.cuda(),
                mask.cuda(),
            )
            with autocast():
                # locate_label : [batch_size, seq_len, 2]
                hidden, type_label, locate_label = model((text, pinyin, mask), label)
            # # 通过mask只更新有用的token
            # locate_loss
            loss_mask = mask.view(-1).bool()
            pre_locate_label = locate_label.reshape(-1, locate_label.size()[-1])
            active_pre_loc = pre_locate_label.masked_select(
                loss_mask.unsqueeze(1)
            ).view(-1, pre_locate_label.size(-1))
            active_loc = label_loc.view(-1).masked_select(loss_mask)
            # type loss
            pre_type_label = type_label[mask == 1].view(-1, type_label.shape[-1])
            type_label, type_hidden = label[mask == 1].view(-1), hidden[mask == 1].view(
                -1, hidden.shape[-1]
            )
            sub_type_label = torch.tensor(
                [0] * type_label.shape[0], device=type_label.device
            )

            # 将label转成sub_label，使得往最靠近的sub_label靠近
            for i, v in enumerate(type_hidden):
                id = int(type_label[i])
                # if id: # 如果O要多个点聚类的话，这里需要改，一个点不用改
                sub_type_label[i] = km.get_sub(v, id)

            loss_loc = locate_loss(active_pre_loc, active_loc)
            loss_subtype = sup_contrastive_loss(
                type_hidden, sub_type_label, tau=tau
            )  # 细粒度类间的距离拉开
            # loss_subtype = contrastive_loss(type_hidden, type_label, tau=0.1)
            loss_type = type_loss(
                pre_type_label, type_label
            )  # 粗粒度类间的距离拉开，可以不拉开
            l = loss_loc + loss_subtype + loss_type
            # l = loss_loc + loss_type

            l.backward()
            Loss += l

            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            opt.step()
            opt.zero_grad(True)
            scheduler.step()

            if idx_d % (int(len(dataloader) / 10) + 1) == 0:
                # print(type_label)
                print(
                    "loss_loc: {:5f}   loss_type: {:5f}   loss_subtype: {:5f}".format(
                        loss_loc, loss_type, loss_subtype
                    )
                )
                print("loss: {:.5f}   lr: {:.2e}".format(Loss, scheduler.get_lr()[0]))
                Loss = 0
        print("Training one epoch use {:.3f}s".format(time.time() - start))

        origin_model = model
        if isinstance(model, DataParallel):
            origin_model = model.module
        source_dir = origin_model.save(
            origin_model,
            save_dir=save_dir,
            model_type=model_type,
            max_k=max_k,
            # coarse=f"{data_name}_{epochs}_km_y",
        )
    print(
        f"epochs: {epochs}, lr: {lr}, max_length: {max_length}, batch_size: {batch_size}, max_k: {max_k}, tau: {tau}"
    )
    return source_dir


def target_trainer():
    # source train
    # source_dir = source_trainer()
    source_dir = "./models/en_conll03/bert_10.pkl"
    print("source dir: {}".format(source_dir))
    # exit()

    # target train
    epochs = 30
    lr = 1e-4
    max_length = 128
    max_grad_norm = 1.0
    data_size = 1
    batch_size = 2
    weight_decay = 0
    warmup_proportion = 1e-1
    max_k = 10
    tau = 0.1
    model_type = "bert"
    data_name = "few-nerd-c2f"
    is_nest = data_name in ["ace04", "ace05", "ace2004", "genia"]
    assert model_type in ["bert", "roberta", "chinesebert-base", "chinesebert-large"]

    vec_dir = os.path.join(
        "..", "Nest_KNN", "models", "pretrained_models", "bert-base-uncased"
    )
    dir = os.path.join(".", "data_flat", data_name)
    save_dir = os.path.join(".", "models", data_name)
    # data_dir = os.path.join(dir, '10-shot-train.bio')
    # dev_dir = os.path.join(dir, 'test.bio')
    # label_dir = os.path.join(dir, 'ner_labels.txt')

    # corse to fine时使用
    data_dir = os.path.join(
        ".",
        "data_flat",
        "few-nerd-c2f",
        "medium-fine",
        "5shot",
        "5-shot-train1.ner.bio",
    )
    dev_dir = os.path.join(
        ".", "data_flat", "few-nerd-c2f", "medium-fine", "test.ner.bio"
    )
    label_dir = os.path.join(
        ".", "data_flat", "few-nerd-c2f", "medium-fine", "ner_labels.txt"
    )

    if model_type == "chinesebert-base":
        tokenizer = AutoTokenizer.from_pretrained(
            "iioSnail/ChineseBERT-base", trust_remote_code=True, cache_dir=vec_dir
        )
    elif model_type == "chinesebert-large":
        tokenizer = AutoTokenizer.from_pretrained(
            "iioSnail/ChineseBERT-large", trust_remote_code=True, cache_dir=vec_dir
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(vec_dir)

    dataset = NerDataLoader(
        tokenizer,
        data_dir,
        label_dir,
        max_length=max_length,
        data_size=data_size,
        model_type=model_type,
    )
    label2idx, idx2label = dataset.get_soups()
    print(label2idx, len(idx2label), "max_len: {}".format(dataset.max_len))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # source model
    with open(source_dir, "rb") as f:
        model = pickle.load(f)
        model.update(len(idx2label))

    km = KMean(
        dataset=dataset, dataloader=dataloader, max_length=max_length, max_k=max_k
    )
    km.update(model)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = DataParallel(model)
    model = model.cuda()

    opt = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        no_deprecation_warning=True,
    )
    epoch_steps = len(dataloader)
    # print(len(dataset), len(dataloader))
    updates_total_stage = epoch_steps * epochs
    scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=int(warmup_proportion * updates_total_stage),
        num_training_steps=updates_total_stage,
    )

    locate_loss = nn.CrossEntropyLoss()
    type_loss = nn.CrossEntropyLoss()

    dev_dataset = NerDataLoader(
        tokenizer, dev_dir, label_dir, max_length=max_length, model_type=model_type
    )
    dev_dataset.get_soups(dev=False)
    evaluater = Evaluater(dev_dataset, is_nest=is_nest, model_dir=None)
    # evaluater.model = model
    # evaluater.evaluate_ner(km=km)

    model.train()
    best_f1, best_epoch = 0, 0
    for epoch in tqdm(range(epochs), ncols=50):
        print("Epoch: {}".format(epoch))
        Loss, start = 0, time.time()
        if epoch:
            km.update(model)  # 更新聚类点
        for idx_d, (text, pinyin, label, label_loc, mask, idx) in enumerate(dataloader):
            # print(text, pinyin, label, mask, idx)
            text, pinyin, label, label_loc, mask = (
                text.cuda(),
                pinyin.cuda(),
                label.cuda(),
                label_loc.cuda(),
                mask.cuda(),
            )
            with autocast():
                # locate_label : [batch_size, seq_len, 2]
                hidden, type_label, locate_label = model((text, pinyin, mask), label)
            # # 通过mask只更新有用的token
            # locate_loss
            loss_mask = mask.view(-1).bool()
            pre_locate_label = locate_label.reshape(-1, locate_label.size()[-1])
            active_pre_loc = pre_locate_label.masked_select(
                loss_mask.unsqueeze(1)
            ).view(-1, pre_locate_label.size(-1))
            active_loc = label_loc.view(-1).masked_select(loss_mask)
            # type loss
            pre_type_label = type_label[mask == 1].view(-1, type_label.shape[-1])
            type_label, type_hidden = label[mask == 1].view(-1), hidden[mask == 1].view(
                -1, hidden.shape[-1]
            )
            sub_type_label = torch.tensor(
                [0] * type_label.shape[0], device=type_label.device
            )

            # 将label转成sub_label，使得往最靠近的sub_label靠近
            for i, v in enumerate(type_hidden):
                id = int(type_label[i])
                # if id: # 如果O要多个点聚类的话，这里需要改，一个点不用改
                sub_type_label[i] = km.get_sub(v, id)

            loss_loc = locate_loss(active_pre_loc, active_loc)
            loss_subtype = sup_contrastive_loss(
                type_hidden, sub_type_label, tau=tau
            )  # 细粒度类间的距离拉开
            # loss_subtype = contrastive_loss(type_hidden, sub_type_label, tau=0.1)
            loss_type = type_loss(
                pre_type_label, type_label
            )  # 粗粒度类间的距离拉开，可以不拉开
            l = loss_loc + loss_subtype + loss_type
            # l = loss_loc + loss_type

            l.backward()
            Loss += l

            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            opt.step()
            opt.zero_grad(True)
            scheduler.step()

            if idx_d % (int(len(dataloader) / 10) + 1) == 0:
                # print(type_label)
                print(
                    "loss_loc: {:5f}   loss_type: {:5f}   loss_subtype: {:5f}".format(
                        loss_loc, loss_type, loss_subtype
                    )
                )
                print("loss: {:.5f}   lr: {:.2e}".format(Loss, scheduler.get_lr()[0]))
                Loss = 0
        print("Training one epoch use {:.3f}s".format(time.time() - start))
        evaluater.model = model
        dev_f1 = evaluater.evaluate_ner_batch(km=km)

        if dev_f1 >= best_f1:
            best_f1 = dev_f1
            best_epoch = epoch
            origin_model = model
            if isinstance(model, DataParallel):
                origin_model = model.module
            origin_model.save(
                origin_model, save_dir=save_dir, model_type=model_type, max_k=max_k
            )
            with open(
                os.path.join(save_dir, "km_{}.pkl".format(model_type)), "wb"
            ) as f:
                # 减少需保存的数据
                th_km = deepcopy(km)
                th_km.vecs, th_km.dataset, th_km.dataloader = None, None, None
                pickle.dump(th_km, f)
    print(
        f"epochs: {epochs}, lr: {lr}, max_length: {max_length}, batch_size: {batch_size}, max_k: {max_k}, tau: {tau}"
    )
    print("best epoch: {} f1_score: {:.4f}".format(best_epoch, best_f1))


def target_nokmeans_trainer():
    # source train
    # source_dir = source_trainer()
    source_dir = "./models/en_conll03/bert_20.pkl"
    print("source dir: {}".format(source_dir))
    # exit()

    # target train
    epochs = 20
    lr = 1e-4
    max_length = 215
    max_grad_norm = 1.0
    data_size = 1
    batch_size = 2
    weight_decay = 0
    warmup_proportion = 1e-1
    model_type = "bert"
    data_name = "few-nerd-c2f"
    is_nest = data_name in ["ace04", "ace05", "ace2004", "genia"]
    assert model_type in ["bert", "roberta", "chinesebert-base", "chinesebert-large"]

    vec_dir = os.path.join(
        "..", "Nest_KNN", "models", "pretrained_models", "bert-base-uncased"
    )
    dir = os.path.join(".", "data_flat", data_name)
    save_dir = os.path.join(".", "models", data_name)
    # data_dir = os.path.join(dir, '10-shot-train.bio')
    # dev_dir = os.path.join(dir, 'test.bio')
    # label_dir = os.path.join(dir, 'ner_labels.txt')

    # corse to fine时使用
    data_dir = os.path.join(
        ".",
        "data_flat",
        "few-nerd-c2f",
        "medium-fine",
        "5shot",
        "5-shot-train3.ner.bio",
    )
    dev_dir = os.path.join(
        ".", "data_flat", "few-nerd-c2f", "medium-fine", "test.ner.bio"
    )
    label_dir = os.path.join(
        ".", "data_flat", "few-nerd-c2f", "medium-fine", "ner_labels.txt"
    )

    if model_type == "chinesebert-base":
        tokenizer = AutoTokenizer.from_pretrained(
            "iioSnail/ChineseBERT-base", trust_remote_code=True, cache_dir=vec_dir
        )
    elif model_type == "chinesebert-large":
        tokenizer = AutoTokenizer.from_pretrained(
            "iioSnail/ChineseBERT-large", trust_remote_code=True, cache_dir=vec_dir
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(vec_dir)

    dataset = NerDataLoader(
        tokenizer,
        data_dir,
        label_dir,
        max_length=max_length,
        data_size=data_size,
        model_type=model_type,
    )
    label2idx, idx2label = dataset.get_soups()
    print(label2idx, len(idx2label), "max_len: {}".format(dataset.max_len))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # source model
    with open(source_dir, "rb") as f:
        model = pickle.load(f)
        model.update(len(idx2label))

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = DataParallel(model)
    model = model.cuda()

    opt = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        no_deprecation_warning=True,
    )
    epoch_steps = len(dataloader)
    # print(len(dataset), len(dataloader))
    updates_total_stage = epoch_steps * epochs
    scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=int(warmup_proportion * updates_total_stage),
        num_training_steps=updates_total_stage,
    )

    locate_loss = nn.CrossEntropyLoss()
    type_loss = nn.CrossEntropyLoss()

    dev_dataset = NerDataLoader(
        tokenizer, dev_dir, label_dir, max_length=max_length, model_type=model_type
    )
    dev_dataset.get_soups(dev=True)
    evaluater = Evaluater(dev_dataset, is_nest=is_nest, model_dir=None)
    # evaluater.model = model
    # evaluater.evaluate_ner(km=km)

    best_f1, best_epoch = 0, 0
    model.train()
    for epoch in tqdm(range(epochs), ncols=50):
        print("Epoch: {}".format(epoch))
        Loss, start = 0, time.time()
        for idx_d, (text, pinyin, label, label_loc, mask, idx) in enumerate(dataloader):
            # print(text, pinyin, label, mask, idx)
            text, pinyin, label, label_loc, mask = (
                text.cuda(),
                pinyin.cuda(),
                label.cuda(),
                label_loc.cuda(),
                mask.cuda(),
            )
            with autocast():
                # locate_label : [batch_size, seq_len, 2]
                _, type_label, locate_label = model((text, pinyin, mask), label)
                # # 通过mask只更新有用的token
                # locate_loss
                loss_mask = mask.view(-1).bool()
                pre_locate_label = locate_label.reshape(-1, locate_label.size()[-1])
                active_pre_loc = pre_locate_label.masked_select(
                    loss_mask.unsqueeze(1)
                ).view(-1, pre_locate_label.size(-1))
                active_loc = label_loc.view(-1).masked_select(loss_mask)
                # type loss
                pre_type_label = type_label[mask == 1].view(-1, type_label.shape[-1])
                type_label = label[mask == 1].view(-1)

                loss_loc = locate_loss(active_pre_loc, active_loc)
                loss_type = type_loss(
                    pre_type_label, type_label
                )  # 粗粒度类间的距离拉开，可以不拉开
                l = loss_loc + loss_type

                l.backward()
            Loss += l

            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            opt.step()
            opt.zero_grad(True)
            scheduler.step()

            if idx_d % (int(len(dataloader) / 10) + 1) == 0:
                # print(type_label)
                print("loss_loc: {:5f}   loss_type: {:5f}".format(loss_loc, loss_type))
                print("loss: {:.5f}   lr: {:.2e}".format(Loss, scheduler.get_lr()[0]))
                Loss = 0
        print("Training one epoch use {:.3f}s".format(time.time() - start))
        evaluater.model = model
        dev_f1 = evaluater.evaluate_ner(km=None)

        if dev_f1 >= best_f1:
            best_f1 = dev_f1
            best_epoch = epoch
            origin_model = model
            if isinstance(model, DataParallel):
                origin_model = model.module
            origin_model.save(
                origin_model, save_dir=save_dir, model_type=model_type, max_k=0
            )

    print("best epoch: {} f1_score: {:.4f}".format(best_epoch, best_f1))


def trainer():
    epochs = 20
    lr = 2e-5
    max_length = 125
    max_grad_norm = 1.0
    data_size = 1
    batch_size = 4
    dropout_locate = 0.2
    dropout_type = 0.2
    weight_decay = 0
    warmup_proportion = 1e-1
    stage_one_lr_scale = 1.5
    max_k = 10
    model_type = "bert"
    data_name = "gum"
    is_nest = data_name in ["ace04", "ace05", "ace2004", "genia"]
    assert model_type in ["bert", "roberta", "chinesebert-base", "chinesebert-large"]

    vec_dir = os.path.join(".", "models", "pretrained_models", "bert-base-cased")
    dir = os.path.join(".", "data_flat", data_name)
    save_dir = os.path.join(".", "models", data_name)
    # data_dir = os.path.join(dir, '10-shot-train.bio')
    # dev_dir = os.path.join(dir, 'test.bio')
    # label_dir = os.path.join(dir, 'ner_labels.txt')

    # coarse to fine
    data_dir = os.path.join(
        ".", "data_flat", "few-shot", "gum", "5shot", "5-shot-train1.ner.bio"
    )
    dev_dir = os.path.join(".", "data_flat", "few-shot", "gum", "test.ner.bio")
    label_dir = os.path.join(".", "data_flat", "few-shot", "gum", "ner_labels.txt")

    if model_type == "chinesebert-base":
        tokenizer = AutoTokenizer.from_pretrained(
            "iioSnail/ChineseBERT-base", trust_remote_code=True, cache_dir=vec_dir
        )
    elif model_type == "chinesebert-large":
        tokenizer = AutoTokenizer.from_pretrained(
            "iioSnail/ChineseBERT-large", trust_remote_code=True, cache_dir=vec_dir
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(vec_dir)

    dataset = NerDataLoader(
        tokenizer,
        data_dir,
        label_dir,
        max_length=max_length,
        data_size=data_size,
        model_type=model_type,
    )
    label2idx, idx2label = dataset.get_soups()
    print(label2idx, len(idx2label), "max_len: {}".format(dataset.max_len))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    model = NerModel(
        vec_dir=vec_dir,
        num_labels=len(idx2label),
        dropout_locate=dropout_locate,
        dropout_type=dropout_type,
        model_type=model_type,
    ).cuda()

    km = KMean(
        dataset=dataset, dataloader=dataloader, max_length=max_length, max_k=max_k
    )
    km.update(model)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = DataParallel(model)
    model = model.cuda()

    opt = AdamW(
        model.parameters(),
        lr=lr * stage_one_lr_scale,
        weight_decay=weight_decay,
        no_deprecation_warning=True,
    )
    epoch_steps = len(dataloader)
    # print(len(dataset), len(dataloader))
    updates_total_stage = epoch_steps * epochs
    scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=int(warmup_proportion * updates_total_stage),
        num_training_steps=updates_total_stage,
    )

    locate_loss = nn.CrossEntropyLoss()
    type_loss = nn.CrossEntropyLoss()

    dev_dataset = NerDataLoader(
        tokenizer, dev_dir, label_dir, max_length=max_length, model_type=model_type
    )
    dev_dataset.get_soups(dev=True)
    evaluater = Evaluater(dev_dataset, is_nest=is_nest, model_dir=None)
    # evaluater.model = model
    # evaluater.evaluate_ner(km=km)

    best_f1, best_epoch = 0, 0
    model.train()
    for epoch in tqdm(range(epochs), ncols=50):
        print("Epoch: {}".format(epoch))
        Loss, start = 0, time.time()
        if epoch:
            km.update(model)  # 更新聚类点
        for idx_d, (text, pinyin, label, label_loc, mask, idx) in enumerate(dataloader):
            # print(text, pinyin, label, mask, idx)
            text, pinyin, label, label_loc, mask = (
                text.cuda(),
                pinyin.cuda(),
                label.cuda(),
                label_loc.cuda(),
                mask.cuda(),
            )
            with autocast():
                # locate_label : [batch_size, seq_len, 2]
                hidden, type_label, locate_label = model((text, pinyin, mask), label)
                # # 通过mask只更新有用的token
                # locate_loss
                loss_mask = mask.view(-1).bool()
                pre_locate_label = locate_label.reshape(-1, locate_label.size()[-1])
                active_pre_loc = pre_locate_label.masked_select(
                    loss_mask.unsqueeze(1)
                ).view(-1, pre_locate_label.size(-1))
                active_loc = label_loc.view(-1).masked_select(loss_mask)
                # type loss
                pre_type_label = type_label[mask == 1].view(-1, type_label.shape[-1])
                type_label, type_hidden = label[mask == 1].view(-1), hidden[
                    mask == 1
                ].view(-1, hidden.shape[-1])
                sub_type_label = torch.tensor(
                    [0] * type_label.shape[0], device=type_label.device
                )

                # 将label转成sub_label，使得往最靠近的sub_label靠近
                for i, v in enumerate(type_hidden):
                    id = int(type_label[i])
                    # if id: # 如果O要多个点聚类的话，这里需要改，一个点不用改
                    sub_type_label[i] = km.get_sub(v, id)

                loss_loc = locate_loss(active_pre_loc, active_loc)
                loss_subtype = sup_contrastive_loss(
                    type_hidden, sub_type_label, tau=0.1
                )  # 细粒度类间的距离拉开
                # loss_subtype = contrastive_loss(type_hidden, type_label, tau=0.1)
                loss_type = type_loss(
                    pre_type_label, type_label
                )  # 粗粒度类间的距离拉开，可以不拉开
                l = loss_loc + loss_subtype + loss_type

                l.backward()
            Loss += l

            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            opt.step()
            opt.zero_grad(True)
            scheduler.step()

            if idx_d % (int(len(dataloader) / 10) + 1) == 0:
                # print(type_label)
                print(
                    "loss_loc: {:5f}   loss_type: {:5f}   loss_subtype: {:5f}".format(
                        loss_loc, loss_type, loss_subtype
                    )
                )
                print("loss: {:.5f}   lr: {:.2e}".format(Loss, scheduler.get_lr()[0]))
                Loss = 0
        print("Training one epoch use {:.3f}s".format(time.time() - start))
        evaluater.model = model
        dev_f1 = evaluater.evaluate_ner(km=km)

        if dev_f1 >= best_f1:
            best_f1 = dev_f1
            best_epoch = epoch
            origin_model = model
            if isinstance(model, DataParallel):
                origin_model = model.module
            origin_model.save(
                origin_model, save_dir=save_dir, model_type=model_type, max_k=max_k
            )
            with open(
                os.path.join(save_dir, "km_{}.pkl".format(model_type)), "wb"
            ) as f:
                # 减少需保存的数据
                th_km = deepcopy(km)
                th_km.vecs, th_km.dataset, th_km.dataloader = None, None, None
                pickle.dump(th_km, f)

    print("best epoch: {} f1_score: {:.4f}".format(best_epoch, best_f1))


if __name__ == "__main__":
    target_trainer()
    # target_nokmeans_trainer()
    # trainer()
