import math
import argparse
import time
from typing import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from hrm.hrm import *
from datasets.build_boardpath_dataset import *

def main(
        hrm_params: HRMParameters,
        hrm_train_params: HRMTrainParameters,
        train_loader: DataLoader,
        val_loader: DataLoader
):
    filename = "hrm_boardpath.pt"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    torch.manual_seed(42)

    hrm = HRM(
        vocab_cnt=hrm_params.vocab_cnt,
        seq_len=hrm_params.seq_len,
        d_model=hrm_params.d_model,
        nhead=hrm_params.nhead,
        dim_feedforward=hrm_params.dim_feedforward,
        H_layers=hrm_params.H_layer_cnt,
        L_layers=hrm_params.L_layer_cnt,
        H_cycles=hrm_params.H_cycle_cnt,
        L_cycles=hrm_params.L_cycle_cnt,
        dropout=hrm_params.dropout,
    ).to(device)

    hrm_summary(hrm_params, hrm_train_params, hrm, device)

    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        hrm.parameters(),
        lr=hrm_train_params.lr,
        weight_decay=hrm_train_params.weight_decay
    )

    best_val_loss = math.inf
    for epoch_idx in range(hrm_train_params.epoch_cnt):
        epoch_start_time = time.time()

        train_loss = train_one_epoch(
            hrm=hrm,
            loader=train_loader,
            ce_loss=ce_loss,
            optimizer=optimizer,
            device=device,
            segment_cnt=hrm_params.segment_cnt,
            grad_clip=hrm_train_params.grad_clip
        )
        val_loss, val_acc = evaluate(hrm, val_loader, device)

        epoch_time = time.time() - epoch_start_time

        print(f"epoch: {epoch_idx+1:03d} [trn] loss: {train_loss:.4f} [val] loss: {val_loss:.4f} acc: {val_acc:.3f} (time: {epoch_time:.0f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(hrm.state_dict(), filename)

    print(f"Done. Model saved to: {filename}, best val loss: {best_val_loss}")

def get_config_1():
    board_size = 4
    train_size = 5000
    val_size = 500
    wall_prob = 0.3

    hrm_params = HRMParameters(
        # board_size=board_size,
        seq_len=board_size * board_size,
        vocab_cnt=get_vocab_cnt(),
        d_model=128,
        nhead=4,
        dim_feedforward=256,
        H_layer_cnt=4,
        L_layer_cnt=4,
        H_cycle_cnt=2,
        L_cycle_cnt=2,
        segment_cnt=2,
        dropout=0.1
    )

    hrm_train_params = HRMTrainParameters(
        epoch_cnt=10,
        weight_decay=0.01, # default: 0.01, try 0.1
        grad_clip=None, # 1.0
        batch_size=64,
        lr=3e-4
    )

    train_ds = BoardPathDataset(size=board_size, count=train_size, wall_prob=wall_prob)
    val_ds = BoardPathDataset(size=board_size, count=val_size, wall_prob=wall_prob)
    train_loader = DataLoader(
        train_ds,
        batch_size=hrm_train_params.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=hrm_train_params.batch_size,
        shuffle=False
    )

    return hrm_params, hrm_train_params, train_loader, val_loader

if __name__ == '__main__':
    hrm_params, hrm_train_params, train_loader, val_loader = get_config_1()
    main(
        hrm_params,
        hrm_train_params,
        train_loader,
        val_loader
    )
