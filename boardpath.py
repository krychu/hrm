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
        boardpath_params: BoardPathParameters,
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

    hrm = HRM(
        vocab_cnt=hrm_params.vocab_cnt,
        seq_len=hrm_params.seq_len,
        d_model=hrm_params.d_model,
        head_cnt=hrm_params.head_cnt,
        sdpa_dropout=hrm_params.sdpa_dropout,
        bias_qkv=hrm_params.bias_qkv,
        bias_o=hrm_params.bias_o,
        expansion=hrm_params.expansion,
        elementwise_affine=hrm_params.elementwise_affine,
        dropout=hrm_params.dropout,
        H_block_cnt=hrm_params.H_block_cnt,
        L_block_cnt=hrm_params.L_block_cnt,
        H_cycle_cnt=hrm_params.H_cycle_cnt,
        L_cycle_cnt=hrm_params.L_cycle_cnt,
        infer_segment_cnt=hrm_params.infer_segment_cnt,
        head_bias=hrm_params.head_bias
    ).to(device)

    print()
    boardpath_summary(boardpath_params)
    hrm_summary(hrm_params, hrm_train_params, hrm, device)

    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        hrm.parameters(),
        lr=hrm_train_params.lr,
        weight_decay=hrm_train_params.weight_decay
    )

    initial_val_loss, initial_val_acc = evaluate(
        hrm=hrm,
        segment_cnt=hrm_params.infer_segment_cnt,
        loader=val_loader,
        device=device
    )
    print(f"[val] initial loss: {initial_val_loss:.4f} acc: {initial_val_acc:.3f}")

    best_val_loss = math.inf
    for epoch_idx in range(hrm_train_params.epoch_cnt):
        epoch_start_time = time.time()

        train_loss = train_one_epoch(
            hrm=hrm,
            loader=train_loader,
            ce_loss=ce_loss,
            optimizer=optimizer,
            device=device,
            segment_cnt=hrm_train_params.train_segment_cnt,
            grad_clip=hrm_train_params.grad_clip
        )
        val_loss, val_acc = evaluate(
            hrm=hrm,
            segment_cnt=hrm_params.infer_segment_cnt,
            loader=val_loader,
            device=device
        )

        epoch_time = time.time() - epoch_start_time

        print(f"epoch: {epoch_idx+1:03d} [trn] loss: {train_loss:.4f} [val] loss: {val_loss:.4f} acc: {val_acc:.3f} (time: {epoch_time:.0f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(hrm.state_dict(), filename)

    print()
    print(f"Model saved to: {filename}, best val loss: {best_val_loss:.4f}")

# def get_config_1():
#     boardpath_params = BoardPathParameters(
#         board_size=4,
#         train_count=5000,
#         val_count=500,
#         wall_prob=0.3
#     )

#     hrm_params = HRMParameters(
#         seq_len=boardpath_params.board_size * boardpath_params.board_size,
#         vocab_cnt=get_vocab_cnt(),
#         d_model=128,
#         nhead=4,
#         dim_feedforward=256,
#         H_layer_cnt=4,
#         L_layer_cnt=4,
#         H_cycle_cnt=2,
#         L_cycle_cnt=2,
#         infer_segment_cnt=1,
#         dropout=0.1
#     )

#     hrm_train_params = HRMTrainParameters(
#         train_segment_cnt=2,
#         epoch_cnt=1,
#         weight_decay=0.01, # default: 0.01, try 0.1
#         grad_clip=None, # 1.0
#         batch_size=64,
#         lr=3e-4
#     )

#     train_ds, val_ds = build_datasets(boardpath_params)
#     train_loader = DataLoader(
#         train_ds,
#         batch_size=hrm_train_params.batch_size,
#         shuffle=True
#     )
#     val_loader = DataLoader(
#         val_ds,
#         batch_size=hrm_train_params.batch_size,
#         shuffle=False
#     )

#     return boardpath_params, hrm_params, hrm_train_params, train_loader, val_loader

def get_config_1():
    boardpath_params = BoardPathParameters(
        board_size=4,
        train_count=5000,
        val_count=500,
        wall_prob=0.3
    )

    hrm_params = HRMParameters(
        seq_len=boardpath_params.board_size * boardpath_params.board_size,
        vocab_cnt=get_vocab_cnt(),
        d_model=128,
        head_cnt=4,
        sdpa_dropout=0.1,
        bias_qkv=False,
        bias_o=False,
        expansion=2,
        elementwise_affine=True,
        dropout=0.1,
        H_block_cnt=4,
        L_block_cnt=4,
        H_cycle_cnt=2,
        L_cycle_cnt=2,
        infer_segment_cnt=1,
        head_bias=False
    )

    hrm_train_params = HRMTrainParameters(
        train_segment_cnt=2,
        epoch_cnt=10,
        weight_decay=0.01, # default: 0.01, try 0.1
        grad_clip=None, # 1.0
        batch_size=64,
        lr=3e-4
    )

    train_ds, val_ds = build_datasets(boardpath_params)
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

    return boardpath_params, hrm_params, hrm_train_params, train_loader, val_loader

if __name__ == '__main__':
    set_all_seeds(42)
    boardpath_params, hrm_params, hrm_train_params, train_loader, val_loader = get_config_1()
    main(
        boardpath_params,
        hrm_params,
        hrm_train_params,
        train_loader,
        val_loader
    )
