import math
import argparse
import time
from typing import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from hrm.hrm import *
from datasets.build_boardpath_dataset import *

def setup_model_and_device(hrm_params: HRMParameters) -> Tuple[HRM, torch.device]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # device = torch.device("cpu")

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

    return hrm, device

def run_training(
        boardpath_params: BoardPathParameters,
        hrm_params: HRMParameters,
        hrm_train_params: HRMTrainParameters,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_path: str
):
    hrm, device = setup_model_and_device(hrm_params)

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
            torch.save(hrm.state_dict(), model_path)

    print()
    print(f"Model saved to: {model_path}, best val loss: {best_val_loss:.4f}")

def run_inference(
        boardpath_params: BoardPathParameters,
        hrm_params: HRMParameters,
        model_path: str
):
    """Run inference on a single random board sample."""
    hrm, device = setup_model_and_device(hrm_params)

    try:
        hrm.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from: {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Please train a model first.")
        return

    hrm.eval()

    # Generate a single sample
    input_board, target_board = generate_board(
        size=boardpath_params.board_size,
        max_wall_prob=boardpath_params.wall_prob
    )

    input_flat = input_board.flatten().unsqueeze(0).to(device)  # [1, seq_len]

    with torch.no_grad():
        z = hrm.init_z(input_flat)

        # Run a fixed number of segments (think time)
        for _ in range(hrm_params.infer_segment_cnt):
            z, logits_bsv = hrm(z, input_flat)
            # z = (z[0].detach(), z[1].detach())

        predicted = logits_bsv.argmax(dim=-1) # [B,S]
        # predicted = torch.argmax(output_logits, dim=-1)  # [1, seq_len]

    print("\nINPUT BOARD:")
    print(format_board(input_board.flatten(), boardpath_params.board_size))

    print("\nTARGET BOARD (ground truth):")
    print(format_board(target_board.flatten(), boardpath_params.board_size))

    print("\nPREDICTED BOARD:")
    print(format_board(predicted.squeeze(0).cpu(), boardpath_params.board_size))

    print("\nLegend: . = Floor, # = Wall, S = Start, E = End, * = Path")

def get_config_1() -> Tuple[
        BoardPathParameters,
        HRMParameters,
        HRMTrainParameters,
        DataLoader,
        DataLoader
]:
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
        expansion=2.0,
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

def get_config_1b() -> Tuple[
        BoardPathParameters,
        HRMParameters,
        HRMTrainParameters,
        DataLoader,
        DataLoader
]:
    boardpath_params = BoardPathParameters(
        board_size=8,
        train_count=5000,
        val_count=500,
        wall_prob=0.3
    )

    hrm_params = HRMParameters(
        seq_len=boardpath_params.board_size * boardpath_params.board_size,
        vocab_cnt=get_vocab_cnt(),
        d_model=256,
        head_cnt=8,
        sdpa_dropout=0.1,
        bias_qkv=False,
        bias_o=False,
        expansion=2.0,
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
        epoch_cnt=50,
        weight_decay=0.01, # default: 0.01, try 0.1
        grad_clip=None, # 1.0
        batch_size=64,
        lr=1e-4
    )

    # TODO: This is not needed for inference
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

# paper
def get_config_2() -> Tuple[
        BoardPathParameters,
        HRMParameters,
        HRMTrainParameters,
        DataLoader,
        DataLoader
]:
    boardpath_params = BoardPathParameters(
        board_size=4,
        train_count=2000,
        val_count=500,
        wall_prob=0.3
    )

    hrm_params = HRMParameters(
        seq_len=boardpath_params.board_size * boardpath_params.board_size,
        vocab_cnt=get_vocab_cnt(),
        d_model=512,
        head_cnt=8,
        sdpa_dropout=0.1,
        bias_qkv=False,
        bias_o=False,
        expansion=4.0,
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
        epoch_cnt=100,
        weight_decay=1.0, # default: 0.01, try 0.1
        grad_clip=None, # 1.0
        batch_size=768,
        lr=1e-4
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

def format_board(board_tensor: torch.Tensor, board_size: int) -> str:
    """Format a flattened board tensor as a visual grid."""
    board = board_tensor.view(board_size, board_size)
    symbols = {FLOOR: '.', WALL: '#', START: 'S', END: 'E', PATH: '*'}

    result = []
    for row in board:
        result.append(' '.join(symbols.get(int(cell), str(int(cell))) for cell in row))
    return '\n'.join(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HRM Boardpath Training and Inference')
    parser.add_argument('--mode', choices=['train', 'inference'], required=True,
                        help='Mode to run: train (trains and saves model) or inference (loads model and runs on random sample)')
    parser.add_argument('--model', default='hrm_boardpath.pt',
                        help='Model file path (default: hrm_boardpath.pt)')
    args = parser.parse_args()

    set_all_seeds(42)
    boardpath_params, hrm_params, hrm_train_params, train_loader, val_loader = get_config_1() # get_config_1b()

    if args.mode == 'train':
        run_training(
            boardpath_params,
            hrm_params,
            hrm_train_params,
            train_loader,
            val_loader,
            args.model
        )
    elif args.mode == 'inference':
        run_inference(
            boardpath_params,
            hrm_params,
            args.model
        )
