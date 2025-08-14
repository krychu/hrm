import torch
import random
from collections import deque
from torch.utils.data import Dataset, DataLoader, TensorDataset
from dataclasses import dataclass

FLOOR = 0
WALL  = 1
START = 2
END   = 3
PATH  = 4

@dataclass
class BoardPathParameters:
    board_size: int
    train_count: int
    val_count: int
    wall_prob: float

def get_vocab_cnt():
    return 5

def generate_board(size=4, max_wall_prob=0.3):
    """
    Generate a single valid board (input, target).
    input:  ints {0=FLOOR,1=WALL,2=START,3=END}
    target: same but with shortest path cells marked as PATH=4
    """

    while True:  # loop until we make a solvable board
        board = [[FLOOR]*size for _ in range(size)]

        # Place start and end
        start = (random.randrange(size), random.randrange(size))
        end = (random.randrange(size), random.randrange(size))
        while end == start:
            end = (random.randrange(size), random.randrange(size))

        board[start[0]][start[1]] = START
        board[end[0]][end[1]] = END

        # Place walls
        for r in range(size):
            for c in range(size):
                if (r,c) not in (start, end) and random.random() < max_wall_prob:
                    board[r][c] = WALL

        # BFS to check solvability & record path
        prev = {start: None}
        q = deque([start])
        found = False
        while q:
            r, c = q.popleft()
            if (r, c) == end:
                found = True
                break
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < size and 0 <= nc < size:
                    if board[nr][nc] != WALL and (nr, nc) not in prev:
                        prev[(nr, nc)] = (r, c)
                        q.append((nr, nc))

        if found:
            # Reconstruct path
            target = [row[:] for row in board]
            cur = end
            while cur != start:
                if cur != end:
                    target[cur[0]][cur[1]] = PATH
                cur = prev[cur]
            return torch.tensor(board, dtype=torch.long), torch.tensor(target, dtype=torch.long)
        # else: loop again

def generate_unique_samples(count, board_size=4, max_wall_prob=0.3):
    """Generate unique board samples with deduplication."""
    inputs, targets = [], []
    seen_boards = set()

    while len(inputs) < count:
        board, target = generate_board(board_size, max_wall_prob)
        # Use board tensor as hashable key for deduplication
        board_key = tuple(board.flatten().tolist())

        if board_key not in seen_boards:
            seen_boards.add(board_key)
            inputs.append(board.flatten())
            targets.append(target.flatten())

    return torch.stack(inputs), torch.stack(targets)

def build_datasets(params: BoardPathParameters):
    """Build train and validation datasets with guaranteed no overlap."""
    print(f"Generating {params.train_count + params.val_count} unique samples...")
    inputs, targets = generate_unique_samples(
        params.train_count + params.val_count,
        params.board_size,
        params.wall_prob
    )

    train_inputs = inputs[:params.train_count]
    train_targets = targets[:params.train_count]
    val_inputs = inputs[params.train_count:]
    val_targets = targets[params.train_count:]

    return (
        TensorDataset(train_inputs, train_targets),
        TensorDataset(val_inputs, val_targets)
    )

def boardpath_summary(params: BoardPathParameters) -> None:
    print("BoardPath Dataset Parameters:")
    print("-" * 31)
    print(f"{'board_size':<20} {params.board_size:>10}")
    print(f"{'seq_len':<20} {params.board_size * params.board_size:>10}")
    print(f"{'train_count':<20} {params.train_count:>10}")
    print(f"{'val_count':<20} {params.val_count:>10}")
    print(f"{'wall_prob':<20} {params.wall_prob:>10}")
    print(f"{'vocab_cnt':<20} {get_vocab_cnt():>10}")
    print()

if __name__ == '__main__':
    params = BoardPathParameters(board_size=4, train_count=5000, val_count=1000, wall_prob=0.3)
    boardpath_summary(params)
    train_dataset, val_dataset = build_datasets(params)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for x_bs, y_bs in train_loader:
        print("Input board:")
        print(x_bs[0].view(params.board_size, params.board_size))
        print("Target board:")
        print(y_bs[0].view(params.board_size, params.board_size))
        break
