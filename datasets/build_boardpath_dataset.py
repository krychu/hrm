import torch
import random
from collections import deque
from torch.utils.data import Dataset, DataLoader

FLOOR = 0
WALL  = 1
START = 2
END   = 3
PATH  = 4

def get_vocab_cnt():
    return 5

def generate_board_4x4(max_wall_prob=0.3):
    """
    Generate a single valid 4x4 board (input, target).
    input:  ints {0=FLOOR,1=WALL,2=START,3=END}
    target: same but with shortest path cells marked as PATH=4
    """
    size = 4

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

class BoardPathDataset(Dataset):
    def __init__(self, size=4, count=1000, wall_prob=0.3):
        self.size = size
        self.count = count
        self.wall_prob = wall_prob

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        x, y = generate_board_4x4(max_wall_prob=self.wall_prob)
        return x.flatten(), y.flatten()  # each is [S]

if __name__ == '__main__':
    dataset = BoardPathDataset(size=4, count=5000, wall_prob=0.3)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for x_bs, y_bs in loader:
        print("Input board:")
        print(x_bs[0].view(4,4))
        print("Target board:")
        print(y_bs[0].view(4,4))
        break
