import numpy as np

def create_board(n):
    return np.zeros((n, n), dtype=int)  

def print_board(board):
    for row in board:
        print(" ".join(['X' if cell == 1 else 'O' if cell == 2 else '.' for cell in row]))

def check_winner(board, player, n):
    for i in range(n):
        if all(board[i, :] == player): 
            return True
        if all(board[:, i] == player):  
            return True
    
    if all(board[i, i] == player for i in range(n)): 
        return True
    if all(board[i, n - 1 - i] == player for i in range(n)):  
        return True
    
    return False

def draw(board):
    return not (board == 0).any()  

def tictactoe():
    n = int(input("Enter the size of the board (e.g., 3 for 3x3): ")) 
    board = create_board(n) 
    current_player = 1  

    while True:
        print_board(board) 
        print(f"Player {current_player}'s turn")

        row = int(input(f"Enter row (0-{n-1}): "))
        col = int(input(f"Enter column (0-{n-1}): "))

        if row < 0 or row >= n or col < 0 or col >= n:
            print("Invalid input. Please try again.")
            continue

        if board[row, col] == 0:  
            board[row, col] = current_player  
        else:
            print("Cell is already taken. Try again.")
            continue  

        if check_winner(board, current_player, n):
            print_board(board)
            print(f"Player {current_player} wins!")
            break  

        if draw(board):
            print_board(board)
            print("It's a draw!")
            break  
        current_player = 2 if current_player == 1 else 1

tictactoe()
