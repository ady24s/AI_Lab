import numpy as np

rows = 3
column = 3

size = rows * column

def create_board():
    return np.array([[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]])

def print_board(board):
    for row in board:
        print(" ".join(['X' if cell == 1 else 'O' if cell == 2 else '.' for cell in row]))

def check_winner(board, player):
    for i in range(3):
        if all(board[i, :] == player) or all(board[:, i] == player):
            return True

    if all(board[i, i] == player for i in range(3)) or all(board[i, 2 - i] == player for i in range(3)):
        return True
    return False

def draw(board):
    return not (board == 0).any()

def tictactoe():
    board = create_board()  
    current_player = 1  

    while True:
        print_board(board)
        print(f"Player {current_player}'s turn")

        row = int(input("Enter row (0-2): "))
        col = int(input("Enter column (0-2): "))

        if board[row, col] == 0: 
            board[row, col] = current_player  
        else:
            print("Cell is already taken. Try again.")
            continue  

        if check_winner(board, current_player):
            print_board(board)
            print(f"Player {current_player} wins!")
            break  

        if draw(board):  
            print_board(board)
            print("It's a draw!")
            break  

        current_player = 2 if current_player == 1 else 1  
tictactoe()  
