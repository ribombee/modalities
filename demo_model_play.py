from train_on_demo import ff_model
import curses
from engine import TetrisEngine
import torch
from torch.autograd import Variable
import time

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def play_game_agent(model, speed, duration):

    # Run a TAMER session for N minutes at speed <speed>

    #print(f"scalar feedback at speed {speed}")
    stdscr = curses.initscr()
    width, height = 10, 20 # standard tetris friends rules
    env = TetrisEngine(width, height)

    # init timer
    start = time.time()
    curr = time.time()

    # init our game
    initialize_game(speed, stdscr)
    stdscr.clear()
    env.clear()

    # init render
    stdscr.addstr(str(env))

    # store information about playthrough
    # df = pd.DataFrame(columns=["state", "reward", "done", "action"])
    db = []
    reward_sum = 0

    # Sends a single NOOP action at the start of the game, I guess?
    # If there's some way to grab the initial state idk
    model_state = FloatTensor(env.clear()[None,None,:,:])
    model_state = torch.flatten(model_state)
    # initialize our model
    while curr - start < duration:
        curr = time.time()
        key = stdscr.getch()
        
        # Game step
        # TODO: ensure this is block works with model input
        predictions = model(Variable(model_state).type(FloatTensor))

        # print(predictions)
        # action = predictions.data.max(1)[1].view(1, 1)
        action = int(torch.argmax(predictions).item())

        state, reward, done = env.step(action)
        model_state = FloatTensor(state[None,None,:,:])
        model_state = torch.flatten(model_state)
        reward_sum += reward
        db.append((state, reward, done, action))
        
        # Render
        stdscr.clear()
        stdscr.addstr(str(env))

        # can remove these if desired
        stdscr.addstr('\ncumulative reward: ' + str(reward_sum))
        stdscr.addstr('\ntime: ' + str(round(curr - start, 2)) + ' seconds')
    
    terminate_game(stdscr)
    return db

def initialize_game(speed, stdscr):
    # Don't display user input
    curses.noecho()
    # React to keys without pressing enter (700ms delay)
    #curses.halfdelay(speed)
    curses.cbreak()
    # Enumerate keys
    stdscr.keypad(True)
    stdscr.timeout(speed)

def terminate_game(stdscr):
    curses.nocbreak()
    stdscr.keypad(False)
    curses.echo()
    curses.endwin()


def main():
    play_game_agent(100, 1000)


if __name__ == "__main__":
    pass