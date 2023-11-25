"""
Written with reference to user_engine.py script in tetrisRL repo
"""
from pathlib import Path
from omegaconf import OmegaConf

from tamer import TAMER 

# added imports for tetrisRL
import curses
import numpy as np
import pickle
import os
from engine import TetrisEngine
from datetime import datetime

import torch
import time


def play_game_TAMER(speed, duration):

    # Run a TAMER session for N minutes at speed <speed>

    print(f"scalar feedback at speed {speed}")
    stdscr = curses.initscr()
    width, height = 10, 20 # standard tetris friends rules
    env = TetrisEngine(width, height)
    
    use_cuda = torch.cuda.is_available()
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

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

    # initialize our model
    tamer = TAMER(speed)
    model_state = FloatTensor(env.clear()[None,None,:,:])
    while curr - start < duration:
        curr = time.time()
        key = stdscr.getch()
        
        human_reward = 0

        if key == ord("z"):
            human_reward = 1
        elif key == ord("x"):
            human_reward = -1

        # Game step
        action = int(tamer.select_action(model_state))
        state, reward, done = env.step(action)
        model_state = FloatTensor(state[None,None,:,:])
        reward_sum += reward
        db.append((state, reward, done, action))
        
        # Render
        stdscr.clear()
        stdscr.addstr(str(env))

        tamer.training_step(model_state, action, human_reward)

        # can remove these if desired
        stdscr.addstr('\ncumulative reward: ' + str(reward_sum))
        stdscr.addstr('\ntime: ' + str(round(curr - start, 2)) + ' seconds')
        if human_reward == 1:
            stdscr.addstr('\nPOSITIVE REWARD')
        elif human_reward == -1:
            stdscr.addstr('\nNEGATIVE REWARD')
    
    terminate_game(stdscr)
    return db

def play_game_demonstration(speed, duration, conf):

    # Run a normal tetris session for N minutes at speed <speed>
    # An agent is learning something from this when it's demonstration time
    stdscr = curses.initscr()
    width, height = conf.tetris_settings.width, conf.tetris_settings.height
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

    while curr - start < duration:
        curr = time.time()
        action = 6
        key = stdscr.getch()

        if key == -1: # No key pressed
            action = 6
        elif key == ord('a'): # Shift left
            action = 0
        elif key == ord('d'): # Shift right
            action = 1
        elif key == ord('w'): # Hard drop
            action = 2
        elif key == ord('s'): # Soft drop
            action = 3
        elif key == ord('q'): # Rotate left
            action = 4
        elif key == ord('e'): # Rotate right
            action = 5

        # Game step
        state, reward, done = env.step(action)
        reward_sum += reward
        db.append((state, reward, done, action))
        # db = pd.concat([df, pd.DataFrame({"state": state, "reward": reward, "done": done, "action": action})], ignore_index=True)

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

def run_all_experiments(log_path, conf, experiment_settings):

    slow_speed = conf.speeds.slow
    normal_speed = conf.speeds.medium
    fast_speed = conf.speeds.fast

    # Game duration in minutes
    game_duration = conf.game_length

    # Experiment order is a list of (type, speed) tuples
    experiment_order = []

    for modality in experiment_settings.modality_order:
        for speed in experiment_settings[modality]:
            experiment_order.append((modality, speed))

    # Run tetris for N minutes so people get used to it.
    input("Hello! Welcome to a pilot study on human-in-the-loop reinforcement learning modalities. Press enter for practice time!")
    warmup_log_db = play_game_demonstration(normal_speed, game_duration, conf)
    with (log_path / f"warmup_game.pickle").open("wb") as fp:
        pickle.dump(warmup_log_db, fp)

    input("Practice session over! When you are ready to proceed, press enter.")
    for modality, speed in experiment_order:

        if modality == "pref":

            print(f"ABOUT TO START A SCALAR FEEDBACK GAAAAMEEEE")
            feedback_log_db = play_game_TAMER(conf.speeds[speed], duration=game_duration)
            with (log_path / f"feedback_{speed}.pickle").open("wb") as fp:
                pickle.dump(feedback_log_db, fp)
        elif modality == "demo":

            print(f"ABOUT TO START A DEMO GAAAAAAMEEEEE")

            demo_log_db = play_game_demonstration(conf.speeds[speed], duration=game_duration, conf=conf)
            print(demo_log_db[-1])
            with (log_path / f"demo_{speed}.pickle").open("wb") as fp:
                pickle.dump(demo_log_db, fp)
        else:
            raise ValueError("Experiment modality must be pref or demo.")

        print(f"Thank you! You've just completed a {modality} game at the {speed} speed.")
        print(f"Type 'next' to continue to the next game!")

        while True:

            user_input = input()
            if user_input == "next":
                break

    with (log_path / "conf.yaml").open("w") as fp:
        OmegaConf.save(conf, fp)


def __read_conf():

    conf_file = "config.yaml"
    participant_setting_file = "participant_settings.yaml"

    conf = OmegaConf.load(conf_file)
    participant_settings = OmegaConf.load(participant_setting_file)

    print(f" Loaded configs: {conf}")
    print(f" Loaded participants: {participant_settings}")

    return conf, participant_settings


def __parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("log_dir")
    parser.add_argument("participant_id", type=int)

    args = parser.parse_known_args()

    print(type(args[0].log_dir), type(args[0].participant_id))
    
    return Path(args[0].log_dir), args[0].participant_id

if __name__ == "__main__":
    log_path, participant_id = __parse_args()

    conf, participant_settings = __read_conf()

    experiment_settings = participant_settings[participant_id]

    print(f"This participant's settings: {experiment_settings}")

    if not log_path.exists():
        log_path.mkdir()
    time_now = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    participant_folder_path = log_path / f"participant_{participant_id}_{time_now}"
    if not participant_folder_path.exists():
        participant_folder_path.mkdir()

    run_all_experiments(log_path=participant_folder_path, conf=conf, experiment_settings=experiment_settings)