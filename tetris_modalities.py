"""
Written with reference to user_engine.py script in tetrisRL repo
"""

from pathlib import Path
from omegaconf import OmegaConf

# added imports for tetrisRL
import curses
import numpy as np
import os
from engine import TetrisEngine

import time

def play_game_demonstration(speed, duration):

    # Run a demonstration session for N minutes at speed <speed>
    print(f"demo at speed {speed}")
    pass

def play_game_TAMER(speed, duration):

    # Run a TAMER session for N minutes at speed <speed>

    print(f"preferences at speed {speed}")
    pass

def play_game_warmup(speed, duration):

    # Run a normal tetris session for N minutes at speed <speed>
    # No agent is learning anything from this.
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
        db.append((state,reward,done,action))

        # Render
        stdscr.clear()
        stdscr.addstr(str(env))

        # can remove these if desired
        stdscr.addstr('\ncumulative reward: ' + str(reward_sum))
        stdscr.addstr('\ntime: ' + str(round(curr - start, 2)) + ' seconds')
    
    terminate_game(stdscr)

    return

def initialize_game(speed, stdscr):
    # Don't display user input
    curses.noecho()
    # React to keys without pressing enter (700ms delay)
    curses.halfdelay(speed)
    # Enumerate keys
    stdscr.keypad(True)

def terminate_game(stdscr):
    curses.nocbreak()
    stdscr.keypad(False)
    curses.echo()
    curses.endwin()

def run_all_experiments(log_path, conf, experiment_settings):

    slow_speed = conf.speeds.slow
    normal_speed = conf.speeds.normal
    fast_speed = conf.speeds.fast

    # Game duration in minutes
    game_duration = conf.game_length

    # Experiment order is a list of (type, speed) tuples
    experiment_order = []

    for modality in experiment_settings.modality_order:
        for speed in experiment_settings[modality]:
            experiment_order.append((modality, speed))

    print(experiment_order)

    # Run tetris for N minutes so people get used to it.

    play_game_warmup(normal_speed, game_duration)

    for modality, speed in experiment_order:

        if modality == "pref":
            pass
        elif modality == "demo":
            pass
        else:
            raise ValueError("Experiment modality must be pref or demo.")



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

    run_all_experiments(log_path=log_path, conf=conf, experiment_settings=experiment_settings)