from pathlib import Path
from omegaconf import OmegaConf
# import tetrisRL

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
    # Reset the environment
    '''
    env = tetrisrl.engine.TetrisEngine

    obs = env.clear()

    while True:
        # Get an action from a theoretical AI agent
        action = agent(obs)

        # Sim step takes action and returns results
        obs, reward, done = env.step(action)

        # Done when game is lost
        if done:
            break
    print("tetris warmup")
    '''
    pass

def run_all_experiments(log_path, conf, experiment_settings):

    slow_speed = conf.slow_speed
    normal_speed = conf.slow_speed
    fast_speed = conf.fast_speed

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

    return Path(args.log_dir), args.participant_id

if __name__ == "__main__":

    log_path, participant_id = __parse_args()

    conf, participant_settings = __read_conf()

    experiment_settings = participant_settings[str(participant_id)]

    print(f"This participant's settings: {experiment_settings}")

    run_all_experiments(log_path=log_path, conf=conf, experiment_settings=experiment_settings)