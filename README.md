# modalities


This repository contains the code used to run a pilot study to compare the user experience of human-in-the-loop machine learning modalities across changing environmental characteristics.

To run the study run ` python tetris_modalities.py [log_dir] [participant_id]`, which outputs logs, configurations and models to log_dir. Ensure that participant_id exists in `participant_settings.yaml`

This code uses pytorch as well as dependancies listed in requirements.txt

Parts of this repository are adapted from [Tetris RL](https://github.com/jaybutera/tetrisRL)
