from argparse import ArgumentParser
from pathlib import Path
import gymnasium as gym
import imitation
import pickle
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from datetime import datetime


def load_demonstration(demo_path):

    with demo_path.open("rb") as fp:
        demo_db = pickle.load(fp)

    return demo_db


class DemoDataset(torch.utils.data.Dataset):

    def __init__(self, demo_log):
        self.n_SA_pairs= len(demo_log)
        self.states, self.actions = self.strip_to_SA_pairs(demo_log)

    def __len__(self):
        return self.n_SA_pairs

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.states[idx]
        labels = self.actions[idx]

        return sample, labels

    def strip_to_SA_pairs(self, demo_log):

        states = []
        actions = []

        for item in demo_log:
            state = item[0]
            flat_state = []
            [flat_state.extend(row) for row in state]
            flat_state = list(map(int, flat_state)) # Convert from bool to int
            flat_state = torch.tensor(flat_state, dtype=torch.float32)
            action = torch.tensor(item[-1], dtype=torch.int64)

            states.append(flat_state)
            actions.append(action)

        return states, actions


def get_spaces(conf):

    width = conf.tetris_settings.width
    height = conf.tetris_settings.height

    action_space = gym.spaces.Discrete(n=7) #n=7 for the 7 actions, each a number in [0;6]

    observation_space = gym.spaces.MultiBinary(n=(width, height))

    return action_space, observation_space


class ff_model(nn.Module):
    def __init__(self, conf):
        super(ff_model, self).__init__()
        width = conf.tetris_settings.width
        height = conf.tetris_settings.height
        input_size = width * height
        output_size = 7
        self.l1 = nn.Linear(input_size, 100)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(100, 10)
        self.l3 = nn.Linear(10, output_size)

    def forward(self, x):
        output = self.l1(x)
        output = self.relu(output)
        output = self.l2(output)
        output = self.relu(output)
        output = self.l3(output)
        return output


def train_model(conf, dataloader):

    model = ff_model(conf)

    # TODO: there is a significant chance that the amount of NOOPS will result in
    #  difficulty converging and we may need to set weights.
    loss = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr= conf.training.lr)
    epochs = conf.training.epochs

    loss_list = []
    for j in range(epochs):
        for i, (x_train, y_train) in enumerate(dataloader):  # prediction
            y_pred = model(x_train)

            # calculating loss
            cost = loss(y_pred, y_train)

            # backprop
            opt.zero_grad()
            cost.backward()
            opt.step()
        if j % 50 == 0:
            print(cost)
            loss_list.append(cost)

    return model, loss_list


def save_model(model, conf, output_path):

    time_now = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    model_path = output_path / f"ff_model_{time_now}"
    torch.save(model.state_dict, model_path)
    with (output_path / f"conf_{time_now}.yaml").open("w") as fp:
        OmegaConf.save(conf, fp)
def __read_conf():

    conf_file = "config.yaml"
    participant_setting_file = "participant_settings.yaml"

    conf = OmegaConf.load(conf_file)

    print(f" Loaded configs: {conf}")

    return conf

def __parse_args():
    parser = ArgumentParser()

    parser.add_argument("demo_path")
    parser.add_argument("output_path")

    args = parser.parse_args()

    return Path(args.demo_path), Path(args.output_path)


if __name__ == "__main__":

    demo_pickle_path, output_path = __parse_args()
    conf = __read_conf()
    # Load the demo log as a list of (state, reward, done, action) tuples
    demo_db = load_demonstration(demo_pickle_path)
    dataset = DemoDataset(demo_db)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, shuffle=True, batch_size=conf.training.batch_size)

    model, loss_list = train_model(conf, dataloader)

    loss_list = [item.detach().numpy() for item in loss_list]
    plt.plot(loss_list)
    plt.show()

    save_model(model, conf=conf, output_path=output_path)
