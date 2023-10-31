import argparse
import math
import multiprocessing
import os
import random
from collections import deque
from itertools import count
from typing import TypedDict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from src.ai.model import DQN, DuelDQN
from src.ai.utils import ReplayMemory, Transition, VideoRecorder
from src.ai.wrapper import AtariWrapper

# parser
parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default="breakout", type=str, help="env name")
parser.add_argument(
    "--model", default="dqn", type=str, choices=["dqn", "dueldqn"], help="dqn model"
)
parser.add_argument("--device", default="cpu", type=str, help="which device to use")
parser.add_argument("--lr", default=2.5e-4, type=float, help="learning rate")
parser.add_argument("--epoch", default=10001, type=int, help="training epoch")
parser.add_argument("--batch-size", default=32, type=int, help="batch size")
parser.add_argument("--ddqn", action="store_true", help="double dqn/dueldqn")
parser.add_argument("--eval-cycle", default=500, type=int, help="evaluation cycle")
args = parser.parse_args()


class Args(TypedDict):
    env_name: str
    model: str
    device: str
    lr: float
    epoch: int
    batch_size: int
    ddqn: bool
    eval_cycle: int


# some hyperparameters
GAMMA = 0.99  # bellman function
EPS_START = 1
EPS_END = 0.05
EPS_DECAY = 50000
WARMUP = 1000  # don't update net until WARMUP steps


def train(
    queue: multiprocessing.Queue,
    event: "multiprocessing.Event",
    env_name: str,
    device="cpu",
    model="dqn",
    lr=2.5e-4,
    n_epoch=10001,
    batch_size=32,
    ddqn=False,
    eval_cycle=500,
) -> None:
    steps_done = 0
    eps_threshold = EPS_START
    # environment
    env = gym.make(env_name)
    env = AtariWrapper(env)

    n_action = env.action_space.n  # pong:6; breakout:4; boxing:18

    # make dir to store result
    if ddqn:
        methodname = f"double_{model}"
    else:
        methodname = model
    log_dir = os.path.join(f"log_{env_name.split('/')[-1]}", methodname)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, "log.txt")

    # video
    video = VideoRecorder(log_dir)

    # create network and target network
    if model == "dqn":
        policy_net = DQN(in_channels=4, n_actions=n_action).to(device)
        target_net = DQN(in_channels=4, n_actions=n_action).to(device)
    else:
        policy_net = DuelDQN(in_channels=4, n_actions=n_action).to(device)
        target_net = DuelDQN(in_channels=4, n_actions=n_action).to(device)
    # let target model = model
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # replay memory
    memory = ReplayMemory(50000)

    # optimizer
    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)

    def select_action(state: torch.Tensor) -> torch.Tensor:
        """
        epsilon greedy
        - epsilon: choose random action
        - 1-epsilon: argmax Q(a,s)

        Input: state shape (1,4,84,84)

        Output: action shape (1,1)
        """
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * steps_done / EPS_DECAY
        )
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]]).to(device)

    # warming up
    print("Warming up...")
    warmupstep = 0
    for n_epoch in count():
        obs, info = env.reset()  # (84,84)
        obs = torch.from_numpy(obs).to(device)  # (84,84)
        # stack four frames together, hoping to learn temporal info
        obs = torch.stack((obs, obs, obs, obs)).unsqueeze(0)  # (1,4,84,84)

        # step loop
        for step in count():
            warmupstep += 1
            # take one step
            action = torch.tensor([[env.action_space.sample()]]).to(device)
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            # convert to tensor
            reward = torch.tensor([reward], device=device)  # (1)
            done = torch.tensor([done], device=device)  # (1)
            next_obs = torch.from_numpy(next_obs).to(device)  # (84,84)
            next_obs = torch.stack(
                (next_obs, obs[0][0], obs[0][1], obs[0][2])
            ).unsqueeze(
                0
            )  # (1,4,84,84)

            # store the transition in memory
            memory.push(obs, action, next_obs, reward, done)

            # move to next state
            obs = next_obs

            if done:
                break

        if warmupstep > WARMUP:
            break

    rewardList = []
    lossList = []
    rewarddeq = deque([], maxlen=100)
    lossdeq = deque([], maxlen=100)
    avgrewardlist = []
    avglosslist = []
    # epoch loop
    for epoch in range(n_epoch):
        obs, info = env.reset()  # (84,84)
        obs = torch.from_numpy(obs).to(device)  # (84,84)
        # stack four frames together, hoping to learn temporal info
        obs = torch.stack((obs, obs, obs, obs)).unsqueeze(0)  # (1,4,84,84)

        total_loss = 0.0
        total_reward = 0

        # step loop
        for step in count():
            # take one step
            action = select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            total_reward += reward
            done = terminated or truncated

            # convert to tensor
            reward = torch.tensor([reward], device=device)  # (1)
            done = torch.tensor([done], device=device)  # (1)
            next_obs = torch.from_numpy(next_obs).to(device)  # (84,84)
            next_obs = torch.stack(
                (next_obs, obs[0][0], obs[0][1], obs[0][2])
            ).unsqueeze(
                0
            )  # (1,4,84,84)

            # store the transition in memory
            memory.push(obs, action, next_obs, reward, done)

            # move to next state
            obs = next_obs

            # train
            policy_net.train()
            transitions = memory.sample(batch_size)
            batch = Transition(
                *zip(*transitions)
            )  # batch-array of Transitions -> Transition of batch-arrays.
            state_batch = torch.cat(batch.state)  # (bs,4,84,84)
            next_state_batch = torch.cat(batch.next_state)  # (bs,4,84,84)
            action_batch = torch.cat(batch.action)  # (bs,1)
            reward_batch = torch.cat(batch.reward).unsqueeze(1)  # (bs,1)
            done_batch = torch.cat(batch.done).unsqueeze(1)  # (bs,1)

            # Q(st,a)
            state_qvalues = policy_net(state_batch)  # (bs,n_actions)
            selected_state_qvalue = state_qvalues.gather(1, action_batch)  # (bs,1)

            with torch.no_grad():
                # Q'(st+1,a)
                next_state_target_qvalues = target_net(
                    next_state_batch
                )  # (bs,n_actions)
                if ddqn:
                    # Q(st+1,a)
                    next_state_qvalues = policy_net(next_state_batch)  # (bs,n_actions)
                    # argmax Q(st+1,a)
                    next_state_selected_action = next_state_qvalues.max(
                        1, keepdim=True
                    )[
                        1
                    ]  # (bs,1)
                    # Q'(st+1,argmax_a Q(st+1,a))
                    next_state_selected_qvalue = next_state_target_qvalues.gather(
                        1, next_state_selected_action
                    )  # (bs,1)
                else:
                    # max_a Q'(st+1,a)
                    next_state_selected_qvalue = next_state_target_qvalues.max(
                        1, keepdim=True
                    )[
                        0
                    ]  # (bs,1)

            # td target
            tdtarget = (
                next_state_selected_qvalue * GAMMA * ~done_batch + reward_batch
            )  # (bs,1)

            # optimize
            criterion = nn.SmoothL1Loss()
            loss = criterion(selected_state_qvalue, tdtarget)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # let target_net = policy_net every 1000 steps
            if steps_done % 1000 == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                # eval
                if epoch % eval_cycle == 0:
                    with torch.no_grad():
                        video.reset()
                        evalenv = gym.make(env_name)
                        evalenv = AtariWrapper(evalenv, video=video)
                        obs, info = evalenv.reset()
                        obs = torch.from_numpy(obs).to(device)
                        obs = torch.stack((obs, obs, obs, obs)).unsqueeze(0)
                        evalreward = 0
                        policy_net.eval()
                        for _ in count():
                            action = policy_net(obs).max(1)[1]
                            (
                                next_obs,
                                reward,
                                terminated,
                                truncated,
                                info,
                            ) = evalenv.step(action.item())
                            evalreward += reward
                            next_obs = torch.from_numpy(next_obs).to(device)  # (84,84)
                            next_obs = torch.stack(
                                (next_obs, obs[0][0], obs[0][1], obs[0][2])
                            ).unsqueeze(
                                0
                            )  # (1,4,84,84)
                            obs = next_obs
                            if terminated or truncated:
                                if info["lives"] == 0:  # real end
                                    break
                                else:
                                    obs, info = evalenv.reset()
                                    obs = torch.from_numpy(obs).to(device)
                                    obs = torch.stack((obs, obs, obs, obs)).unsqueeze(0)
                        evalenv.close()
                        video.save(f"{epoch}.mp4")
                        torch.save(
                            policy_net, os.path.join(log_dir, f"model{epoch}.pth")
                        )
                break

        output = f"Epoch {epoch}: Loss {total_loss:.2f}, Reward {total_reward}, Avgloss {avgloss:.2f}, Avgreward {avgreward:.2f}, Epsilon {eps_threshold:.2f}, TotalStep {steps_done}"
        print(output)

        queue.put((epoch, n_epoch, total_reward, total_loss))
        rewardList.append(total_reward)
        lossList.append(total_loss)
        rewarddeq.append(total_reward)
        lossdeq.append(total_loss)
        avgreward = sum(rewarddeq) / len(rewarddeq)
        avgloss = sum(lossdeq) / len(lossdeq)
        avglosslist.append(avgloss)
        avgrewardlist.append(avgreward)
    env.close()
    event.set()
