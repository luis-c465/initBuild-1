import math
import os
import random
from collections import deque
from itertools import count
from typing import List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import nptyping as npt
import torch
import torch as T
import torch.nn as nn
from torch import optim

from src.ai.model import DQN, DuelDQN
from src.ai.utils import ReplayMemory, Transition, VideoRecorder
from src.ai.wrapper import AtariWrapper

obs_type = npt.NDArray[npt.Shape["84, 84"], npt.Number]


class Trainer:
    GAMMA = 0.99  # bellman function
    EPS_START = 1
    EPS_END = 0.05
    EPS_DECAY = 50000
    WARMUP = 1000  # don't update net until WARMUP steps
    MEMORY_SIZE = 50_000

    steps_done = 0
    n_epochs = 0
    eps_threshold = EPS_START

    env: gym.Env[obs_type, int]

    n_action: int  # number of actions for the env

    policy_net: nn.Module
    target_net: nn.Module
    optimizer: optim.Optimizer
    memory: ReplayMemory

    log_dir: str
    video: VideoRecorder

    def __init__(
        self,
        env_name: str,
        model="dqn",
        device="cpu",
        lr=2.5e-4,
        epochs=10001,
        batch_size=32,
        use_ddqn=False,
        eval_freq=500,
        logging=False,
        **kwargs: float,
    ) -> None:
        self.env_name = env_name
        self.model = model
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_ddqn = use_ddqn
        self.eval_freq = eval_freq

        self.logging = logging

        for k, v in kwargs.items():
            setattr(self, k.upper(), v)

        self.env = gym.make(env_name)
        self.env = AtariWrapper(self.env)
        self.n_action = self.env.action_space.n  # type: ignore

        if use_ddqn:
            methodname = f"double_{model}"
        else:
            methodname = model
        self.log_dir = os.path.join(f"log_{env_name.split('/')[-1]}", methodname)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.video = VideoRecorder(self.log_dir)

        if model == "dqn":
            self.policy_net = DQN(in_channels=4, n_actions=self.n_action).to(device)
            self.target_net = DQN(in_channels=4, n_actions=self.n_action).to(device)
        else:
            self.policy_net = DuelDQN(in_channels=4, n_actions=self.n_action).to(device)
            self.target_net = DuelDQN(in_channels=4, n_actions=self.n_action).to(device)
        # let target model = model
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory(self.MEMORY_SIZE)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)

    def warm_up(self) -> None:
        """
        Warm up the memory with random actions

        Recommended to run before training
        """
        if self.logging:
            print("Warming up...")

        warmupstep = 0
        for _epoch in count():
            obs, _info = self.env.reset()  # (84,84)
            obs = torch.from_numpy(obs).to(self.device)  # (84,84)
            # stack four frames together, hoping to learn temporal info
            obs = torch.stack((obs, obs, obs, obs)).unsqueeze(0)  # (1,4,84,84)

            # step loop
            for _step in count():
                warmupstep += 1
                # take one step
                action = torch.tensor([[self.env.action_space.sample()]]).to(
                    self.device
                )
                next_obs, reward, terminated, truncated, _info = self.env.step(
                    action.item()  # type: ignore
                )
                done = terminated or truncated

                # convert to tensor
                reward = torch.tensor([reward], device=self.device)  # (1)
                done = torch.tensor([done], device=self.device)  # (1)
                next_obs = torch.from_numpy(next_obs).to(self.device)  # (84,84)
                next_obs = torch.stack(
                    (next_obs, obs[0][0], obs[0][1], obs[0][2])
                ).unsqueeze(
                    0
                )  # (1,4,84,84)

                # store the transition in memory
                self.memory.push(obs, action, next_obs, reward, done)

                # move to next state
                obs = next_obs

                if done:
                    break

            if warmupstep > self.WARMUP:
                break

    def epoch(self) -> Tuple[float, float]:
        """Does one epoch of training

        Returns total reward and total loss for the epoch
        """

        obs, _info = self.env.reset()  # (84,84)
        obs = torch.from_numpy(obs).to(self.device)  # (84,84)
        # stack four frames together, hoping to learn temporal info
        obs = torch.stack((obs, obs, obs, obs)).unsqueeze(0)  # (1,4,84,84)

        total_loss = 0.0
        total_reward = 0

        done = False
        while not done:
            reward, loss, done = self.step(obs)

            total_reward += reward
            total_loss += loss

        self.n_epochs += 1
        return total_reward, total_loss

    def step(
        self,
        obs: T.Tensor,
    ) -> Tuple[float, float, bool]:
        """Does a single step of an epoch

        Returns reward and loss for the step
        """
        # take one step
        action = self.select_action(obs)
        next_obs, reward, terminated, truncated, _info = self.env.step(action.item())  # type: ignore
        done = terminated or truncated
        n_done = done

        i_reward = reward

        # convert to tensor
        reward = torch.tensor([reward], device=self.device)  # (1)
        done = torch.tensor([done], device=self.device)  # (1)
        next_obs = torch.from_numpy(next_obs).to(self.device)  # (84,84)
        next_obs = torch.stack((next_obs, obs[0][0], obs[0][1], obs[0][2])).unsqueeze(
            0
        )  # (1,4,84,84)

        # store the transition in memory
        self.memory.push(obs, action, next_obs, reward, done)

        # move to next state
        obs = next_obs

        # train
        self.policy_net.train()
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(
            *zip(*transitions)
        )  # batch-array of Transitions -> Transition of batch-arrays.
        state_batch = torch.cat(batch.state)  # (bs,4,84,84)
        next_state_batch = torch.cat(batch.next_state)  # (bs,4,84,84)
        action_batch = torch.cat(batch.action)  # (bs,1)
        reward_batch = torch.cat(batch.reward).unsqueeze(1)  # (bs,1)
        done_batch = torch.cat(batch.done).unsqueeze(1)  # (bs,1)

        # Q(st,a)
        state_qvalues = self.policy_net(state_batch)  # (bs,n_actions)
        selected_state_qvalue = state_qvalues.gather(1, action_batch)  # (bs,1)

        with torch.no_grad():
            # Q'(st+1,a)
            next_state_target_qvalues = self.target_net(
                next_state_batch
            )  # (bs,n_actions)
            if self.use_ddqn:
                # Q(st+1,a)
                next_state_qvalues = self.policy_net(next_state_batch)  # (bs,n_actions)
                # argmax Q(st+1,a)
                next_state_selected_action = next_state_qvalues.max(1, keepdim=True)[
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
            next_state_selected_qvalue * self.GAMMA * ~done_batch + reward_batch
        )  # (bs,1)

        # optimize
        criterion = nn.SmoothL1Loss()
        loss = criterion(selected_state_qvalue, tdtarget)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # let self.target_net = policy_net every 1000 steps
        if self.steps_done % 1000 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.logging and done and self.n_epochs % self.eval_freq == 0:
            self.log()

        return i_reward, loss.item(), n_done  # type: ignore

    def log(self):
        """Logs the performance of the model, records a video and saves the model

        If logging is set to true this is called every 500 epochs by default
        """
        with torch.no_grad():
            self.video.reset()
            evalenv = gym.make(self.env_name)
            evalenv = AtariWrapper(evalenv, video=self.video)
            obs, info = evalenv.reset()  # type: ignore
            obs = torch.from_numpy(obs).to(self.device)
            obs = torch.stack((obs, obs, obs, obs)).unsqueeze(0)
            evalreward = 0
            self.policy_net.eval()
            for _ in count():
                action = self.policy_net(obs).max(1)[1]
                (
                    next_obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = evalenv.step(action.item())
                evalreward += reward  # type: ignore
                next_obs = torch.from_numpy(next_obs).to(self.device)  # (84,84)
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
                        obs, info = evalenv.reset()  # type: ignore
                        obs = torch.from_numpy(obs).to(self.device)
                        obs = torch.stack((obs, obs, obs, obs)).unsqueeze(0)
            evalenv.close()
            self.video.save(f"{self.n_epochs}.mp4")

            self.save()
            print(f"Eval epoch {self.n_epochs}: Reward {evalreward}")

    def save(self, file: str | None = None) -> bool:
        """Saves the model and other data to a file

        Returns if the save was successful
        """
        file = file if file is not None else self.log_dir
        try:
            torch.save(
                self.policy_net,
                os.path.join(file, f"model{self.n_epochs}.pth"),
            )
            return True
        except:
            return False

    def select_action(self, state: T.Tensor) -> T.Tensor:
        """
        epsilon greedy
        - epsilon: choose random action
        - 1-epsilon: argmax Q(a,s)

        Input: state shape (1,4,84,84)

        Output: action shape (1,1)
        """
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1.0 * self.steps_done / self.EPS_DECAY
        )
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]]).to(self.device)

    def close(self):
        """Closes the environment and frees up resources

        You should call this once training is done to free up resources
        But make sure to save before because this will delete the model and any other data
        """

        try:
            self.env.close()
            del self.env
            del self.policy_net
            del self.target_net
            del self.optimizer
            del self.memory
            del self.video
        except:
            pass

    def save_and_close(self):
        """Saves and closes the environment"""

        self.save()
        self.close()

    def training_loop(
        self,
    ) -> Tuple[Tuple[List[float], List[float]], Tuple[List[float], List[float]]]:
        """Runs the full training loop

        Returns (rewardList, lossList), (avgrewardlist, avglosslist)
        """
        self.warm_up()

        if self.logging:
            print("Training...")

        rewardList = []
        lossList = []
        rewarddeq = deque([], maxlen=100)
        lossdeq = deque([], maxlen=100)
        avgrewardlist = []
        avglosslist = []

        for n_epoch in range(self.epochs):
            total_reward, total_loss = self.epoch(n_epoch, self.steps_done)

            rewardList.append(total_reward)
            lossList.append(total_loss)
            rewarddeq.append(total_reward)
            lossdeq.append(total_loss)
            avgreward = sum(rewarddeq) / len(rewarddeq)
            avgloss = sum(lossdeq) / len(lossdeq)
            avglosslist.append(avgloss)
            avgrewardlist.append(avgreward)

            if self.logging:
                print(
                    f"Epoch {n_epoch}: Loss {total_loss:.2f}, Reward {total_reward}, Avgloss {avgloss:.2f}, Avgreward {avgreward:.2f}, Epsilon {self.eps_threshold:.2f}, TotalStep {self.steps_done}"
                )

        self.env.close()

        if self.logging != None:
            # plot loss-epoch and reward-epoch
            plt.figure(1)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.plot(range(len(lossList)), lossList, label="loss")
            plt.plot(range(len(lossList)), avglosslist, label="avg")
            plt.legend()
            plt.savefig(os.path.join(self.log_dir, "loss.png"))

            plt.figure(2)
            plt.xlabel("Epoch")
            plt.ylabel("Reward")
            plt.plot(range(len(rewardList)), rewardList, label="reward")
            plt.plot(range(len(rewardList)), avgrewardlist, label="avg")
            plt.legend()
            plt.savefig(os.path.join(self.log_dir, "reward.png"))

        return (rewardList, lossList), (avgrewardlist, avglosslist)

    @staticmethod
    def train(
        env_name: str,
        model="dqn",
        device="cpu",
        lr=2.5e-4,
        epochs=10001,
        batch_size=32,
        use_ddqn=False,
        eval_freq=500,
        logging=False,
        **kwargs: float,
    ):
        """Creates a new trainer object, runs the training loop and returns the data and the trainer object"""
        trainer = Trainer(
            env_name,
            model,
            device,
            lr,
            epochs,
            batch_size,
            use_ddqn,
            eval_freq,
            logging,
            **kwargs,
        )

        data = trainer.training_loop()

        return data, trainer
