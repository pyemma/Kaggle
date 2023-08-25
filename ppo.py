import time

import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

import numpy as np


# Trick 1: vector environment
def make_env(gym_id, capture_video, seed=1001):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            env = gym.wrappers.RecordVideo(env, 'videos', episode_trigger=lambda x: x % 100 == 0, name_prefix=gym_id)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


# Trick 2: orthogonal & constant layer initialization
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# Agent class
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),  # prod is to flatten the observation shape
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),  # why do we use 1 as stdv here instead of sqrt(2)?
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),  # prod is to flatten the observation shape
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            # make sure each action has simialr weights instead of varying too much to take advantage
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )
    
    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

# Parameters
num_steps = 128
num_envs = 4


envs = gym.vector.SyncVectorEnv([make_env("LunarLander-v2", False, 1001 + i) for i in range(num_envs)])
print(f"envs.single_observation_spase.shape: {envs.single_observation_space.shape}")
print(f"envs.single_action_space.n: {envs.single_action_space.n}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = Agent(envs)
# Trick 3: Adam optimizer eps parameter
optimizer = optim.Adam(agent.parameters(), lr=1e-4, eps=1e-5)

# ALGO Logic: Storage

obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
logprobs = torch.zeros((num_steps, num_envs)).to(device)
rewards = torch.zeros((num_steps, num_envs)).to(device)
dones = torch.zeros((num_steps, num_envs)).to(device)
values = torch.zeros((num_steps, num_envs)).to(device)

global_step = 0
start_time = time.time()
next_obs = torch.Tensor(envs.reset()).to(device)
next_done = torch.zeros(num_envs).to(device)

batch_size = num_envs * num_steps
total_timesteps = 50000
num_updates = total_timesteps // batch_size

agent.to(device)

gamma, gae_lambda = 0.99, 0.95

anneal_lr = True
learning_rate = 1e-2
# Trick 4: anneal learning rate
for update in range(1, num_updates+1):
    if anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

    for step in range(0, num_steps):
        global_step += 1 * num_envs
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        # TRY NOT TO MODIFY: execute the game and log data, mainly transfer data between cpu and gpu
        next_obs, reward, done, info = envs.step(action.cpu().numpy())
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

        for item in info:
            if "episode" in item.keys():
                print(f"global_step={global_step}, episode_return={item['episode']['r']}")
                break

    # Trick 5: GAE 
    # TODO understand how GAE works and what's the idea behind
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnontermnal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnontermnal = 1.0 - dones[t+1]
                nextvalues = values[t+1]
            delta = rewards[t] + gamma * nextvalues * nextnontermnal - values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnontermnal * lastgaelam
        returns = advantages + values

        # logic to compute regular advantage using TD error: a = r + gamma * V(s') - V(s)
        # returns = torch.zeros_like(rewards).to(device)
        # for t in reversed(range(num_steps)):
        #     if t == num_steps - 1:
        #         nextnontermnal = 1.0 - next_done
        #         nextvalues = next_value
        #     else:
        #         nextnontermnal = 1.0 - dones[t+1]
        #         nextvalues = values[t+1]
        #     returns[t] = rewards[t] + gamma * nextvalues * nextnontermnal
        # advantages = returns - values

    # flatten the batch
    b_obs = obs.reshape((-1, ) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1, ) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    minbatch_size = batch_size // 4

    # optimize the policy and value network
    b_inds = np.arange(batch_size)
    clipfracs = []  # record how offen the clip is triggered
    for epoch in range(4):
        np.random.shuffle(b_inds)  # shuffle the batch
        for start in range(0, batch_size, minbatch_size):   # Trick 6: minibatch optimization
            end = start + minbatch_size
            mb_inds = b_inds[start:end]
            
            _, newlogprob, entropy, new_values = agent.get_action_and_value(
                b_obs[mb_inds], b_actions.long()[mb_inds]
            )
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            # debug variables
            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > 0.2).float().mean()]

            # Trick 7: advantage normalization
            mb_advantages = b_advantages[mb_inds]
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Trick 8: Policy loss CLIP
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Trick 8: Value loss CLIP
            new_values = new_values.view(-1)
            v_loss_unclipped = (new_values - b_returns[mb_inds]) ** 2
            v_clipped = b_values[mb_inds] + torch.clamp(new_values - b_values[mb_inds], -0.2, 0.2)
            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()

            # v_loss = 0.5 * ((new_values - b_returns[mb_inds]) ** 2).mean()

            # Trick 9: entropy loss
            entropy_loss = entropy.mean()
            loss = pg_loss - 0.01 * entropy_loss + v_loss * 0.5

            # Trick 10: global gradient norm
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
    
    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y