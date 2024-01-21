"""
Created on Fri May 19 16:07:55 2023

@author: YangYing
"""
from asyncio import start_server
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import copy
from re import M
import torch.nn.functional as F
import numpy as np
import random
import time
from routingEnv import routingEnv
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import traceback
import tqdm
import csv
version = 2

#################################################################################

################################## set device ##################################

print("============================================================================================")

# set device to cpu or cuda2
device = torch.device('cpu')

if (torch.cuda.is_available()):
    device = torch.device('cuda:5')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

print("============================================================================================")


################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.pathNums = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.pathNums[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init
        self.state_dim = state_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(self.state_dim, action_dim, has_continuous_action_space, self.action_std).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        self.optimizer_actor = torch.optim.Adam(self.policy.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.policy.critic.parameters(), lr=lr_critic)
        self.policy_old = ActorCritic(self.state_dim, action_dim, has_continuous_action_space, self.action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def decay_action_std(self, action_std_decay_rate, min_action_std):
        # print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        # print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                action, action_logprob = self.policy_old.act(state)

            return action, action_logprob

        else:
            with torch.no_grad():
                action, action_logprob = self.policy_old.act(state)

            return action.item(), action_logprob

    def addBufferSA(self, action, state, logprob):
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(logprob)
    
    def addBufferRD(self, reward, done): 
        self.buffer.rewards.append((reward))
        self.buffer.is_terminals.append(done)

    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            c_loss = self.MseLoss(state_values, rewards)  
            self.optimizer_critic.zero_grad()
            c_loss.mean().backward()
            self.optimizer_critic.step()

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO 0.01

            a_loss = -torch.min(surr1, surr2) - 0.1 * dist_entropy

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            # take gradient step
            self.optimizer_actor.zero_grad()
            a_loss.mean().backward()
            self.optimizer_actor.step()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear() 
            
        return loss.mean().detach().cpu().numpy(), a_loss.mean().detach().cpu().numpy(), c_loss.mean().detach().cpu().numpy(), dist_entropy.mean().detach().cpu().numpy(), self.policy.critic.state_dict()
                    
    def update_dict(self, global_dict):
        
        self.policy.critic.load_state_dict(global_dict)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()


    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

def write_information(states, actions, rewards, index):
    with open('state.csv', 'a+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the data
        writer.writerows([str(index)] + states)

    with open('action.csv', 'a+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the data
        writer.writerows([str(index)] + actions)

    with open('reward.csv', 'a+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the data
        reward = []
        reward.append(rewards)
        writer.writerows([str(index)] + reward)

def FedAvg(thetas):
    w_avg = copy.deepcopy(thetas[0])
    for k in w_avg.keys():
        for i in range(1, len(thetas)):
            w_avg[k] += thetas[i][k]
        w_avg[k] = torch.div(w_avg[k], len(thetas))
    return w_avg


def main():
    summary_writer = SummaryWriter()
    num_iteration = 40000
    num_step = 50
    env = routingEnv(621)

    
    action_dim = env.action_dim
    lr_actor = 2e-5
    lr_critic = 1e-4
    gamma = 0.99
    K_epochs = 10
    eps_clip = 0.2
    action_std_decay_rate = 0.01
    min_action_std = 0.01
    save_model_freq = 5000 * num_step

    print("============================================================================================")
    agents = []
    for i in range(env.node_num):
        state_dim = 6
        agents.append(PPO(int(state_dim), action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, True, action_std_init=0.6))
    
    step = 0
    for i in tqdm.tqdm(range(num_iteration)):
        episode = i
        states = env.reset(episode)
        ep_total_reward = 0
        ep_local_reward = 0
        local_r_agent = np.zeros(env.node_num)
        local_rd_agent = np.zeros(env.node_num)
        local_rp_agent = np.zeros(env.node_num)

        ep_rd = 0
        ep_rp = 0
        for j in range(num_step):
            for node in range(env.node_num):
                state = states[node].getLocalState()[0]
                print("state", state)
                agent = agents[node]
                state = torch.FloatTensor(state).to(device)
                action, action_prob = agent.select_action(state)
                nor_action = action + 1
                nor_action = torch.clamp(nor_action, 0, 2)
                act = nor_action.detach().cpu().numpy()
                print("action", act)
                agent.addBufferSA(action, state, action_prob)
                env.store_action(node, act)
            
            states__, st_rd, st_rp, done = env.step()
            ep_rd += np.mean(st_rd)
            ep_rp += np.mean(st_rp)
            if j == num_step - 1:
                done = True
            for node in range(env.node_num):
                reward = states__[node].getRewards()
                print("reward", reward[0][0])
                local_reward = states__[node].getLocalRewards()
                ep_local_reward += local_reward[0][0]
                # local reward for each agent
                local_r_agent[node] += local_reward[0][0]
                local_rd_agent[node] += st_rd[node]
                local_rp_agent[node] += st_rp[node]
                for re in range(len(reward)):
                    ep_total_reward += reward[re][0]
                    agent = agents[node]
                    agent.addBufferRD(reward[0][0], done)
            states_ = env.get_state()
        
            if (step+1) % (num_step) == 0:
                total_a_loss = 0
                total_c_loss = 0
                total_entropy = 0
                a_loss_agent = []
                c_loss_agent = []
                for node in range(env.node_num):
                    if len(agents[node].buffer.states) == 0:
                        continue
                    loss, a_loss, c_loss, entropy, theta = agents[node].update()
                    total_a_loss += a_loss
                    total_c_loss += c_loss
                    total_entropy += entropy
                    a_loss_agent.append(a_loss)
                    c_loss_agent.append(c_loss)
                a_loss_var = np.var(a_loss_agent, ddof=1)
                c_loss_var = np.var(c_loss_agent, ddof=1)
                summary_writer.add_scalar('a_loss_var', a_loss_var, i)
                summary_writer.add_scalar('c_loss_var', c_loss_var, i)
                summary_writer.add_scalar('a_loss', total_a_loss/env.node_num, i)
                summary_writer.add_scalar('c_loss', total_c_loss/env.node_num, i)
                summary_writer.add_scalar('entropy', total_entropy/env.node_num, i)
            if (step+1) % (10*1600) == 0:
                for node in range(env.node_num):
                    if agents[node].action_std > 0.1:
                        agents[node].decay_action_std(action_std_decay_rate, min_action_std)
                    else:
                        agents[node].decay_action_std(0.002, 0.01)

            # save model weights
            # if (step + 1) % save_model_freq == 0:
            #     # print("--------------------------------------------------------------------------------------------")
            #     # print("saving model at : " + checkpoint_path)
            #     record = step + 1
            #     checkpoint_path = "PPO_seed"
            #     checkpoint_path = checkpoint_path + '_' + str(record) + '.pth'
            #     agent.save(checkpoint_path)

            step += 1
            states = states_
        ep_total_reward /= num_step
        ep_total_reward /= env.node_num
        ep_local_reward /= num_step
        ep_local_reward /= env.node_num
        ep_rd /= num_step
        ep_rp /= num_step

        local_r_agent /= num_step
        reward_var = np.var(local_r_agent, ddof=1)
        local_rd_agent /= num_step
        local_rp_agent /= num_step
        rd_var = np.var(local_rd_agent, ddof=1)
        rp_var = np.var(local_rp_agent, ddof=1)

        print("Episode: \t{} Average Reward: \t{:0.6f}".format(i, ep_total_reward))
        summary_writer.add_scalar('total_reward_change 15 30 zif standard round2', ep_total_reward, i)
        summary_writer.add_scalar('reward_var', reward_var, i)
        summary_writer.add_scalar('rd_var', rd_var, i)
        summary_writer.add_scalar('rp_var', rp_var, i)
        summary_writer.add_scalar('local_reward', ep_local_reward, i)
        summary_writer.add_scalar('ep_rd', ep_rd, i)
        summary_writer.add_scalar('ep_rp', ep_rp, i)


if __name__ == '__main__':
        main()
