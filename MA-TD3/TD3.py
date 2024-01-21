"""
Created on Fri May 19 16:07:55 2023

@author: YangYing
"""
import argparse
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import numpy as np
from routingEnv import routingEnv
from torch.utils.tensorboard import SummaryWriter
import tqdm
import csv
import torch.optim as optim
import os
version = 2

#################################################################################

################################## set device ##################################

print("============================================================================================")

# set device to cpu or cuda
device = torch.device('cpu')

if (torch.cuda.is_available()):
    device = torch.device('cuda:4')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

print("============================================================================================")

parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
parser.add_argument("--env_name", default="Pendulum-v0")  # OpenAI gym environment name， BipedalWalker-v2
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--iteration', default=5, type=int) #test

parser.add_argument('--learning_rate', default=3e-4, type=float)
parser.add_argument('--gamma', default=0.9, type=int) # discounted factor 0.9
parser.add_argument('--capacity', default=50000, type=int) # replay buffer size 50000
# parser.add_argument('--num_iteration', default=500000, type=int) #  num of  games train
parser.add_argument('--batch_size', default=256, type=int) # mini batch size 256
parser.add_argument('--seed', default=1500, type=int)

# optional parameters
parser.add_argument('--num_hidden_layers', default=2, type=int)
parser.add_argument('--sample_frequency', default=256, type=int)
parser.add_argument('--activation', default='Relu', type=str)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=80000, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--policy_noise', default=0.2, type=float)
parser.add_argument('--noise_clip', default=0.5, type=float)
parser.add_argument('--policy_delay', default=2, type=int)
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=2000, type=int)
parser.add_argument('--print_log', default=5, type=int)
args = parser.parse_args()

script_name = os.path.basename(__file__)

directory = './exp' + script_name + args.env_name +'./1'

class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, u, r, y, d = [], [], [], [], []

        for i in ind:
            X, U, R, Y, D = self.storage[i]
            x.append(np.array(X, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            y.append(np.array(Y, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(u), np.array(r).reshape(-1, 1), np.array(y), np.array(d).reshape(-1, 1)

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        a = torch.tanh(self.fc3(a)) * self.max_action
        return a


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        q = F.relu(self.fc1(state_action))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q


class TD3():
    def __init__(self, state_dim, action_dim, max_action, env):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-5)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=5e-5)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=5e-5)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.max_action = max_action
        self.memory = Replay_buffer(args.capacity)
        self.writer = SummaryWriter(directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0
        self.action_dim = action_dim
        self.env = env


    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1)).float().to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, k_epoch):

        if self.num_training % 500 == 0:
            print("====================================")
            print("model has been trained for {} times...".format(self.num_training))
            print("====================================")
        for i in range(k_epoch):
            x, u, r, y, d = self.memory.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            reward = torch.FloatTensor(r).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)

            # Select next action according to target policy:
            next_action = (self.actor_target(next_state))
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * args.gamma * target_Q).detach()

            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()

            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()

            # Delayed policy updates:
            if i % args.policy_delay == 0:
                # Compute actor loss:
                actor_loss = - self.critic_1(state, self.actor(state)).mean()
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(((1 - args.tau) * target_param.data) + args.tau * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(((1 - args.tau) * target_param.data) + args.tau * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(((1 - args.tau) * target_param.data) + args.tau * param.data)

                self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1
        self.num_training += 1
        return loss_Q1.mean().detach().cpu().numpy(), loss_Q2.mean().detach().cpu().numpy(), actor_loss.mean().detach().cpu().numpy()

    def save(self):
        torch.save(self.actor.state_dict(), directory+'actor.pth')
        torch.save(self.actor_target.state_dict(), directory+'actor_target.pth')
        torch.save(self.critic_1.state_dict(), directory+'critic_1.pth')
        torch.save(self.critic_1_target.state_dict(), directory+'critic_1_target.pth')
        torch.save(self.critic_2.state_dict(), directory+'critic_2.pth')
        torch.save(self.critic_2_target.state_dict(), directory+'critic_2_target.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.actor_target.load_state_dict(torch.load(directory + 'actor_target.pth'))
        self.critic_1.load_state_dict(torch.load(directory + 'critic_1.pth'))
        self.critic_1_target.load_state_dict(torch.load(directory + 'critic_1_target.pth'))
        self.critic_2.load_state_dict(torch.load(directory + 'critic_2.pth'))
        self.critic_2_target.load_state_dict(torch.load(directory + 'critic_2_target.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")

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
    num_iteration = 40000
    num_step = 50
    env = routingEnv(621)
    action_dim = env.action_dim
    max_action = float(1)
    var = 0.6

    print("============================================================================================")
    agents = []
    for i in range(env.node_num):
        state_dim = env.state_dim[i]
        agents.append(TD3(int(state_dim), action_dim, max_action, env))

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

        # Noise attenuation
        if (i + 1) % 320 == 0:
            if var >= 0.1:
                var -= 0.01
            elif 0.01 < var < 0.1:
                var -= 0.0025
            else:
                var = 0.01

        for j in range(num_step):
            ptr = (num_step * i + j) % args.capacity
            for node in range(env.node_num):
                state = states[node].getState()[0]
                agent = agents[node]
                print("state", state)
                state_push = np.array(state)
                agent.memory.push((state_push, ))
                state = torch.FloatTensor(state).to(device)
                action = agent.select_action(state)
                action = np.clip(action, -max_action, max_action)
                action = action.reshape(1, -1)
                agent.memory.storage[ptr] += tuple(action)
                nor_action = action + np.random.normal(0, var, size=env.action_dim)
                nor_action = nor_action + 1
                act = np.clip(nor_action, 0, 2)
                print("action", act)
                env.store_action(node, act)

            states__, st_rd, st_rp, done = env.step()
            ep_rd += np.mean(st_rd)
            ep_rp += np.mean(st_rp)
            if j == num_step - 1:
                done = True
            for node in range(env.node_num):
                agent = agents[node]
                reward = states__[node].getRewards()
                print("reward", reward[0][0])
                agent.memory.storage[ptr] += tuple(reward[0])
                local_reward = states__[node].getLocalRewards()
                ep_local_reward += local_reward[0][0]
                # local reward for each agent
                local_r_agent[node] += local_reward[0][0]
                local_rd_agent[node] += st_rd[node]
                local_rp_agent[node] += st_rp[node]
                for re in range(len(reward)):
                    ep_total_reward += reward[re][0]
            states_ = env.get_state()

            for node in range(env.node_num):
                agent = agents[node]
                next_state_push = np.array(states_[node].total_state)
                agent.memory.storage[ptr] += tuple(next_state_push)
                agent.memory.storage[ptr] += tuple([np.float64(done)])
            step += 1
            states = states_

        if i > 1000:
            total_loss_Q1 = 0
            total_loss_Q2 = 0
            total_actor_loss = 0
            for node in range(env.node_num):
                loss_Q1, loss_Q2, actor_loss = agents[node].update(10)
                total_loss_Q1 += loss_Q1
                total_loss_Q2 += loss_Q2
                total_actor_loss += actor_loss
            agent.writer.add_scalar('loss_Q1', total_loss_Q1/env.node_num, i)
            agent.writer.add_scalar('loss_Q2', total_loss_Q2/env.node_num, i)
            agent.writer.add_scalar('actor_loss', total_actor_loss/env.node_num, i)

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

        print("ep_total_reward", ep_total_reward)
        agent.writer.add_scalar('total_reward_change 15 30 zif standard round2', ep_total_reward, global_step=i)
        agent.writer.add_scalar('reward_var', reward_var, i)
        agent.writer.add_scalar('rd_var', rd_var, i)
        agent.writer.add_scalar('rp_var', rp_var, i)
        agent.writer.add_scalar('local_reward', ep_local_reward, global_step=i)
        agent.writer.add_scalar('ep_rd', ep_rd, i)
        agent.writer.add_scalar('ep_rp', ep_rp, i)
        agent.writer.add_scalar('var', var, global_step=i)




if __name__ == '__main__':
        main()
