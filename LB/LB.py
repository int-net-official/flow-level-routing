"""
Created on Fri May 19 16:07:55 2023

@author: YangYing
"""
import argparse
import torch
from routingEnv import routingEnv
from torch.utils.tensorboard import SummaryWriter
import tqdm
import os
version = 2

#################################################################################

################################## set device ##################################

print("============================================================================================")

# set device to cpu or cuda
device = torch.device('cpu')

if (torch.cuda.is_available()):
    device = torch.device('cuda:3')
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
parser.add_argument('--num_iteration', default=500000, type=int) #  num of  games train
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

def main():
    summary_writer = SummaryWriter()
    num_iteration = 40000
    num_step = 50
    env = routingEnv(621)

    step = 0
    for i in tqdm.tqdm(range(num_iteration)):
        episode = i
        states = env.reset(episode)
        ep_total_reward = 0
        ep_local_reward = 0
        ep_rd = 0
        ep_rp = 0

        for j in range(num_step):
            for node in range(env.node_num):
                state = states[node].getState()[0]
                act = [[1, 1, 1]]
                env.store_action(node, act)
            states__, st_rd, st_rp, done = env.step()
            ep_rd += st_rd
            ep_rp += st_rp
            if j == num_step - 1:
                done = True
            for node in range(env.node_num):
                reward = states__[node].getRewards()
                local_reward = states__[node].getLocalRewards()
                ep_local_reward += local_reward[0][0]
                for re in range(len(reward)):
                    ep_total_reward += reward[re][0]
            states_ = env.get_state()
            step += 1
            states = states_
        ep_total_reward /= num_step
        ep_total_reward /= env.node_num
        ep_local_reward /= num_step
        ep_local_reward /= env.node_num
        ep_rd /= num_step
        ep_rp /= num_step
        print("local_reward", ep_local_reward)
        summary_writer.add_scalar('total_reward_change 15 30 zif standard round2', ep_total_reward, global_step=i)
        summary_writer.add_scalar('local_reward', ep_local_reward, global_step=i)
        summary_writer.add_scalar('ep_rd', ep_rd, i)
        summary_writer.add_scalar('ep_rp', ep_rp, i)

if __name__ == '__main__':
        main()
