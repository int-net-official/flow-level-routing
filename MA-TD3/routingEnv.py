"""
Created on Fri May 19 16:07:55 2023

@author: YangYing
"""

import networkx as nx
import numpy as np
from decimal import Decimal
from router import router
from switch import switch
import random
import copy


class routingEnv:
    """
    Attributes:
       miu: service rate
       edge_num: Number of edges in the network (bidirectional links)
       node_num: Number of routers in the network
       capacity: Router capacity
    """

    def __init__(self, seed):

        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        G = nx.read_gml("Claranet.gml")
        H = G.to_directed()
        adj = nx.adjacency_matrix(H).todense()
        topo = adj
        self.G = nx.DiGraph(topo)
        self.max_hop = nx.diameter(self.G)  # Network Diameter
        self.edge_num = self.G.number_of_edges()
        self.node_num = self.G.number_of_nodes()
        self.candidate_path_num = 3
        self.flow_num = self.node_num  # Total traffic volume in one step
        self.capacity = Decimal(10000)
        self.state_dim = []
        self.neighbors = []
        for i in range(self.node_num):
            informationNum = self.G.degree(i) / 2 + 1
            self.state_dim.append(6 * informationNum)
            self.neighbors.append(list(self.G.neighbors(i)))
        self.action_dim = self.candidate_path_num
        self.alpha = 0.7
        self.avg_rate = 500
        self.zip_dist(self.avg_rate, 3, 100)  # Zipf distribution with a center at avg_rate and a step size of 100
        self.routers = []
        self.switches = []

        for i in range(self.node_num):
            self.routers.append(router(self.candidate_path_num))
            self.switches.append(switch(self.candidate_path_num))
        self.init_node_miu_load()  # Initialization of router arrival rate
        self.generate_flows()

    def init_node_miu_load(self):
        for i in range(self.node_num):
            self.G.nodes[i]['miu'] = np.random.choice([1000, 2000, 3000])
        for (u, v, d) in self.G.edges(data=True):
            self.G.edges[(u, v)]['weight'] = 1 / self.G.nodes[v]['miu']

    def zip_dist(self, mean_rate, zip_len, interval):
        sknewness = 1
        self.candidate_rates = []
        sum_zip = 0
        for i in range(1, zip_len + 1):
            if (i % 2 == 0):
                rate = mean_rate + int(i / 2) * interval
            else:
                rate = mean_rate - int(i / 2) * interval
            self.candidate_rates.append(rate)
            sum_zip += i ** (-1 * sknewness)
        self.zip_prob = []
        for i in range(zip_len):
            prob = (i + 1) ** (-1 * sknewness) / sum_zip
            self.zip_prob.append(prob)

    def k_paths(self, src_node, dst_node):
        paths = list(nx.shortest_simple_paths(self.G, source=src_node, target=dst_node, weight='weight'))
        if len(paths) <= self.candidate_path_num:
            return paths
        K_paths = []
        for i in range(self.candidate_path_num):
            K_paths.append(paths[i])
        return K_paths

    def get_paths(self):
        self.paths = []
        for i in range(self.flow_num):
            paths = self.k_paths(self.src_nodes[i], self.dst_nodes[i])
            self.paths.append(paths)

    def reset(self, episode):
        """
        reset the env
        Returns:  state

        """
        # Import the initial traffic
        self.assignment_flows(episode)
        # Find K paths for each flow
        self.get_paths()
        state = self.get_state()
        return state

    def generate_flows(self):
        self.src_nodes_init = []
        self.dst_nodes_init = []
        self.rates_init = []
        self.rates_change = []
        self.types_init = []
        for i in range(self.flow_num):
            src_node = i
            self.src_nodes_init.append(src_node)
            dst_node = np.random.randint(0, self.node_num)
            while (src_node == dst_node):
                dst_node = np.random.randint(0, self.node_num)
            self.dst_nodes_init.append(dst_node)
            rate = np.random.choice(self.candidate_rates, p=self.zip_prob)
            self.rates_init.append(rate)
            flow_type = np.random.choice([0, 1])
            self.types_init.append(flow_type)

    def assignment_flows(self, episode):
        """
        Assign the initialized flow information to the incoming flow information entering the network

        """

        self.src_nodes = []
        self.dst_nodes = []
        self.src_nodes = copy.deepcopy(self.src_nodes_init)
        self.dst_nodes = copy.deepcopy(self.dst_nodes_init)
        self.rates = []
        self.types = []
        self.types = copy.deepcopy(self.types_init)
        for i in range(self.flow_num):
            rate_init = self.rates_init[i]
            rate = np.clip(np.random.normal(loc=rate_init, scale=10), rate_init - 20, rate_init + 20)
            self.rates.append(rate)

    def add_flows(self):
        """
        Integrate the generated flows into the network topology

        """

        for node in range(self.node_num):
            Info = self.routers[node].getInfo()
            actions = self.routers[node].getAction()
            for j in range(len(Info)):
                index = Info[j][0]  # Flow identifier to be processed
                action = actions[j]
                rates = []
                for k in range(Info[j][1]):
                    rate = self.rates[index] * action[k]
                    rates.append(rate)
                    path = self.paths[index][k]
                    for pathIndex in range(len(path)):
                        pathNode = path[pathIndex]
                        switch = self.switches[pathNode]
                        if pathIndex == 0:
                            switch.addInRate(index, k, rate)
                        if pathIndex == len(path) - 1:
                            next_node = -1
                        else:
                            next_node = path[pathIndex + 1]
                        switch.addReminderPath(index, k, next_node)  # Record the next hop
                self.routers[node].add_rateInfo(index, rates)  # Store the splitting ratio for this flow

        for count in range(10):
            for node in range(self.node_num):
                switch = self.switches[node]
                lam = switch.getSumInrate()
                loss = self.loss(lam, node)
                for j in range(len(switch.InRate)):
                    index = switch.InRate[j][0]
                    InRate = switch.InRate[j][1]
                    OutRate = InRate * (1 - loss)
                    switch.addOutRate2(index, OutRate)
                    for k in range(len(switch.reminderPath)):
                        if index == switch.reminderPath[k][0]:
                            next_node = switch.reminderPath[k][1]
                            if next_node != -1:
                                self.switches[next_node].addInRate2(index, OutRate)

    def get_reward(self):
        """
        Calculate the total network delay
        """

        st_rd = []
        st_rp = []
        for node in range(self.node_num):
            Info = self.routers[node].getInfo()
            RateInfo = self.routers[node].getRateInfo()
            for i in range(len(Info)):  # Number of flows under this router
                index = Info[i][0]  # Flow identifier to be processed
                path_delay = 0
                path_packet_ava = 0
                rate = self.rates[index]
                for j in range(Info[i][1]):
                    path = self.paths[index][j]  # One of the k paths
                    single_path_delay = 0
                    single_packet_ava = RateInfo[i][1][j]
                    if (single_packet_ava == 0):
                        continue
                    for p in path:
                        lam = self.switches[p].getSumInrate()
                        delay = self.MMK(lam, node)
                        if p == path[len(path) - 1]:
                            single_packet_ava = self.switches[p].getOutRate(index, j)
                        single_path_delay += delay
                    path_delay += RateInfo[i][1][j] / rate * single_path_delay  # Packet Weighted Delay
                    path_packet_ava += single_packet_ava
                r_d = 1 - path_delay * self.G.nodes[node]['miu'] / (self.max_hop * float(self.capacity))
                r_p = path_packet_ava / rate
                if self.types[index] == 0:
                    reward = r_d * self.alpha + r_p * (1 - self.alpha)
                else:
                    reward = r_d * (1 - self.alpha) + r_p * self.alpha
                reward = round(reward, 2)
                self.routers[node].add_local_reward([reward])
                st_rd.append(r_d)
                st_rp.append(r_p)
        for node in range(self.node_num):
            sumRate = 0
            for neighbor in self.neighbors[node]:
                sumRate += self.rates[neighbor]
            neighborsReward = 0
            for neighbor in self.neighbors[node]:
                omega = self.rates[neighbor] / sumRate
                neighborsReward += self.routers[neighbor].getLocalRewards()[0][0] * omega
            total_reward = 0.6 * self.routers[node].getLocalRewards()[0][0] + 0.4 * neighborsReward
            total_reward = round(total_reward, 2)
            self.routers[node].add_total_reward([total_reward])
        return st_rd, st_rp

    def get_state(self):
        for i in range(self.node_num):
            self.routers[i].reset()
            self.switches[i].reset()
        for i in range(self.flow_num):
            index = self.src_nodes[i]
            state = []
            state.append(round(self.dst_nodes[i] / self.node_num, 4))
            state.append(round(self.rates[i] / self.avg_rate, 4))
            paths_num = len(self.paths[i])
            for j in range(paths_num):
                state.append(len(self.paths[i][j]) / self.max_hop)
            for j in range(self.candidate_path_num - paths_num):
                state.append(0)
            # add flow type
            state.append(self.types[i])
            self.routers[index].add_state(state)
            self.routers[index].add_info(i, paths_num)
        for i in range(self.node_num):
            total_state = []
            total_state += self.routers[i].getLocalState()[0]
            for neighbor in self.neighbors[i]:
                total_state += self.routers[neighbor].getLocalState()[0]
            self.routers[i].add_total_state(total_state)
        return self.routers

    def MMK(self, lam, node):
        """Calculate the delay for a single node"""
        rho = lam / self.G.nodes[node]['miu']
        rho = Decimal(rho)
        loss = self.loss(lam, node)
        if rho == 1:
            queue_length = self.capacity / 2
        else:
            queue_length = rho / (1 - rho) - (self.capacity + 1) * rho ** (self.capacity + 1) \
                           / (1 - rho ** (self.capacity + 1))
        queue_length = float(queue_length)
        delay = queue_length / (lam * (1 - loss))
        return delay

    def loss(self, lam, node):
        """Calculate the packet loss rate for a single node"""
        rho = lam / self.G.nodes[node]['miu']
        rho = Decimal(rho)
        if rho == 1:
            loss = 1 / (self.capacity + 1)
        else:
            loss = ((1 - rho) * rho ** self.capacity) \
                   / (1 - rho ** (self.capacity + 1))
        return float(loss)

    def step(self):
        """By obtaining actions from the agent, reintroduce each flow into the network,
        and obtain experimental metrics such as packet loss rate.

        Returns:
            routers: Relevant information of the router
            r_d: Reward value for latency
            r_p: Reward value for packet loss rate
        """
        self.add_flows()
        st_rd, st_rp = self.get_reward()
        self.change_flows()
        return self.routers, st_rd, st_rp, False

    def change_flows(self):
        """
        Adjust the arrival rate of a portion of flows with maximum delay at each step.

        """
        rewards = []
        for node in range(self.node_num):
            rewards.append([self.routers[node].getRewards()[0][0], node])
        rewards.sort(key=lambda x: x[0])
        for i in range(len(rewards)):
            if i <= len(rewards) / 2:
                self.rates[rewards[i][1]] -= self.rates[rewards[i][1]] * 0.01
            else:
                self.rates[rewards[i][1]] += self.rates[rewards[i][1]] * 0.01

    def store_action(self, index, action):
        self.routers[index].add_action(action)

