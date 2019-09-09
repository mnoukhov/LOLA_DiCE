# coding: utf-8

import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from copy import deepcopy

from envs import IPD


def magic_box(x):
    return torch.exp(x - x.detach())

class Memory():
    def __init__(self):
        self.self_logprobs = []
        self.other_logprobs = []
        self.values = []
        self.rewards = []

    def add(self, lp, other_lp, v, r):
        self.self_logprobs.append(lp)
        self.other_logprobs.append(other_lp)
        self.values.append(v)
        self.rewards.append(r)

    def dice_objective(self, gamma, use_baseline):
        self_logprobs = torch.stack(self.self_logprobs, dim=1)
        other_logprobs = torch.stack(self.other_logprobs, dim=1)
        values = torch.stack(self.values, dim=1)
        rewards = torch.stack(self.rewards, dim=1)

        # apply discount:
        cum_discount = torch.cumprod(gamma * torch.ones(*rewards.size()), dim=1)/gamma
        discounted_rewards = rewards * cum_discount
        discounted_values = values * cum_discount

        # stochastics nodes involved in rewards dependencies:
        dependencies = torch.cumsum(self_logprobs + other_logprobs, dim=1)

        # logprob of each stochastic nodes:
        stochastic_nodes = self_logprobs + other_logprobs

        # dice objective:
        dice_objective = torch.mean(torch.sum(magic_box(dependencies) * discounted_rewards, dim=1))

        if use_baseline:
            # variance_reduction:
            baseline_term = torch.mean(torch.sum((1 - magic_box(stochastic_nodes)) * discounted_values, dim=1))
            dice_objective = dice_objective + baseline_term

        return -dice_objective # want to minimize -objective

    def value_loss(self):
        values = torch.stack(self.values, dim=1)
        rewards = torch.stack(self.rewards, dim=1)
        return torch.mean((rewards - values)**2)

def act(batch_states, theta, values):
    batch_states = torch.from_numpy(batch_states).long()
    probs = torch.sigmoid(theta)[batch_states]
    m = Bernoulli(1-probs)
    actions = m.sample()
    log_probs_actions = m.log_prob(actions)
    return actions.numpy().astype(int), log_probs_actions, values[batch_states]

def get_gradient(objective, theta):
    # create differentiable gradient for 2nd orders:
    grad_objective = torch.autograd.grad(objective, (theta), create_graph=True)[0]
    return grad_objective

def step(theta1, theta2, values1, values2, ipd, len_rollout):
    # just to evaluate progress:
    (s1, s2), _ = ipd.reset()
    score1 = 0
    score2 = 0
    for t in range(len_rollout):
        a1, lp1, v1 = act(s1, theta1, values1)
        a2, lp2, v2 = act(s2, theta2, values2)
        (s1, s2), (r1, r2),_,_ = ipd.step((a1, a2))
        # cumulate scores
        score1 += np.mean(r1)/float(len_rollout)
        score2 += np.mean(r2)/float(len_rollout)
    return (score1, score2)

class Agent():
    def __init__(self, lr_out, lr_v, gamma, use_baseline, len_rollout):
        # init theta and its optimizer
        self.theta = nn.Parameter(torch.zeros(5, requires_grad=True))
        self.theta_optimizer = torch.optim.Adam((self.theta,),lr=lr_out)
        # init values and its optimizer
        self.values = nn.Parameter(torch.zeros(5, requires_grad=True))
        self.value_optimizer = torch.optim.Adam((self.values,),lr=lr_v)
        self.gamma = gamma
        self.use_baseline = use_baseline
        self.len_rollout = len_rollout

    def theta_update(self, objective):
        self.theta_optimizer.zero_grad()
        objective.backward(retain_graph=True)
        self.theta_optimizer.step()

    def value_update(self, loss):
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

    def in_lookahead(self, other_theta, other_values, ipd):
        (s1, s2), _ = ipd.reset()
        other_memory = Memory()
        for t in range(self.len_rollout):
            a1, lp1, v1 = act(s1, self.theta, self.values)
            a2, lp2, v2 = act(s2, other_theta, other_values)
            (s1, s2), (r1, r2),_,_ = ipd.step((a1, a2))
            other_memory.add(lp2, lp1, v2, torch.from_numpy(r2).float())

        other_objective = other_memory.dice_objective(self.gamma, self.use_baseline)
        grad = get_gradient(other_objective, other_theta)
        return grad

    def out_lookahead(self, other_theta, other_values, ipd):
        (s1, s2), _ = ipd.reset()
        memory = Memory()
        for t in range(self.len_rollout):
            a1, lp1, v1 = act(s1, self.theta, self.values)
            a2, lp2, v2 = act(s2, other_theta, other_values)
            (s1, s2), (r1, r2),_,_ = ipd.step((a1, a2))
            memory.add(lp1, lp2, v1, torch.from_numpy(r1).float())

        # update self theta
        objective = memory.dice_objective(self.gamma, self.use_baseline)
        self.theta_update(objective)
        # update self value:
        v_loss = memory.value_loss()
        self.value_update(v_loss)

def play(agent1, agent2, n_lookaheads, ipd, n_update, lr_in, len_rollout):
    joint_scores = []
    print("start iterations with", n_lookaheads, "lookaheads:")
    for update in range(n_update):
        # copy other's parameters:
        theta1_ = agent1.theta.clone().detach().requires_grad_(True)
        values1_ = agent1.values.clone().detach().requires_grad_(True)
        theta2_ = agent2.theta.clone().detach().requires_grad_(True)
        values2_ = agent2.values.clone().detach().requires_grad_(True)

        for k in range(n_lookaheads):
            # estimate other's gradients from in_lookahead:
            grad2 = agent1.in_lookahead(theta2_, values2_, ipd)
            grad1 = agent2.in_lookahead(theta1_, values1_, ipd)
            # update other's theta
            theta2_ = theta2_ - lr_in * grad2
            theta1_ = theta1_ - lr_in * grad1

        # update own parameters from out_lookahead:
        agent1.out_lookahead(theta2_, values2_, ipd)
        agent2.out_lookahead(theta1_, values1_, ipd)

        # evaluate progress:
        score = step(agent1.theta, agent2.theta, agent1.values, agent2.values, ipd, len_rollout)
        joint_scores.append(0.5*(score[0] + score[1]))

        # print
        if update%10==0 :
            p1 = [p.item() for p in torch.sigmoid(agent1.theta)]
            p2 = [p.item() for p in torch.sigmoid(agent2.theta)]
            print('update', update, 'score (%.3f,%.3f)' % (score[0], score[1]) , 'policy (agent1) = {%.3f, %.3f, %.3f, %.3f, %.3f}' % (p1[0], p1[1], p1[2], p1[3], p1[4]),' (agent2) = {%.3f, %.3f, %.3f, %.3f, %.3f}' % (p2[0], p2[1], p2[2], p2[3], p2[4]))

    return joint_scores

# plot progress:
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr-out", default=0.2)
    parser.add_argument("--lr-in", default=0.3)
    parser.add_argument("--lr-v", default=0.3)
    parser.add_argument("--gamma", default=0.96)
    parser.add_argument("--n-update", default=200)
    parser.add_argument("--len-rollout", default=150)
    parser.add_argument("--batch-size", default=128)
    parser.add_argument("--use-baseline", action="store_true")
    parser.add_argument("--seed", default=42)

    args = parser.parse_args()
    colors = ['b','c','m','r']

    torch.manual_seed(args.seed)
    for i in range(4):
        ipd = IPD(args.len_rollout, args.batch_size)
        scores = play(Agent(args.lr_out, args.lr_v, args.gamma, args.use_baseline, args.len_rollout),
                      Agent(args.lr_out, args.lr_v, args.gamma, args.use_baseline, args.len_rollout),
                      i,
                      ipd,
                      args.n_update,
                      args.lr_in,
                      args.len_rollout)
        plt.plot(scores, colors[i], label=str(i)+" lookaheads")

    plt.legend()
    plt.xlabel('rollouts', fontsize=20)
    plt.ylabel('joint score', fontsize=20)
    plt.show()
