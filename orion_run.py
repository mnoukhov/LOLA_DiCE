import argparse

from envs import IPD
from ipd_DiCE import Agent, play

parser = argparse.ArgumentParser()
parser.add_argument("--lr-out", default=0.2)
parser.add_argument("--lr-in", default=0.3)
parser.add_argument("--lr-v", default=0.3)
parser.add_argument("--gamma", default=0.96)
parser.add_argument("--n-update", default=200)
parser.add_argument("--len-rollout", default=150)
parser.add_argument("--batch-size", default=128)
parser.add_argument("--use-baseline", action="store_true")
parser.add_argument("--order", default=0)
parser.add_argument("--seed", default=42)

args = parser.parse_args()

ipd = IPD(args.len_rollout, args.batch_size)
scores = play(Agent(args.lr_out, args.lr_v, args.gamma, args.use_baseline, args.len_rollout),
              Agent(args.lr_out, args.lr_v, args.gamma, args.use_baseline, args.len_rollout),
              args.order,
              ipd,
              args.n_update,
              args.lr_in,
              args.len_rollout)

return -max(scores)
