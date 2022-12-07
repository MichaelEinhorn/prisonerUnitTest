"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import CfgNode as CN
import torch.nn.functional as F
import json

def whiten(values, shift_mean=True):
    """Whiten values."""
    # single element is equal to mean
    if values.shape[1] == 1:
        return torch.tensor(0, dtype=values.dtype)
    mean, var = torch.mean(values), torch.var(values)
    # 1e-8 is too small for fp16
    whitened = (values - mean) * torch.rsqrt(var + 1e-6)
    if not shift_mean:
        whitened += mean
    return whitened

def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy

def clip_by_value(x, tensor_min, tensor_max):
    """
    Tensor extenstion to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped

class PrisonerTrainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0

        C.alg_name = "ppo"
        C.gamma = 0
        C.lam = 0.95
        C.cliprange = .2
        C.cliprange_value = .2
        C.vf_coef = .1

        return C

    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.optimizer = None
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    # constraint on payoff mat to have same equilibrium discount for starting with coop and defect
    # https://www.wolframalpha.com/input?i=%28b+-+a%29+%2F+%28b+-+c%29+%3D+%28c+-+d%29+%2F+%28a+-+d%29
    def payOffMat(self):
        # row is x's choice
        # column is y's choice
        # 0 is defect, 1 is coop
        # d = a - b + c, b - c != 0 for equal discount
        # c = -a, d = -b, a + b!=0 for zero sum
        # b = 3a for 0.5 discount
        # mat = [[0.5, -1.5],
        #        [1.5,   -0.5]]
        # discount coop > discount defect, leads to alternating
        # https://www.wolframalpha.com/input?i=%28b+-+a%29+%2F+%28b+-+c%29+%3E+%28c+-+d%29+%2F+%28a+-+d%29%2C+b+%3E+a+%3E+c+%3E+d%2C+a+%2B+b+%2B+c+%2B+d+%3D+0%2C+%28b+-+a%29+%2F+%28b+-+c%29+%2B+%28c+-+d%29+%2F+%28a+-+d%29+%3D+1%2C+a+%3D+0.5%2C+b+%3D+2
        mat = [[0.5, -1.76139],
               [2, -0.738613]]
        # discount coop < discount defect
        # https://www.wolframalpha.com/input?i=%28b+-+a%29+%2F+%28b+-+c%29+%3C+%28c+-+d%29+%2F+%28a+-+d%29%2C+b+%3E+a+%3E+c+%3E+d%2C+a+%2B+b+%2B+c+%2B+d+%3D+0%2C+%28b+-+a%29+%2F+%28b+-+c%29+%2B+%28c+-+d%29+%2F+%28a+-+d%29+%3D+1%2C+a+%3D+0.5%2C+b+%3D+1
        # mat = [[0.5, -1.27526],
        #        [1, -0.224745]]
        # mat = [[0.5, -1.20718],
        #        [0.7, 0.00717968]]

        return torch.tensor(mat)

    # http://www.statslab.cam.ac.uk/~rrw1/oc/prisoner2.pdf
    # https://www.wolframalpha.com/input?i=a+%2F+%281+-+g%29+%3D+b+%2B+c*g+%2F+%281+-+g%29%2C+solve+for+g
    def equilibriumDiscount(self, startCoop=True):
        mat = self.payOffMat()
        # cooperate rew is discounted coop reward
        # defective rew is one defect on coop, and then double defect
        if startCoop:
            gamma = (mat[1][0] - mat[0][0]) / (mat[1][0] - mat[1][1])
        else:
            gamma = (mat[0][1] - mat[1][1]) / (mat[0][1] - mat[0][0])
        return gamma

    # reward x
    def payOffMatEx(self, x, y):
        # row is x's choice
        # column is y's choice
        # 0 is defect, 1 is coop
        mat = [[0.5, -1],
               [1,   0]]

        return mat[x][y]

    def scores(self, seq):
        score = torch.zeros((seq.shape[0], seq.shape[1] - 1))
        for b in range(seq.shape[0]):
            for i in range(1, seq.shape[1]):
                score[b, i-1] = self.payOffMatEx(seq[b, i], seq[b, i-1])
        return score


    def run(self):
        model, config = self.model, self.config

        rew_dict = {}
        avg_rets = []
        iter_list = []
        loss_list = []

        possible_rew = torch.flatten(self.payOffMat())
        for i in range(possible_rew.shape[0]):
            rew_dict[possible_rew[i].item()] = []
        # print(rew_dict)

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)
        firstTokenSampler = torch.distributions.bernoulli.Bernoulli(probs=torch.tensor([0.5]))

        model.train()
        gen_len = 100
        self.loss = torch.tensor(0)
        input_ids = firstTokenSampler.sample(sample_shape=(self.config.batch_size,)).long().to("cuda")
        # print(input_ids.shape) for some reason its (batch, 1)
        while True:
            # get the last generation from the model
            input_ids = input_ids[:, -1:]
            # generate 1 more to get next value. Prevents there from being a last game
            input_ids, logits_old, values_old = model.generateLogits(input_ids, gen_len + 1, temperature=1.0, do_sample=True, top_k=None)
            logits_old = torch.stack(logits_old).transpose(0, 1)
            values_old = torch.stack(values_old).transpose(0, 1)

            rewards = self.payOffMat()[input_ids[:, 1:], input_ids[:, :-1]].to("cuda")

            # remove the extra 1, but keep the extra value
            input_ids = input_ids[:, :-1]
            logits_old = logits_old[:, :-1]
            reward_next = rewards[:, -1]
            rewards = rewards[:, :-1]
            values_next = values_old[:, -1, 0]
            values_old = values_old[:, :-1, 0]

            old_logprobs = logprobs_from_logits(logits_old[:, :, :], input_ids[:, 1:])


            # rewards2 = self.scores(input_ids).to("cuda")
            # print(rewards.shape, rewards2.shape)
            # print(torch.sum(rewards - rewards2))
            returns = rewards.clone()
            returns[:, -1] += self.config.gamma * reward_next[:]
            for i in range(rewards.shape[1] - 2, -1, -1):
                returns[:, i] += self.config.gamma * returns[:, i + 1]

            if self.iter_num % 1 == 0:
                rewardsStat = torch.unique(rewards, return_counts=True)
                out, count = rewardsStat
                # print("iter ", self.iter_num, "loss", self.loss.item(), rewardsStat, "returns", torch.mean(returns).item())
                loss_list.append(self.loss.item())
                iter_list.append(self.iter_num)
                for k in rew_dict:
                    rew_dict[k].append(0)
                for i in range(out.shape[0]):
                    rew_dict[out[i].detach().to("cpu").item()][-1] = count[i].detach().to("cpu").item()
                avg_rets.append(torch.mean(returns).detach().to("cpu").item())
                with open("rewStats.json", 'w') as file:
                    json.dump((iter_list, rew_dict, avg_rets, loss_list), file)

            for i in range(4):
                logits, _, values = model(input_ids, outputVal=True)
                if "ppo" in self.config.alg_name:
                    self.loss = self.ppoLoss(logits, values, old_logprobs, values_old, rewards, input_ids, gen_len, values_next, returns)
                elif self.config.alg_name == "reject":
                    self.loss = self.rejectLoss(logits, values, old_logprobs, values_old, rewards, input_ids, gen_len, values_next, returns)
                # backprop and update the parameters
                model.zero_grad(set_to_none=True)
                self.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.optimizer.step()

                self.trigger_callbacks('on_batch_end')
                self.iter_num += 1
                print("\r iter ", self.iter_num, "/", config.max_iters, end="")
                tnow = time.time()
                self.iter_dt = tnow - self.iter_time
                self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break

    def rejectLoss(self, logits, vpred, old_logprobs, values_old, rewards, input_ids, gen_len, values_next, returns):
        returns = rewards.clone()
        for i in range(rewards.shape[1]-2, -1, -1):
            returns[:, i] += self.config.gamma * returns[:, i+1]

        top_n = self.config.batch_size * gen_len // 4
        # print(top_n)
        logits = logits[:, :-1]
        input_ids = input_ids[:, 1:]

        logits = torch.flatten(logits, start_dim=0, end_dim=1)
        input_ids = torch.flatten(input_ids, start_dim=0, end_dim=1)
        returns = torch.flatten(returns, start_dim=0, end_dim=1)

        # print(logits.shape, input_ids.shape)
        ret_sort = torch.argsort(returns, descending=True)
        # print(ret_sort.shape)
        ret_sort = ret_sort[: top_n]
        loss = F.cross_entropy(logits, input_ids, ignore_index=-1, reduction='none')
        loss_rejected = loss[ret_sort]
        return torch.mean(loss_rejected)

    def ppoLoss(self, logits, vpred, old_logprobs, values_old, rewards, input_ids, gen_len, values_next, returns_unused):
        """Calculate policy and value losses."""
        lastgaelam = torch.zeros(values_old.shape[0], device=values_old.device)
        advantages_reversed = []

        rewards = rewards[:, -gen_len:]
        values_old = values_old[:, -gen_len:]

        for t in reversed(range(gen_len)):
            nextvalues = values_old[:, t + 1] if t < gen_len - 1 else values_next
            delta = rewards[:, t] + self.config.gamma * nextvalues - values_old[:, t]
            lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values_old
        advantages = whiten(advantages)
        advantages = advantages.detach()

        # computed batched before this method called
        # logits, vpred = self.forward(model_input, outputVals=True)

        logprob = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])

        # only the generation part of the values/logprobs is needed
        # both logits and values are shifted 1 left from the input
        # start = querry_len - 1
        # end = querry_len + gen_len - 1
        # right pad
        # logprob, vpred = logprob[:, start:end], vpred[:, start:end]
        # left pad
        # logits were already shifted
        logprob, vpred = logprob[:, -gen_len:], vpred[:, -(gen_len + 1):-1, 0]
        # logprob, vpred = logprob[:, total_len-gen_len:total_len], vpred[:, total_len-gen_len - 1:total_len-1]

        vpredclipped = clip_by_value(vpred,
                                     values_old - self.config.cliprange_value,
                                     values_old + self.config.cliprange_value)

        vf_losses1 = (vpred - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))
        vf_clipfrac = torch.mean(torch.gt(vf_losses2, vf_losses1).double())

        ratio = torch.exp(logprob - old_logprobs)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio,
                                               1.0 - self.config.cliprange,
                                               1.0 + self.config.cliprange)

        pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))
        pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).double())

        loss = pg_loss + self.config.vf_coef * vf_loss

        return loss