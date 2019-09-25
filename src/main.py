import os
import argparse
import time
import torch
import logging as log
import numpy as np
from Data import Data_loader
from Trainer import Trainer
from Agent import Agent

class Option:
    def __init__(self, d):
        self.__dict__ = d

    def save(self):
        with open (os.path.join(self.this_expsdir, "option.txt"), "w", encoding='UTF-8') as f:
            for key, value in sorted(self.__dict__.items(), key = lambda x: x[0]):
                f.write("{}, {}\n".format(key, str(value)))

def main():
    log.basicConfig(level=log.INFO)

    parser = argparse.ArgumentParser(description = "Experiment setup")

    # Log configuration
    parser.add_argument('--exps_dir', default="../exps/", type=str)
    parser.add_argument('--exp_name', default="demo", type=str)
    
    # Dataset
    parser.add_argument('--datadir', default="../datasets", type=str)
    parser.add_argument('--dataset', default="WN18RR", type=str)

    # Agent configuration
    parser.add_argument('--state_embed_size', default=50, type=int)
    parser.add_argument('--relation_embed_size', default=50, type=int)
    parser.add_argument('--mlp_hidden_size', default=100, type=int)
    parser.add_argument('--use_entity_embed', default=False, type=bool)
    parser.add_argument('--entity_embed_size', default=50, type=int)
    parser.add_argument("--grad_clip_norm", default=5, type=int)

    parser.add_argument('--train_times', default=20, type=int)
    parser.add_argument('--test_times', default=100, type=int)
    parser.add_argument("--train_batch", default=200, type=int)
    parser.add_argument('--max_out', default=100, type=int)
    parser.add_argument('--max_step_length', default=3, type=int)

    # Learning configuration
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--decay_weight', default=0.02, type=float)
    parser.add_argument('--decay_batch', default=100, type=int)
    parser.add_argument('--decay_rate', default=0.9, type=float)

    parser.add_argument('--gamma', default=1, type=float)
    parser.add_argument('--Lambda', default=0.05, type=float)
    parser.add_argument('--beta', default=0.05, type=float)


    d = vars(parser.parse_args())
    option = Option(d)
    
    if option.exp_name is None:
        option.tag = time.strftime("%y-%m-%d-%H-%M")
    else:
        option.tag = option.exp_name

    if torch.cuda.is_available():
        option.use_cuda = True
    else:
        option.use_cuda = False

    option.this_expsdir = os.path.join(option.exps_dir, option.tag)
    if not os.path.exists(option.exps_dir):
        os.makedirs(option.exps_dir)
    if not os.path.exists(option.this_expsdir):
        os.makedirs(option.this_expsdir)

    if option.use_entity_embed is False:
        option.action_embed_size = option.relation_embed_size
    else:
        option.action_embed_size = option.relation_embed_size + option.entity_embed_size

    option.save()

    data_loader = Data_loader(option)
    option.num_entity = data_loader.num_entity
    option.num_relation = data_loader.num_relation
    agent = Agent(option, data_loader)
    trainer = Trainer(option, agent, data_loader)

    trainer.train()
    trainer.save_model()
    trainer.load_model()
    trainer.test()

if __name__ == "__main__":
    #torch.set_printoptions(threshold=100000)
    main()



