import argparse
from math import sqrt 

def parse_args(world_size):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", default="stl10", help="dataset name", choices=["stl10", "cifar10", "mini-imagenet-1k", "sub-imagenet100", "imagenet100", "imagenet1000"])
    parser.add_argument("--model_name", default="resnet18", help="encoder backbone model name", choices=["resnet18", "resnet50", "vit_b_16"])
    parser.add_argument("--optimizer", default="adam", help="optimizer name", choices=["adam", "lars", "sgd"])
    parser.add_argument("--sampler", default="None", help="negative sampler name", choices=["gumbel_max", "mcmc"])
    parser.add_argument("--alpha", default=1.0, type=float, help="(1-alpha) * simclr_grad + alpha * gumbel_grad")
    parser.add_argument("--beta", default=1.0, type=float)
    parser.add_argument("--momentum_decay", default=0.9, type=float)
    parser.add_argument("--rho", default=0.01, type=float, help="cache refresh ratio")
    parser.add_argument("--cache_frac", default=1, type=float, help="size of the cache, use streaming cache when cache_frac < 1")
    parser.add_argument("--P", default=4096, type=int, help="projected dimension")
    parser.add_argument("--epoch", default=100, type=int, help="number of total epochs to run")
    parser.add_argument("--second_stage_epoch", default=50, type=int, help="the epoch to start second stage")
    parser.add_argument("--pos_batch_size", default=256, type=int, help="positive batch size of the whole system")
    parser.add_argument("--neg_batch_size", default=0, type=int, help="negative batch size of the each node")
    parser.add_argument("--transform_batch_size", default=2, type=int, help="number of augmentations for every sample")
    parser.add_argument("--eval_transform_batch_size", default=2, type=int, help="number of augmentations for every sample during evaluation")
    parser.add_argument("--compute_batch_size", default=512, type=int, help="batch size of encoder inference")
    parser.add_argument("--gumbel_batch_size", default=2048, type=int, help="batch size of cache dataloader")
    parser.add_argument("--num_workers", default=0, type=int, help="number of dataloader workers")
    parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--dropout", default=0, type=float)
    parser.add_argument("--linear_eval_weight_decay", default=0, type=float)
    parser.add_argument("--mlp_dim", default=512, type=int, help="encoder backbone output dimension, i.e., the dimension of linear probe.")
    parser.add_argument("--dim", default=512, type=int, help="encoder output dimension")
    parser.add_argument("--projector_hidden_layers", default=0, type=int, help="number of hidden layers in projector")
    parser.add_argument("--sogclr_gamma", default=0.9, type=float)

    parser.add_argument("--disable_proj", action="store_true", help="disable projection in cache")
    parser.add_argument("--disable_wandb", action="store_true", help="disable wandb logging")
    parser.add_argument("--disable_batchnorm", action="store_true", help="remove batchnorm layers")
    parser.add_argument("--disable_scheduler", action="store_true", help="remove learning rate scheduler")
    parser.add_argument("--pin_memory", action="store_true", help="pin_memory in dataloader")
    parser.add_argument("--grad_accum", action="store_true", help="accumulate negative gradient to reduce memory consumption")
    parser.add_argument("--finite_aug", action="store_true", help="use a finite set of precomputed augmentations")
    parser.add_argument("--eval_iteration", action="store_true", help="perform evaluation every eval_freq iterations")
    parser.add_argument("--check_gradient_error", action="store_true", help="log the gradient error")
    
    parser.add_argument("--n_aug", default=10, type=int, help="number of precomputed augmentations")

    parser.add_argument("--warmup_epoch", default=10, type=int, help="epochs before scheduler kicks in")

    parser.add_argument("--dist_eval", action="store_true", help="run linear evaluation in distributed manner.")
    parser.add_argument("--eval_freq", default=1, type=int, help="perform evaluation at every eval_freq epochs.")
    parser.add_argument("--display_freq", default=100, type=int, help="display running time stats at display_freq.")
    parser.add_argument("--lower_beta", default=-1, type=float, help="Smallest beta in temperature schedule.")
    parser.add_argument("--upper_beta", default=-1, type=float, help="Largest beta in temperature schedule.")
    parser.add_argument("--temp_period", default=0, type=int, help="Temperture schedule period in epochs.")

    parser.add_argument("--mcmc_burn_in", default=0, type=float, help="fraction of burn in steps in every iteration")
    parser.add_argument("--stop_burn_in_ep", default=9999, type=float, help="fraction of burn in steps in every iteration")
    parser.add_argument("--scorelist_size", default=40, type=int, help="number of negative index to store")

    parser.add_argument("--data_root", default="/home/ec2-user/data", type=str, help="root directory of data storage")
    

    args = parser.parse_args()
    config = vars(args)
    config["world_size"] = world_size

    config["pos_batch_size"] = config["pos_batch_size"] // world_size
    config["compute_batch_size"] = config["compute_batch_size"] // world_size
    

    return config