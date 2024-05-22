import torch
import torch.distributed as dist

from model import ResNetSimCLR, ViTSimCLR
from data import get_imagenet, get_dataset, IMAGE_SIZE
from torch.utils.data import SubsetRandomSampler, DataLoader
from tqdm import tqdm
from cache import EmbeddingCache, StreamingCache
from timer import Timer
from log_wandb import wandb_init
from optimizer import LARS
from sampler import MCMC_BatchController
from utils import model_inference

from model import convert_layers
import os
import os.path as osp
import random
import time

from copy import deepcopy

FINITE_AUG = {"stl10": 10}

def setup(config, local_rank, rank, world_size, 
            wandb_prefix=""): 
    config["program_start_time"] = time.time()
    config["accum_eval_time"] = 0
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        # for reproducibility
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 
        torch.use_deterministic_algorithms(True)

    if local_rank == 0:
        if not os.path.exists(config["data_root"]):
            os.makedirs(config["data_root"])
    
    dev = torch.device('cuda:{}'.format(local_rank) if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(dev)

    ## prepare model
    if "resnet" in config["model_name"]:
        model = ResNetSimCLR(base_model=config["model_name"], 
                            mlp_dim=config["mlp_dim"], 
                            out_dim=config["dim"], 
                            projector_hidden_layers=config["projector_hidden_layers"])
    elif "vit" in config["model_name"]:
        model = ViTSimCLR(base_model=config["model_name"], 
                        mlp_dim=config["mlp_dim"], 
                        out_dim=config["dim"], 
                        projector_hidden_layers=config["projector_hidden_layers"], 
                        image_size=IMAGE_SIZE[config["dataset"]], 
                        dropout=config["dropout"])
    else:
        raise ValueError("unimplemented model {}".format(config["model_name"]))
    
    if config["disable_batchnorm"]:
        model = convert_layers(model, torch.nn.BatchNorm1d, torch.nn.Identity)
        model = convert_layers(model, torch.nn.BatchNorm2d, torch.nn.Identity)
        model = convert_layers(model, torch.nn.BatchNorm3d, torch.nn.Identity)
    elif "resnet" in config["model_name"]:
        # https://www.kaggle.com/code/parthdhameliya77/replace-batchnorm-layer-with-groupnorm-layer
        model = convert_layers(model, torch.nn.BatchNorm1d, torch.nn.GroupNorm, num_groups=32)
        model = convert_layers(model, torch.nn.BatchNorm2d, torch.nn.GroupNorm, num_groups=32)
        model = convert_layers(model, torch.nn.BatchNorm3d, torch.nn.GroupNorm, num_groups=32)
    
    model = model.to(dev)
    
    # if world_size > 1:
    dev_id = [dev] if torch.cuda.is_available() else None
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=dev_id, output_device=local_rank) # https://github.com/pytorch/pytorch/issues/66504
    model.eval()


    ## prepare data
    if config["finite_aug"]:
        n_aug = config["n_aug"]
    else:
        n_aug = None
    if "imagenet" in config["dataset"]:
        data, raw_data, _ = get_imagenet(config["dataset"], osp.join(config["data_root"], "imagenet"), config["transform_batch_size"], 
                                      split="train")
    else:
        data, raw_data = get_dataset(config["dataset"], rank, world_size, config["data_root"], config["transform_batch_size"], finite_aug=config["finite_aug"], n_aug=n_aug, get_subset=config["check_gradient_error"])
    shuffle_train = False if config["disable_wandb"] else True
    dataloader = DataLoader(data, batch_size=config["pos_batch_size"], shuffle=shuffle_train, num_workers=config["num_workers"], pin_memory=config["pin_memory"])
    rawdataloader = DataLoader(raw_data, batch_size=config["pos_batch_size"], shuffle=False, pin_memory=config["pin_memory"])
    if rank == 0:
        dataloader = tqdm(dataloader, desc="aug")
        rawdataloader = tqdm(rawdataloader, desc="raw")

    # dataloader for evaluation
    # linear probe: le_rawdataloader is train samples of linear classifier, i.e. training accuracy
    # linear probe: le_rawtestdataloader is test samples of linear classifier, i.e., testing accuracy
    # knn: knn_dataloader is train samples (reference samples). When knn_dataloader is None, le_rawdataloader is train samples (reference samples).
    # knn: le_rawtestdataloader is test samples

    if config["dataset"] == "mini-imagenet-1k" or config["dataset"] == "sub-imagenet100":
        _, test_data, _ = get_imagenet(config["dataset"], osp.join(config["data_root"], "imagenet"), config["transform_batch_size"], split="test")
        le_rawdataloader = rawdataloader
        le_rawtestdataloader = DataLoader(test_data, batch_size=config["compute_batch_size"], shuffle=False)
        knn_dataloader = None
    
    elif config["dataset"] == "imagenet100":
        _, val_data, _ = get_imagenet(config["dataset"], osp.join(config["data_root"], "imagenet"), config["transform_batch_size"], split="val")
        _, test_data, _ = get_imagenet(config["dataset"], osp.join(config["data_root"], "imagenet"), config["transform_batch_size"], split="test")

        le_rawdataloader = rawdataloader # the same setup as in sogclr: https://github.com/Optimization-AI/SogCLR/blob/PyTorch/lincls.py
        le_rawtestdataloader = DataLoader(test_data, batch_size=config["compute_batch_size"], shuffle=False)
        knn_dataloader = DataLoader(val_data, batch_size=config["compute_batch_size"], shuffle=False)
    
    elif config["dataset"] == "imagenet1000":
        raise NotImplementedError("where do we find the test labels for imagenet1000?")
        # le_rawdataloader = rawdataloader
        # le_rawtestdataloader = DataLoader(test_data, batch_size=config["compute_batch_size"], shuffle=False)
        # knn_dataloader = DataLoader(val_data, batch_size=config["compute_batch_size"], shuffle=False)

    
    elif config["dataset"] == "cifar10":
        _, test_data = get_dataset(config["dataset"], rank, world_size, config["data_root"], config["transform_batch_size"], split="test")
        
        le_rawdataloader = rawdataloader
        le_rawtestdataloader = DataLoader(test_data, batch_size=config["compute_batch_size"], shuffle=False)
        knn_dataloader = None

    elif config["dataset"] == "stl10":
        _, train_data = get_dataset(config["dataset"], rank, world_size, config["data_root"], config["transform_batch_size"], split="train")
        _, test_data = get_dataset(config["dataset"], rank, world_size, config["data_root"], config["transform_batch_size"], split="test")
        
        le_rawdataloader = DataLoader(train_data, batch_size=config["compute_batch_size"], shuffle=False)
        le_rawtestdataloader = DataLoader(test_data, batch_size=config["compute_batch_size"], shuffle=False)
        knn_dataloader = None
    else:
        raise ValueError("dataset {} unimplemented".format(config["dataset"]))
    
    # get data stats
    num_samples = torch.tensor(len(data), device=dev)
    network_num_samples = [torch.tensor(0, dtype=torch.int64, device=dev) for _ in range(world_size)]
    dist.all_gather(network_num_samples, num_samples, async_op=False)
    global_num_samples = sum(network_num_samples)
    epoch_n_batches = torch.tensor(len(dataloader), dtype=torch.int64, device=dev)
    dist.all_reduce(epoch_n_batches, op=dist.ReduceOp.MAX)
    epoch_n_batches = epoch_n_batches.item()


    ## prepare optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == "lars":
        optimizer = LARS(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"], momentum=config["momentum_decay"])
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    else:
        raise NotImplementedError("optimizer {} not implemented.".format(config["optimizer"]))

    ## prepare cache table
    
    if config["sampler"] == "gumbel_max":
        if config["cache_frac"] < 1:
            sampler = StreamingCache(config["dim"], config["P"], network_num_samples, config["cache_frac"], config["beta"], device=dev, gumbel_batch_size=config["gumbel_batch_size"], img_only=True, 
                                            disable_proj=config["disable_proj"])
        else:
            sampler = EmbeddingCache(config["dim"], config["P"], network_num_samples, config["beta"], device=dev, gumbel_batch_size=config["gumbel_batch_size"], img_only=True,
                                            disable_proj=config["disable_proj"], finite_n_aug=n_aug)
        
        with torch.no_grad():
            if isinstance(sampler, StreamingCache):
                # initialize cache table with a random subset of training samples
                init_loader = DataLoader(raw_data, batch_size=config["compute_batch_size"], num_workers=config["num_workers"], sampler=SubsetRandomSampler(sampler.cached_idx))
                if rank == 0:
                    init_loader = tqdm(init_loader)
                for image, _, _, idx, aug_idx in init_loader:
                    _, image_features = model_inference(model, image, dev)
                    sampler.update_cache_table(sampler.to_table_idx(idx), image_features, None)
            else:
                # initialize cache table with all training samples
                if config["finite_aug"]:
                    all_aug_data = deepcopy(data)
                    all_aug_data.aug_batch_size = all_aug_data.n_augmentations # return all augmentations
                    init_loader = DataLoader(all_aug_data, batch_size=config["compute_batch_size"], num_workers=config["num_workers"])
                    if rank == 0:
                        init_loader = tqdm(init_loader)
                    for _, aug_image, _, idx, aug_idx in init_loader:
                        for j in range(all_aug_data.n_augmentations):
                            _, image_features = model_inference(model, aug_image[j], dev)
                            sampler.update_cache_table((idx, aug_idx[j]), image_features, None)
                    del init_loader
                else:
                    for image, _, _, idx, _ in rawdataloader:
                        _, image_features = model_inference(model, image, dev)
                        sampler.update_cache_table(idx, image_features, None)
                    
    elif config["sampler"] == "mcmc":
        sampler = MCMC_BatchController(network_num_samples, rank, world_size, dev, n_aug)
    else:
        sampler = None
    
    iterator = range(config["epoch"])
    # display training progress
    if rank == 0 and not config["disable_wandb"]:
        wandb_init(config, name="{}-{}-{}-{}lr-({},{},{})batch".format( wandb_prefix, 
            config["model_name"], optimizer.__class__.__name__, config["lr"], config["pos_batch_size"]*world_size, config["neg_batch_size"], config["transform_batch_size"]))
    
    timer = Timer()
    
    return optimizer, model, data, raw_data, dataloader, rawdataloader, le_rawdataloader, le_rawtestdataloader, knn_dataloader, dev, network_num_samples, global_num_samples, epoch_n_batches, iterator, timer, sampler