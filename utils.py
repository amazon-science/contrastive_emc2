import torch
import math
from torch.utils.data import SequentialSampler, BatchSampler
import torch.distributed as dist

def eval_round(config, ep, it, epoch_n_batches):
    if config["eval_iteration"]:
        return it % config["eval_freq"] == 0 or (ep == config["epoch"] - 1 and it == epoch_n_batches - 1)
    else:
        # evaluate at end of epoch
        return it == epoch_n_batches - 1 and (ep % config["eval_freq"] == 0 or ep == config["epoch"] - 1)

def sync_gradient(optimizer, world_size):
    # avg gradients
    for group in optimizer.param_groups:
        for prm in group["params"]:
            dist.all_reduce(prm.grad, op=dist.ReduceOp.SUM)
            prm.grad /= world_size
        
def save_current_gradient_to_host_memory(optimizer, world_size):
    gradients = []
    for group in optimizer.param_groups:
        for prm in group["params"]:
            g = prm.grad.detach().clone()
            dist.all_reduce(g, op=dist.ReduceOp.SUM)
            g /= world_size
            gradients.append(g.cpu())
    return gradients

def restore_gradient_to_model(optimizer, gradients):
    i = 0
    for group in optimizer.param_groups:
        for prm in group["params"]:
            prm.grad = gradients[i].to(prm.grad.device)
            i += 1


def adjust_learning_rate(optimizer, epoch, config):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if not config["disable_scheduler"]:
        warmup_epochs, config_lr = config["warmup_epoch"], config["lr"]
        if epoch < warmup_epochs:
            lr = config_lr * epoch / warmup_epochs 
        else:
            lr = config_lr * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (config["epoch"] - warmup_epochs)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def adjust_beta(epoch, config):
    """Temperature schedules https://openreview.net/pdf?id=ejHUr4nfHhD """
    beta = (config["upper_beta"] - config["lower_beta"]) * ( 1 + math.cos(2*math.pi * epoch/config["temp_period"]) ) / 2 + config["lower_beta"]
    return beta


    
def aug_image_inference(model, aug_images, dev):
    aug_images = torch.stack(aug_images)
    trans_batch, pos_batch, ch, he, wid = aug_images.shape
    aug_images = aug_images.view(trans_batch*pos_batch, ch, he, wid)
    # import matplotlib.pyplot as plt
    # if dist.get_rank() == 0:
    #     plt.imshow(  aug_images[2].permute(1, 2, 0)  )
    #     plt.show()
    #     plt.imshow(  aug_images[2+pos_batch].permute(1, 2, 0)  )
    #     plt.show()
    # exit()
    
    _, aug_outputs = model_inference(model, aug_images, dev)
   
    out_dim = aug_outputs.shape[-1]
    aug_outputs = aug_outputs.view(trans_batch, pos_batch, out_dim)
    return aug_outputs # in shapes of [aug_batch_size, pos_batch_size, embd_size]


def model_inference(model, images, dev):
    # batch preprocess
    images = images.to(dev, non_blocking=True)
    hidden_features, output_features = model(images)
    return hidden_features, output_features


def repeat_rows(mat, repeat_n):
    """
    repeat mat[i] by repeat_n[i] times
    """
    assert len(mat) == len(repeat_n), "check len, {} == {}".format(len(mat), len(repeat_n))
    vecs = []
    for i, n in enumerate(repeat_n):
        for _ in range(n):
            vecs.append(mat[i])
    return torch.stack(vecs)


def compute_repeated(flatten_reqs_idx, model, data, feature_dim, dev, compute_batch_size):
    # compute inference of the repeated requests.
    flatten_response_embs = torch.zeros((len(flatten_reqs_idx), feature_dim), dtype=torch.float32, device=dev)

    unique_flatten_reqs = list(set(flatten_reqs_idx.tolist()))
    flatten_reqs_idx_map = {i: [] for i in unique_flatten_reqs} # flatten_reqs_idx_map: img_idx -> idx of flatten_reqs
    for idx, val in enumerate(flatten_reqs_idx.tolist()):
        flatten_reqs_idx_map[val].append(idx)

    # compute the requested negatives in mini-batches
    unique_flatten_reqs = torch.tensor(unique_flatten_reqs, dtype=torch.int64)
    batch_iter = BatchSampler(SequentialSampler(range(len(unique_flatten_reqs))), batch_size=compute_batch_size, drop_last=False)

    for bat in batch_iter:
        compute_neg_idx = unique_flatten_reqs[bat].tolist()
        neg_images = torch.stack([data[k][0] for k in compute_neg_idx])

        repeat_n_emb = [len(flatten_reqs_idx_map[v]) for v in compute_neg_idx]

        _, neg_img_emb = model_inference(model, neg_images, dev)

        neg_img_emb = repeat_rows(neg_img_emb, repeat_n_emb)

        reidx = []
        for idx in compute_neg_idx:
            reidx += flatten_reqs_idx_map[ idx ]
        flatten_response_embs[ reidx ] = neg_img_emb
    
    return flatten_response_embs


def network_sync_samples(idx, dev):
    local_batch_size = torch.tensor(len(idx), dtype=torch.int64, device=dev)
    dist.all_reduce(local_batch_size, op=dist.ReduceOp.MIN)
    truncated_bs = local_batch_size.item()
    return truncated_bs
