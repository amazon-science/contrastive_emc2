"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import SequentialSampler, BatchSampler, SubsetRandomSampler, DataLoader

from cache import StreamingCache
from evaluation import evaluate, compute_gradient_error, compute_gradient_norm

from arguments import parse_args
from setup import setup
from utils import model_inference, aug_image_inference, adjust_learning_rate, repeat_rows, network_sync_samples, sync_gradient, eval_round
import random

# from matplotlib.image import imsave

from main_simclr import simclr_loss

def refresh_cache(model, data, raw_data, timer, config, embds_cache, rho, ep, rank, dev):
    with timer("caching", epoch=ep, rank=rank):
        # update cache table
        model.eval()
        if isinstance(embds_cache, StreamingCache):
            # remove stale rows
            refresh_size = int(rho * len(embds_cache.cached_idx))
            stale_idxs = embds_cache.cached_idx[ : refresh_size]
            embds_cache.drop_rows(refresh_size)
            embds_cache.uncached_idx += stale_idxs
            
            # draw refresh indices
            random.shuffle(embds_cache.uncached_idx)
            fresh_idxs = embds_cache.uncached_idx[ : refresh_size]
            embds_cache.uncached_idx = embds_cache.uncached_idx[ refresh_size : ]

            # update table with fresh_idxs
            refresh_loader = DataLoader(raw_data, shuffle=False, batch_size=config["compute_batch_size"], sampler=SubsetRandomSampler(fresh_idxs))
            for image, _, _, idx in refresh_loader:
                _, image_features = model_inference(model, image, dev)
                embds_cache.append_rows(idx, image_features, None)
            embds_cache.refresh_mapping()
        else:
            refresh_size = int(rho * len(embds_cache.staleness_queue))
            stale_idxs = embds_cache.staleness_queue[: refresh_size]
            if not config["finite_aug"]:
                refresh_loader = DataLoader(raw_data, shuffle=False, batch_size=config["compute_batch_size"], sampler=SubsetRandomSampler(stale_idxs))
                for image, _, _, idx, _ in refresh_loader:
                    _, image_features = model_inference(model, image, dev)
                    embds_cache.update_cache_table(idx, image_features, None)
            else:
                stale_idxs_tensor = torch.tensor(stale_idxs, dtype=torch.int64)
                with timer("caching.3", epoch=ep, rank=rank):
                    batch_iter = BatchSampler(SequentialSampler(range(len(stale_idxs_tensor))), batch_size=config["compute_batch_size"], drop_last=False)
                    for bat in batch_iter:
                        for _idx in stale_idxs_tensor[bat]:
                            aug_images = []
                            _idx_l, _aidx_l = [], []
                            for _aug_idx in range(data.n_augmentations):
                                aug_images.append(data.get_by_aug_idx(_idx, _aug_idx))
                                _idx_l.append(_idx)
                                _aidx_l.append(_aug_idx)
                            aug_images = torch.stack(aug_images)
                            _, image_features = model_inference(model, aug_images, dev)
                            embds_cache.update_cache_table((_idx_l, _aidx_l), image_features, None)
            embds_cache.staleness_queue = embds_cache.staleness_queue[refresh_size: ] + stale_idxs


def gumbel_max_sampling_losses(neg_batch_size, model, image, aug_images, idx, aug_idx, data, raw_data, config, criterion, timer, embds_cache, dev, ep, rank, world_size):
    backward_losses = []
    # number of positive samples in this iteration
    global_pos_batch = torch.tensor(len(idx), dtype=torch.int64, device=dev)
    dist.all_reduce(global_pos_batch, op=dist.ReduceOp.SUM)
    global_pos_batch = global_pos_batch.item()
    

    with timer("computation.1a", epoch=ep, rank=rank):
        aug_image_features = aug_image_inference(model, aug_images, dev) # [aug_batch_size, pos_batch_size, embd_size]

        if config["alpha"] < 1:
            backward_losses.append( (1 - config["alpha"]) * simclr_loss(aug_image_features, criterion, config["beta"], config, dev))
        
        trans_batch, pos_batch, _ = aug_image_features.shape
        
        _, image_features = model_inference(model, image, dev)

        # aug_img_feature_0, aug_img_feature_1 = aug_image_features[0], aug_image_features[1]
        aug_image_features = aug_image_features.reshape(trans_batch * pos_batch, -1)

        # greedy update the cache
        if config["finite_aug"]:
            embds_cache.update_cache_table((idx.unsqueeze(0).expand(trans_batch, -1).flatten(), aug_idx), aug_image_features.detach(), None)
        else:
            embds_cache.update_cache_table(idx, image_features.detach(), None)

    # negative sampling for each positive pair
    with timer("sampling", epoch=ep, rank=rank):
        with torch.no_grad():
            img_neg_samples_global_idxs, cumu_weights = embds_cache.negative_sample_img(neg_batch_size, aug_image_features, idx.unsqueeze(0).expand(trans_batch, -1).flatten().to(dev))



    with timer("computation.1b", epoch=ep, rank=rank):
        # positive sample gradient
        pos_weights = cumu_weights[:,0] # first column contains the positive pair weight
        weighted_aug_image_features = aug_image_features * (1 - pos_weights).unsqueeze(1) # reweight L_i's positive gradient, see Section 3.3 in https://papers.nips.cc/paper/2021/file/2175f8c5cd9604f6b1e576b252d4c86e-Paper.pdf
        repeat_image_features = image_features.unsqueeze(0).expand(trans_batch, -1, -1).reshape(-1, config["dim"]) # [aug_batch_size * pos_batch_size, embd_size]
        assert weighted_aug_image_features.shape == repeat_image_features.shape, "check shape: {}, {}".format(weighted_aug_image_features.shape, repeat_image_features.shape)
        backward_losses.append( config["alpha"] * world_size * -embds_cache.beta * torch.sum(repeat_image_features * weighted_aug_image_features) / global_pos_batch / config["transform_batch_size"] )


    # stage 1: request stage
    # request for negative sample embeddings
    with timer("communication.1", epoch=ep, rank=rank):
        embd_size = config["dim"]
        send_buffers = [{"aug_idx_list": [], "neg_order_idx_list":[], "req_list": [], "embd_list": [], "reweight_list": [], "sample_score_list": []} for _ in range(world_size)]
        for i, aug_img_emb in enumerate(aug_image_features):
            if config["finite_aug"]:
                neg_rank_idx = [embds_cache.to_rank_local_idx(idxx) for idxx in img_neg_samples_global_idxs[0][i]]
                neg_aug_idx = img_neg_samples_global_idxs[1][i]
            else:
                neg_rank_idx = [embds_cache.to_rank_local_idx(idxx) for idxx in img_neg_samples_global_idxs[i]]

            for j, (neighbor_rank, local_idx) in enumerate(neg_rank_idx):
                send_buffers[neighbor_rank]["aug_idx_list"].append(i) # the local aug batch index
                send_buffers[neighbor_rank]["neg_order_idx_list"].append(j) # the local aug batch neg idx, i.e., this is the j-th negative sample of the i-th positive sample
                if config["finite_aug"]:
                    send_buffers[neighbor_rank]["req_list"].append([local_idx, neg_aug_idx[j]]) # the local,aug index
                else:
                    send_buffers[neighbor_rank]["req_list"].append(local_idx) # the local index
                send_buffers[neighbor_rank]["embd_list"].append(( 1 - cumu_weights[i,j] ) * aug_img_emb.detach())
                send_buffers[neighbor_rank]["reweight_list"].append( 1 - cumu_weights[i,j] )
                # if loaded_ref_prm: send_buffers[neighbor_rank]["sample_score_list"].append( current_prm_sample_scores[i,j] )
        
        
    with timer("communication.2", epoch=ep, rank=rank):
        # all_to_all send_counts so each agent know how many recv they expect from each neighbor
        send_counts = [torch.tensor(len(send_buffers[j]["aug_idx_list"]), dtype=torch.int64, device=dev ) for j in range(world_size)]
        recv_counts = [torch.tensor(0, dtype=torch.int64, device=dev) for _ in range(world_size)]
        dist.all_to_all(recv_counts, send_counts)

        send_aug_idx_list = [torch.tensor(send_buffers[r]["aug_idx_list"], dtype=torch.int64, device=dev) for r in range(world_size)]
        send_neg_order_idx_list = [torch.tensor(send_buffers[r]["neg_order_idx_list"], dtype=torch.int64, device=dev) for r in range(world_size)]
        send_req_list = [torch.tensor(send_buffers[r]["req_list"], dtype=torch.int64, device=dev) for r in range(world_size)]
        send_embd_list = [torch.vstack(send_buffers[r]["embd_list"]) if send_counts[r] > 0 else torch.empty((0,embd_size), dtype=aug_image_features[0].dtype, device=dev) for r in range(world_size)]
        send_reweight_list = [torch.tensor(send_buffers[r]["reweight_list"], dtype=cumu_weights.dtype, device=dev) for r in range(world_size)]

        recv_aug_idx_list = [torch.empty(recv_counts[r], dtype=torch.int64, device=dev) for r in range(world_size)]
        recv_neg_order_idx_list = [torch.empty(recv_counts[r], dtype=torch.int64, device=dev) for r in range(world_size)]
        if config["finite_aug"]:
            recv_req_list = [torch.empty((recv_counts[r], 2), dtype=torch.int64, device=dev) for r in range(world_size)] # local_idx, aug_idx
        else:
            recv_req_list = [torch.empty(recv_counts[r], dtype=torch.int64, device=dev) for r in range(world_size)]
        recv_embd_list = [torch.empty((recv_counts[r], embd_size), dtype=aug_image_features[0].dtype, device=dev) for r in range(world_size)]
        recv_reweight_list = [torch.empty(recv_counts[r], dtype=cumu_weights.dtype, device=dev) for r in range(world_size)]

        dist.all_to_all(recv_aug_idx_list, send_aug_idx_list)
        dist.all_to_all(recv_neg_order_idx_list, send_neg_order_idx_list)
        dist.all_to_all(recv_req_list, send_req_list)
        dist.all_to_all(recv_embd_list, send_embd_list)
        dist.all_to_all(recv_reweight_list, send_reweight_list)
        
        
    # stage 2: respond stage
    with timer("computation.2", epoch=ep, rank=rank):
        # compute requests
        flatten_reqs = torch.cat(recv_req_list)
        flatten_embs = torch.cat(recv_embd_list)
        flatten_reweights = torch.cat(recv_reweight_list)
        flatten_response_embs = torch.zeros_like(flatten_embs, dtype=torch.float32)
        
        # compute inference of the repeated requests.
        _all_reqs = flatten_reqs.tolist()
        if config["finite_aug"]:
            _all_reqs = [tuple(r) for r in _all_reqs] # list of tuples
        unique_flatten_reqs = list(set(_all_reqs)) 
        flatten_reqs_idx_map = {i: [] for i in unique_flatten_reqs} # flatten_reqs_idx_map: img_idx -> idx of flatten_reqs
        for idxx, val in enumerate(_all_reqs):
            flatten_reqs_idx_map[val].append(idxx)

        # compute the requested negatives in mini-batches
        unique_flatten_reqs = torch.tensor(unique_flatten_reqs, dtype=torch.int64)
        batch_iter = BatchSampler(SequentialSampler(range(len(unique_flatten_reqs))), batch_size=config["compute_batch_size"], drop_last=False)

        for bat in batch_iter:
            compute_neg_idx = unique_flatten_reqs[bat].tolist()
            if config["finite_aug"]:
                compute_neg_idx = [tuple(i) for i in compute_neg_idx]
                neg_images = torch.stack([data.get_by_aug_idx(_idx, _aug_idx) for _idx, _aug_idx in compute_neg_idx])
            else:
                neg_images = torch.stack([raw_data[k][0] for k in compute_neg_idx])

            recved_emb = []
            batch_flatten_idx = []
            for j in compute_neg_idx: # the requested idx, i.e., the computed batch's indices
                for k in flatten_reqs_idx_map[j]: # the corresponding idx in flatten_reqs
                    recved_emb.append(flatten_embs[k])
                    batch_flatten_idx.append(k)
            recved_emb = torch.stack(recved_emb)
            repeat_n_emb = [len(flatten_reqs_idx_map[v]) for v in compute_neg_idx]

            _, neg_img_emb = model_inference(model, neg_images, dev)

            neg_img_emb = repeat_rows(neg_img_emb, repeat_n_emb)


            # compute scores on reference prm, i.e., exp(beta * Q(phi(x_J; prm_{ref}), phi(y_i; prm_{ref})))
            batch_reweight = torch.ones(len(neg_img_emb), device=dev)

            if config["grad_accum"]:
                # backward to free graph in memory
                ( config["alpha"] * world_size * embds_cache.beta * torch.sum(neg_img_emb * recved_emb * batch_reweight.unsqueeze(1)) / neg_batch_size / config["transform_batch_size"] / global_pos_batch ).backward()
            else:
                backward_losses.append( config["alpha"] * world_size * embds_cache.beta * torch.sum(neg_img_emb * recved_emb * batch_reweight.unsqueeze(1)) / neg_batch_size / config["transform_batch_size"] / global_pos_batch )
            

            reidx = []
            for idxx in compute_neg_idx:
                reidx += flatten_reqs_idx_map[ idxx ]
            flatten_response_embs[ reidx ] = neg_img_emb
        
        flatten_response_embs = flatten_response_embs * flatten_reweights.unsqueeze(1) # reweight for sampling without replacement
    
        
    with timer("communication.3", epoch=ep, rank=rank):
        # send the response
        response_aug_idx_list = recv_aug_idx_list
        response_neg_order_idx_list = recv_neg_order_idx_list
        response_embd_tensor = flatten_response_embs

        recv_response_aug_idx_list = [torch.empty(send_counts[r], dtype=torch.int64, device=dev) for r in range(world_size)]
        recv_response_neg_order_idx_list = [torch.empty(send_counts[r], dtype=torch.int64, device=dev) for r in range(world_size)]
        recv_response_embd_tensor = torch.empty((sum(send_counts), embd_size), dtype=torch.float32, device=dev)

        dist.all_to_all(recv_response_aug_idx_list, response_aug_idx_list)
        dist.all_to_all(recv_response_neg_order_idx_list, response_neg_order_idx_list)
        dist.all_to_all_single(recv_response_embd_tensor, response_embd_tensor, torch.stack(send_counts).tolist(), torch.stack(recv_counts).tolist())


    with timer("computation.3", epoch=ep, rank=rank):
        # stage 3: complete negative sample's gradient
        flatten_idx = torch.cat(recv_response_aug_idx_list)
        flatten_neg_order_idx = torch.cat(recv_response_neg_order_idx_list)
        flatten_embs = recv_response_embd_tensor
        flatten_sample_score_reweights = torch.ones(len(flatten_embs), device=dev)

        # compute scores on current prm, i.e., exp(beta * Q(phi(x_J; prm_t), phi(y_i; prm_t)))
        current_prm_sample_scores = torch.zeros((len(aug_image_features), neg_batch_size), device=dev)
        reshaped_neg_embs = torch.zeros((len(aug_image_features), neg_batch_size, len(flatten_embs[0])), device=dev)
        for i in range(len(flatten_idx)):
            reshaped_neg_embs[ flatten_idx[i], flatten_neg_order_idx[i] ] = flatten_embs[i]
        repeated_aug_image_features = aug_image_features.unsqueeze(1).expand(-1, neg_batch_size, -1)
        assert reshaped_neg_embs.shape == repeated_aug_image_features.shape, "check shape: {} != {}".format(reshaped_neg_embs.shape, repeated_aug_image_features.shape)

        current_prm_sample_scores = torch.sum(repeated_aug_image_features * reshaped_neg_embs, axis=-1) # [pos_batch_size * trans_batch_size]
        current_prm_sample_scores = torch.exp(config["beta"] * current_prm_sample_scores)

        # neighbor responded with an image negative sample, paired with my aug_image sample
        # if config["grad_accum"]:
        #     ( config["alpha"] * world_size * embds_cache.beta * torch.sum(aug_image_features[flatten_idx] * flatten_embs) / neg_batch_size / config["transform_batch_size"] / global_pos_batch ).backward()
        # else:
        backward_losses.append( config["alpha"] * world_size * embds_cache.beta * torch.sum(aug_image_features[flatten_idx] * flatten_embs * flatten_sample_score_reweights.unsqueeze(1)) / neg_batch_size / config["transform_batch_size"] / global_pos_batch )
    
    return backward_losses


def run(local_rank, rank, world_size):
    torch.autograd.set_detect_anomaly(True)
    config = parse_args(world_size)
    if config["cache_frac"] < 1:
        name_prefix = "streaming-gumbel" 
    else:
        name_prefix = "gumbel"
    
    optimizer, model, data, raw_data, dataloader, rawdataloader, le_rawdataloader, le_rawtestdataloader, knn_dataloader, dev, _, global_num_samples, epoch_n_batches, iterator, timer, sampler = setup(config, local_rank, rank, world_size, wandb_prefix=name_prefix)
    criterion = torch.nn.CrossEntropyLoss().to(dev)

    _it = 0
    mean_gerr = []
    mean_loss = []
    acceptance_rates = []
    # training
    for ep in iterator:
        # handles uneven local dataset size by reusing local samples on nodes with less data, same as DistributedSampler(drop_last=False).
        train_iter = iter(dataloader)
        for _t in range(epoch_n_batches):
            with model.no_sync(): # basically we disable DDP because our multi forward multi backward system does not benefit from DDP.
                backward_losses = []
                with timer("dataloading", epoch=ep, rank=rank):
                    try:
                        image, aug_images, _, idx, aug_idx = next(train_iter)
                    except StopIteration:
                        train_iter = iter(dataloader)
                        image, aug_images, _, idx, aug_idx = next(train_iter)

                    if config["finite_aug"]:
                        aug_idx = torch.cat(aug_idx)
                    
                    # if rank == 3:
                    #     imsave('images/test.png', aug_images[1][1].permute(1, 2, 0).numpy())
                    # exit()
                    
                    # keep the batch size consistent on every gpu.
                    truncated_bs = network_sync_samples(idx, dev)
                    image = image[:truncated_bs]
                    aug_images = [augs[:truncated_bs] for augs in aug_images]
                    idx = idx[:truncated_bs]

                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    adjust_learning_rate(optimizer, ep + _t/epoch_n_batches, config)
                    
                backward_losses += gumbel_max_sampling_losses(config["neg_batch_size"], model, image, aug_images, idx, aug_idx, data, raw_data, config, criterion, timer, sampler, dev, ep, rank, world_size)

                with timer("computation.gradient", epoch=ep, rank=rank):
                    torch.autograd.backward( backward_losses )
        
                if config["check_gradient_error"]:
                    gerr, loss = compute_gradient_error(optimizer, world_size, model, data, dev, config)
                    mean_gerr.append(gerr)
                    mean_loss.append(loss)
            # ============ end of DDP no_sync ============

            with timer("communication.gradient", epoch=ep, rank=rank):
                sync_gradient(optimizer, world_size)
                
            # https://discuss.pytorch.org/t/explain-adam-state-when-using-ddp/83096 DDP also sync optimizer states
            
            optimizer.step()
            _it += 1

            with torch.no_grad():
                refresh_cache(model, data, raw_data, timer, config, sampler, config["rho"], ep, rank, dev)
                    
            if rank == 0 and _it % config["display_freq"] == 0:
                print(timer.summary())

            # evaluate performance
            if eval_round(config, ep, _t, epoch_n_batches):
                with timer("evaluation", epoch=ep, rank=rank):
                    extra_stats = {}
                
                    if config["check_gradient_error"]:
                        gnorm, gloss = compute_gradient_norm(optimizer, world_size, model, data, dev, config)
                        mean_gerr = (sum(mean_gerr) / len(mean_gerr)).item()
                        mean_loss = (sum(mean_loss) / len(mean_loss)).item()
                        extra_stats = {**extra_stats, **{"gradient_error": mean_gerr, "running_loss": mean_loss, "gradient_norm": gnorm.item(), "finite_aug_loss": gloss.item()}}
                        mean_gerr = []
                        mean_loss = []
                    print("entering evaluate")
                    evaluate(_it, ep, rank, world_size, global_num_samples, model, data, dataloader, rawdataloader, le_rawdataloader, le_rawtestdataloader, knn_dataloader, dev, config, extra_stats=extra_stats)

    if rank == 0:
        import wandb
        wandb.finish()
    dist.destroy_process_group()


def init_process(local_rank, rank, size, fn):
    """ Initialize the distributed environment. """
    if torch.cuda.is_available():
        backend = "nccl"
    else:
        backend = "gloo"
    
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(local_rank, rank, size)


def test(local_rank, rank, world_size):
    grad_accum=False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    torch.use_deterministic_algorithms(True)

    dev = torch.device('cuda:{}'.format(local_rank) if torch.cuda.is_available() else 'cpu')
    
    linear_model = torch.nn.Linear(world_size, 1, bias=False).to(dev)
    linear_model = torch.nn.parallel.DistributedDataParallel(linear_model, device_ids=[dev])
    optimizer = torch.optim.SGD(linear_model.parameters(), 1, weight_decay=0)

    datas = [torch.randn(world_size) for _ in range(5)]
    if grad_accum:
        for data in datas:
            loss = linear_model(data)
            loss.backward()
    else:
        backward_losses = [linear_model(data) for data in datas]
        torch.autograd.backward( backward_losses )
    if rank == 0:
        print(linear_model.module.weight.grad.clone().detach().cpu().numpy())
    optimizer.step()


def single_machine_main():
    size = 8
    processes = []
    mp.set_start_method("spawn")
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, rank, size, run))
        # p = mp.Process(target=init_process, args=(rank, rank, size, test))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    

def mpi_main():
    LOCAL_RANK = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    WORLD_RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

    # print("node {} alive".format(WORLD_RANK))
    init_process(LOCAL_RANK, WORLD_RANK, WORLD_SIZE, run)


if __name__ == "__main__":
    single_machine_main()
    # mpi_main()

    