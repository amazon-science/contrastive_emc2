"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import SequentialSampler, BatchSampler

from evaluation import evaluate, compute_gradient_error, compute_gradient_norm

from arguments import parse_args
from setup import setup
from utils import model_inference, aug_image_inference, adjust_learning_rate, repeat_rows, network_sync_samples, sync_gradient, eval_round

# from matplotlib.image import imsave

from main_simclr import simclr_loss


def metropolis_hasting_sampling_loss(neg_batch_size, model, aug_dataset, aug_images, idx, aug_idx, raw_data, config, criterion, timer, sampler, dev, ep, rank, world_size):
    backward_losses = []
    # number of positive samples in this iteration
    global_pos_batch = torch.tensor(len(idx), dtype=torch.int64, device=dev)
    dist.all_reduce(global_pos_batch, op=dist.ReduceOp.SUM)
    global_pos_batch = global_pos_batch.item()

    with timer("computation.1a", epoch=ep, rank=rank):
        aug_image_features = aug_image_inference(model, aug_images, dev) # [aug_batch_size, pos_batch_size, embd_size]
        trans_batch, pos_batch, _ = aug_image_features.shape
        neg_batch_size = (global_pos_batch-1) * trans_batch
        burn_in_rate = config["mcmc_burn_in"] # constant burn in rate
        if ep > config["stop_burn_in_ep"]:
            burn_in_rate = 0
        mcmc_burn_in = int( burn_in_rate * neg_batch_size )

        if config["alpha"] < 1:
            backward_losses.append( (1 - config["alpha"]) * simclr_loss(aug_image_features, criterion, config["beta"], config, dev))
            
        aug_img_feature_0, aug_img_feature_1 = aug_image_features[0], aug_image_features[1]

        image_features = None

        aug_image_features = aug_image_features.reshape(trans_batch * pos_batch, -1)

    # negative sampling for each positive pair
    if config["finite_aug"]:
        neg_pair_inners, _ = sampler.negative_sample_img_finite_aug(model, aug_dataset, image_features, aug_image_features, idx.to(dev), aug_idx.to(dev), pos_batch, trans_batch, mcmc_burn_in, config["beta"], timer, ep, rank)
    else:
        neg_pair_inners, oob_tup = sampler.negative_sample_img(image_features, aug_image_features, idx.to(dev), pos_batch, trans_batch, mcmc_burn_in, config["beta"], timer, ep, rank)
    
    # in-batch negative gradient
    backward_losses.append( config["alpha"] * world_size * config["beta"] * torch.sum(neg_pair_inners) / (neg_batch_size-mcmc_burn_in) / config["transform_batch_size"] / global_pos_batch )

    with timer("computation.1b", epoch=ep, rank=rank):
        # positive sample gradient
        backward_losses.append( config["alpha"] * world_size * -config["beta"] * torch.sum(aug_img_feature_0 * aug_img_feature_1) / global_pos_batch)

    if config["finite_aug"]:
        # out-of-batch gradient is handled during sampling
        return backward_losses

    # stage 1: request stage
    # request for negative sample embeddings
    with timer("communication.1", epoch=ep, rank=rank):
        embd_size = config["dim"]
        send_buffers = [{"aug_idx_list": [], "freq_list": [], "req_list": [], "embd_list": []} for _ in range(world_size)]
        for i, (aug_img_emb, target_global_idx, freq) in enumerate(zip(aug_image_features, oob_tup[0], oob_tup[1])):
            if freq > mcmc_burn_in:
                neighbor_rank, local_idx = sampler.to_rank_local_idx(target_global_idx)
                send_buffers[neighbor_rank]["aug_idx_list"].append(i) # the local aug batch index
                send_buffers[neighbor_rank]["req_list"].append(local_idx) # the global index
                send_buffers[neighbor_rank]["freq_list"].append(freq - mcmc_burn_in) # how many times this pair appears as negative pair
                send_buffers[neighbor_rank]["embd_list"].append(aug_img_emb.detach())
        
        
    with timer("communication.2", epoch=ep, rank=rank):
        # all_to_all send_counts so each agent know how many recv they expect from each neighbor
        send_counts = [torch.tensor(len(send_buffers[j]["req_list"]), dtype=torch.int64, device=dev ) for j in range(world_size)]
        recv_counts = [torch.tensor(0, dtype=torch.int64, device=dev) for _ in range(world_size)]
        dist.all_to_all(recv_counts, send_counts)

        send_aug_idx_list = [torch.tensor(send_buffers[r]["aug_idx_list"], dtype=torch.int64, device=dev) for r in range(world_size)]
        send_req_list = [torch.tensor(send_buffers[r]["req_list"], dtype=torch.int64, device=dev) for r in range(world_size)]
        send_freq_list = [torch.tensor(send_buffers[r]["freq_list"], dtype=torch.int64, device=dev) for r in range(world_size)]
        send_embd_list = [torch.vstack(send_buffers[r]["embd_list"]) if send_counts[r] > 0 else torch.empty((0,embd_size), dtype=aug_image_features[0].dtype, device=dev) for r in range(world_size)]

        recv_req_list = [torch.empty(recv_counts[r], dtype=torch.int64, device=dev) for r in range(world_size)]
        recv_freq_list = [torch.empty(recv_counts[r], dtype=torch.int64, device=dev) for r in range(world_size)]
        recv_embd_list = [torch.empty((recv_counts[r], embd_size), dtype=aug_image_features[0].dtype, device=dev) for r in range(world_size)]
        
        dist.all_to_all(recv_req_list, send_req_list)
        dist.all_to_all(recv_freq_list, send_freq_list)
        dist.all_to_all(recv_embd_list, send_embd_list)
        
        
    # stage 2: respond stage
    with timer("computation.2", epoch=ep, rank=rank):
        # compute requests
        flatten_reqs = torch.cat(recv_req_list)
        flatten_freqs = torch.cat(recv_freq_list)
        flatten_embs = torch.cat(recv_embd_list)
        flatten_response_embs = torch.zeros_like(flatten_embs, dtype=torch.float32)

        # compute inference of the repeated requests.
        unique_flatten_reqs = list(set(flatten_reqs.tolist()))
        flatten_reqs_idx_map = {i: [] for i in unique_flatten_reqs} # flatten_reqs_idx_map: img_idx -> idx of flatten_reqs
        for idxx, val in enumerate(flatten_reqs.tolist()):
            flatten_reqs_idx_map[val].append(idxx)

        # compute the requested negatives in mini-batches
        unique_flatten_reqs = torch.tensor(unique_flatten_reqs, dtype=torch.int64)
        batch_iter = BatchSampler(SequentialSampler(range(len(unique_flatten_reqs))), batch_size=config["compute_batch_size"], drop_last=False)

        for bat in batch_iter:
            compute_neg_idx = unique_flatten_reqs[bat].tolist()
            neg_images = torch.stack([raw_data[k][0] for k in compute_neg_idx])

            recved_emb = []
            batch_flatten_idx = []
            for j in compute_neg_idx: # the requested idx, i.e., the computed batch's indices
                for k in flatten_reqs_idx_map[j]: # the corresponding idx in flatten_reqs
                    recved_emb.append(flatten_embs[k] * flatten_freqs[k]) # this neg pair appears flatten_freqs[k] times
                    batch_flatten_idx.append(k)
            recved_emb = torch.stack(recved_emb)
            repeat_n_emb = [len(flatten_reqs_idx_map[v]) for v in compute_neg_idx]

            _, neg_img_emb = model_inference(model, neg_images, dev)

            neg_img_emb = repeat_rows(neg_img_emb, repeat_n_emb)

            if config["grad_accum"]:
                # backward to free graph in memory
                ( config["alpha"] * world_size * config["beta"] * torch.sum(neg_img_emb * recved_emb ) / neg_batch_size / config["transform_batch_size"] / global_pos_batch ).backward()
            else:
                backward_losses.append( config["alpha"] * world_size * config["beta"] * torch.sum(neg_img_emb * recved_emb ) / neg_batch_size / config["transform_batch_size"] / global_pos_batch )
            

            reidx = []
            for idxx in compute_neg_idx:
                reidx += flatten_reqs_idx_map[ idxx ]
            flatten_response_embs[ reidx ] = neg_img_emb
        
        
    with timer("communication.3", epoch=ep, rank=rank):
        # send the response
        response_embd_tensor = flatten_response_embs

        recv_response_embd_tensor = torch.empty((sum(send_counts), embd_size), dtype=torch.float32, device=dev)

        dist.all_to_all_single(recv_response_embd_tensor, response_embd_tensor, torch.stack(send_counts).tolist(), torch.stack(recv_counts).tolist())


    with timer("computation.3", epoch=ep, rank=rank):
        # stage 3: complete negative sample's gradient
        if len(recv_response_embd_tensor) > 0:
            flatten_idx = torch.cat( send_aug_idx_list )
            flatten_freqs = torch.cat( send_freq_list )
            flatten_embs = recv_response_embd_tensor * flatten_freqs.unsqueeze(1) 

            # neighbor responded with an image negative sample, paired with my aug_image sample
            backward_losses.append( config["alpha"] * world_size * config["beta"] * torch.sum(aug_image_features[flatten_idx] * flatten_embs) / neg_batch_size / config["transform_batch_size"] / global_pos_batch )

    return backward_losses



def run(local_rank, rank, world_size):
    torch.autograd.set_detect_anomaly(True)
    config = parse_args(world_size)
    name_prefix = "mcmc"
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
                    
                backward_losses += metropolis_hasting_sampling_loss((config["pos_batch_size"] * config["world_size"] - 1) * 2, model, data, aug_images, idx, aug_idx, raw_data, config, criterion, timer, sampler, dev, ep, rank, world_size)
                acceptance_rates.append( sampler.acceptance_rate )

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

                
            if rank == 0 and _it % config["display_freq"] == 0:
                print(timer.summary())


            # evaluate performance
            if eval_round(config, ep, _t, epoch_n_batches):
                with timer("evaluation", epoch=ep, rank=rank):
                    extra_stats = {"mean_acceptance_rate": torch.mean(torch.tensor(acceptance_rates)).item()}
                    acceptance_rates = []
                
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


# def test(local_rank, rank, world_size):
#     grad_accum=False
#     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
#     torch.manual_seed(0)
#     torch.cuda.manual_seed(0)
#     torch.cuda.manual_seed_all(0)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False 
#     torch.use_deterministic_algorithms(True)

#     dev = torch.device('cuda:{}'.format(local_rank) if torch.cuda.is_available() else 'cpu')
    
#     linear_model = torch.nn.Linear(world_size, 1, bias=False).to(dev)
#     linear_model = torch.nn.parallel.DistributedDataParallel(linear_model, device_ids=[dev])
#     optimizer = torch.optim.SGD(linear_model.parameters(), 1, weight_decay=0)

#     datas = [torch.randn(world_size) for _ in range(5)]
#     if grad_accum:
#         for data in datas:
#             loss = linear_model(data)
#             loss.backward()
#     else:
#         backward_losses = [linear_model(data) for data in datas]
#         torch.autograd.backward( backward_losses )
#     if rank == 0:
#         print(linear_model.module.weight.grad.clone().detach().cpu().numpy())
#     optimizer.step()


def single_machine_main():
    size = 8
    processes = []
    mp.set_start_method("spawn")
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29530'
    
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

    