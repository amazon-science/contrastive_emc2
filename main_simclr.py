"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from evaluation import evaluate, compute_gradient_error, compute_gradient_norm

# import matplotlib.pyplot as plt
# from matplotlib.image import imsave

from arguments import parse_args
from setup import setup
from utils import aug_image_inference, adjust_learning_rate, network_sync_samples, eval_round
from communication import all_gather_layer


def simclr_loss(aug_image_features, criterion, beta, config, dev):
    trans_batch, pos_batch, _ = aug_image_features.shape
    aug_image_features = aug_image_features.view(trans_batch * pos_batch, -1)  # order: [x_1, x_2, ..., x_n, x_1', x_2', ..., x_n']
    bs = trans_batch * pos_batch

    network_features_1 = torch.cat(all_gather_layer.apply(aug_image_features[:bs//2])) 
    network_features_2 = torch.cat(all_gather_layer.apply(aug_image_features[bs//2:]))
    network_features = torch.cat([network_features_1, network_features_2])
    network_bs = network_features.shape[0]

    loss = info_nce_loss(network_features, criterion, dev, 1/beta, network_bs//2)
    loss *= config["world_size"] # DDP average the gradient
    return loss


def info_nce_loss(features, criterion, device, temperature=1, batch_size=256, n_views=2):
    """
    from https://github.com/sthalles/SimCLR
    """
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    # features = torch.nn.functional.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     n_views * batch_size, n_views * batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    loss = criterion(logits, labels)
    return loss


def run(local_rank, rank, world_size):
    config = parse_args(world_size)

    exp_prefix = "SimCLR"

    optimizer, model, data, _, dataloader, rawdataloader, le_rawdataloader, le_rawtestdataloader, knn_dataloader, dev, _, global_num_samples, epoch_n_batches, iterator, timer, _ = setup(config, local_rank, rank, world_size, wandb_prefix=exp_prefix)

    criterion = torch.nn.CrossEntropyLoss().to(dev)
    
    _it = 0
    mean_gerr = []
    mean_loss = []
    # training
    for ep in iterator:
        # handles uneven local dataset size by reusing local samples on nodes with less data, same as DistributedSampler(drop_last=False).
        model.train()
        train_iter = iter(dataloader)
        for _t in range(epoch_n_batches):
            with timer("dataloading", epoch=ep):
                try:
                    _, aug_images, _, idx, _ = next(train_iter)
                except StopIteration:
                    train_iter = iter(dataloader)
                    _, aug_images, _, idx, _ = next(train_iter)
                
                # keep the batch size consistent on every gpu.
                truncated_bs = network_sync_samples(idx, dev)
                aug_images = [augs[:truncated_bs] for augs in aug_images]
                idx = idx[:truncated_bs]
            
            # single machine
            # imsave('images/test_0.png', _imgs[0].permute(1, 2, 0).numpy())
            # imsave('images/test_aug_0.png', aug_images[0][0].permute(1, 2, 0).numpy())
            # imsave('images/test_16.png', _imgs[16].permute(1, 2, 0).numpy())
            # imsave('images/test_aug_16.png', aug_images[0][16].permute(1, 2, 0).numpy())
            # exit()

            # 2 machines
            # imsave('images/rank_{}_test_0.png'.format(rank), _imgs[0].permute(1, 2, 0).numpy())
            # imsave('images/rank_{}_test_aug_0.png'.format(rank), aug_images[0][0].permute(1, 2, 0).numpy())
            # exit()

            adjust_learning_rate(optimizer, ep + _t/epoch_n_batches, config)
            beta = config["beta"]
            with timer("computation.loss", epoch=ep):
                ## modifying https://github.com/sthalles/SimCLR implementation
                aug_image_features = aug_image_inference(model, aug_images, dev) # [transform_batch_size, pos_batch_size, embd_size]

                loss = simclr_loss(aug_image_features, criterion, beta, config, dev)

                optimizer.zero_grad()
                loss.backward()

                if config["check_gradient_error"]:
                    gerr, loss = compute_gradient_error(optimizer, world_size, model, data, dev, config)
                    mean_gerr.append(gerr)
                    mean_loss.append(loss)
                    
                optimizer.step()
                _it += 1
            
            if rank == 0 and _it % config["display_freq"] == 0:
                print(timer.summary())
        

            # evaluate performance
            if eval_round(config, ep, _t, epoch_n_batches):
                with timer("evaluation", epoch=ep):
                    if config["check_gradient_error"]:
                        gnorm, gloss = compute_gradient_norm(optimizer, world_size, model, data, dev, config)
                        mean_gerr = (sum(mean_gerr) / len(mean_gerr)).item()
                        mean_loss = (sum(mean_loss) / len(mean_loss)).item()
                        extra_stats = {"gradient_error": mean_gerr, "running_loss": mean_loss, "gradient_norm": gnorm.item(), "finite_aug_loss": gloss.item()}
                        mean_gerr = []
                        mean_loss = []
                    else:
                        extra_stats = {}
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



def single_machine_main():
    size = 1
    processes = []
    mp.set_start_method("spawn")
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'

    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    

def mpi_main():
    LOCAL_RANK = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    WORLD_RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])

    init_process(LOCAL_RANK, WORLD_RANK, WORLD_SIZE, run)
    # init_process(LOCAL_RANK, WORLD_RANK, WORLD_SIZE, test)


def test(local_rank, rank, world_size, lr):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    torch.use_deterministic_algorithms(True)

    dev = torch.device('cuda:{}'.format(local_rank) if torch.cuda.is_available() else 'cpu')
    all_gather_local_differentiable = all_gather_layer()
    
    linear_model = torch.nn.Linear(world_size, 1, bias=False).to(dev)
    linear_model = torch.nn.parallel.DistributedDataParallel(linear_model, device_ids=[dev])
    optimizer = torch.optim.SGD(linear_model.parameters(), 1, weight_decay=0)

    print(rank, linear_model.module.weight.clone().detach().cpu().numpy())

    data = torch.eye(world_size, dtype=torch.float, device=dev)[rank]
    local_output = linear_model(data)
    network_output = torch.stack(all_gather_local_differentiable.apply(local_output))
    loss = network_output[(rank)%world_size]
    # loss = network_output[(rank-1)%world_size]
    loss.backward()
    optimizer.step()
    print(loss.item(), linear_model.module.weight.clone().detach().cpu().numpy(), linear_model.module.weight.grad.clone().detach().cpu().numpy())

if __name__ == "__main__":
    single_machine_main()
    # mpi_main()
