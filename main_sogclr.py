"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from arguments import parse_args
from setup import setup
from utils import aug_image_inference, adjust_learning_rate, adjust_beta, network_sync_samples, eval_round
from evaluation import evaluate, compute_gradient_error, compute_gradient_norm
from communication import all_gather_layer, all_gather_vectors


class SogCLR():
    def __init__(self, network_num_samples, device, T=1.0, finite_n_aug=None):
        if finite_n_aug is None:
            self.u = torch.zeros(sum(network_num_samples), device=device).reshape(-1, 1)
        else:
            self.u = torch.zeros((sum(network_num_samples), finite_n_aug), device=device).reshape(-1, 1)
        self.T = T # softmax temperature
        self.LARGE_NUM = 1e9
        self.device = device
        base_idxs = torch.cumsum(torch.tensor(network_num_samples), dim=0)
        self.base_idxs = torch.cat((torch.tensor([0]), base_idxs))[:-1].tolist()

    def to_global_idx(self, rank, local_idx):
        # convert (rank, local_idx) to global index
        return self.base_idxs[rank] + local_idx
    
    def dynamic_contrastive_loss(self, hidden1, hidden2, index, aug_idx, gamma=0.9, distributed=True, finite_aug=False):
        # Get (normalized) hidden1 and hidden2.
        hidden1, hidden2 = F.normalize(hidden1, p=2, dim=1), F.normalize(hidden2, p=2, dim=1)
        batch_size = hidden1.shape[0]

        # Gather hidden1/hidden2 across replicas and create local labels.
        if distributed:  
            hidden1_large = torch.cat(all_gather_layer.apply(hidden1), dim=0)
            hidden2_large =  torch.cat(all_gather_layer.apply(hidden2), dim=0)
            enlarged_batch_size = hidden1_large.shape[0]

            labels_idx = torch.arange(enlarged_batch_size, dtype=torch.long)

            labels = F.one_hot(labels_idx, enlarged_batch_size*2).to(self.device) 
            batch_size = enlarged_batch_size
        else:
            hidden1_large = hidden1
            hidden2_large = hidden2
            labels = F.one_hot(torch.arange(batch_size, dtype=torch.long), batch_size * 2).to(self.device) 

        """each agent should compute the whole logits matrix, because u_i is different across the rows."""

        
        logits_aa = torch.matmul(hidden1_large, hidden1_large.T) # (b * world_size, b * world_size)
        logits_bb = torch.matmul(hidden2_large, hidden2_large.T)
        logits_ab = torch.matmul(hidden1_large, hidden2_large.T)
        logits_ba = torch.matmul(hidden2_large, hidden1_large.T)

        #  SogCLR
        neg_mask = 1-labels
        logits_ab_aa = torch.cat([logits_ab, logits_aa ], 1) # neg. pairs inner product, (b * world_size, 2 * b * world_size)
        logits_ba_bb = torch.cat([logits_ba, logits_bb ], 1)

        
        
        neg_logits1 = torch.exp(logits_ab_aa /self.T)*neg_mask   #(B, 2B)
        neg_logits2 = torch.exp(logits_ba_bb /self.T)*neg_mask

        neg_logits1[:, batch_size:].fill_diagonal_(0) # replaces the role of LARGE_NUM
        neg_logits2[:, batch_size:].fill_diagonal_(0) # replaces the role of LARGE_NUM

        if distributed:
            network_indices = all_gather_vectors(index, self.device)
            index = torch.cat(network_indices, dim=0)
            if finite_aug:
                network_aug_idx = all_gather_vectors(aug_idx, self.device)
                aug_idx = torch.cat(network_aug_idx, dim=0)

        if finite_aug:
            u_index = (index, aug_idx)
        else:
            u_index = index
        # u init    
        if self.u[u_index].sum() == 0:
            gamma = 1
        
        u1 = (1 - gamma) * self.u[u_index] + gamma * torch.sum(neg_logits1, dim=1, keepdim=True)/(2*(batch_size-1))
        u2 = (1 - gamma) * self.u[u_index] + gamma * torch.sum(neg_logits2, dim=1, keepdim=True)/(2*(batch_size-1))

        self.u[u_index] = (u1.detach() + u2.detach())/2 

        p_neg_weights1 = (neg_logits1/u1).detach()
        p_neg_weights2 = (neg_logits2/u2).detach()

        def softmax_cross_entropy_with_logits(labels, logits, weights):
            expsum_neg_logits = torch.sum(weights*logits, dim=1, keepdim=True)/(2*(batch_size-1))
            normalized_logits = logits - expsum_neg_logits
            return -torch.sum(labels * normalized_logits, dim=1)

        loss_a = softmax_cross_entropy_with_logits(labels, logits_ab_aa, p_neg_weights1)
        loss_b = softmax_cross_entropy_with_logits(labels, logits_ba_bb, p_neg_weights2)
        loss = (loss_a + loss_b).mean()

        return loss


def run(local_rank, rank, world_size):
    config = parse_args(world_size)

    optimizer, model, data, _, dataloader, rawdataloader, le_rawdataloader, le_rawtestdataloader, knn_dataloader, dev, network_num_samples, global_num_samples, epoch_n_batches, iterator, timer, _ = setup(config, local_rank, rank, world_size, wandb_prefix="SogCLR")

    sogclr = SogCLR(network_num_samples, dev, T=1/config["beta"])

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
                    _, aug_images, _, idx, aug_idx = next(train_iter)
                except StopIteration:
                    train_iter = iter(dataloader)
                    _, aug_images, _, idx, aug_idx = next(train_iter)

                if config["finite_aug"]:
                    aug_idx = torch.cat(aug_idx)

                # keep the batch size consistent on every gpu.
                truncated_bs = network_sync_samples(idx, dev)
                aug_images = [augs[:truncated_bs] for augs in aug_images]
                idx = idx[:truncated_bs]

            adjust_learning_rate(optimizer, ep + _t/epoch_n_batches, config)
            if config["lower_beta"] > 0 and config["upper_beta"] > 0 and config["temp_period"] > 0:
                sogclr.T = 1 / adjust_beta(ep + _t/epoch_n_batches, config)
            else:
                sogclr.T = 1 / config["beta"]
            with timer("computation.1", epoch=ep):
                aug_image_features = aug_image_inference(model, aug_images, dev) # [transform_batch_size, pos_batch_size, embd_size]
                global_idx = sogclr.to_global_idx(rank, idx.to(dev))
                loss = sogclr.dynamic_contrastive_loss(aug_image_features[0], aug_image_features[1], global_idx, aug_idx, gamma=config["sogclr_gamma"], distributed=world_size > 1)
                loss *= world_size # DDP average the gradient
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
    size = 8
    processes = []
    mp.set_start_method("spawn")
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29520'
    
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

    # print("node {} alive".format(WORLD_RANK))
    init_process(LOCAL_RANK, WORLD_RANK, WORLD_SIZE, run)


if __name__ == "__main__":
    single_machine_main()
    # mpi_main()
