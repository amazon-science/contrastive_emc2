import torch
import torch.distributed as dist
import bisect
from torch.utils.data import DataLoader, Dataset

import numpy as np
from communication import all_gather_matrices
from random import shuffle
# import difflogic

# cpu implementation of boolean matrix multiplication:
# https://stackoverflow.com/questions/14580950/fast-multiplication-of-k-x-k-boolean-matrices-where-8-k-16

def estimate_inner_prod(bit_inner, dim: int):
    return torch.cos( torch.abs(torch.pi - 2 * torch.pi * bit_inner / dim) )
    
class IndexedTable(Dataset):
    def __init__(self, x, packbits=False):
        self._data = x
        self.packbits = packbits
    
    def __getitem__(self, idx):
        if self.packbits:
            # packbits/unpackbits still not available in pytorch https://github.com/pytorch/pytorch/issues/42223
            bool_embd = np.unpackbits(self._data[idx]) # uint8 of {0, 1}
            # can do with cupy https://discuss.pytorch.org/t/convert-torch-tensors-directly-to-cupy-tensors/2752/5
        else:
            # without memory saving
            bool_embd = self._data[idx] # uint8 of {0, 1}

        return bool_embd, idx
    
    def __len__(self):
        return len(self._data)
    

class EmbeddingCache():
    def __init__(self, source_dim, proj_dim, network_num_samples, beta, device=torch.device("cpu"), gumbel_batch_size=8192, img_only=False,
                disable_proj=True, disable_mem_saving=True, expnorm_trick=True, finite_n_aug=None):
        assert proj_dim % 8 == 0, "please use proj_dim that is multiple of 8 for implementation simplicity."
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.beta = beta
        self.gumbel_batch_size = gumbel_batch_size
        self.img_only = img_only
        self.disable_proj = disable_proj
        self.disable_mem_saving = disable_mem_saving
        self.expnorm_trick = expnorm_trick
        self.device = device
        self.finite_n_aug = finite_n_aug
        torch.manual_seed(0)
        self.W = torch.randn(source_dim, proj_dim).to(device)

        # statistics of network dataset
        base_idxs = torch.cumsum(torch.tensor(network_num_samples), dim=0)
        self.base_idxs = torch.cat((torch.tensor([0]), base_idxs))[:-1].tolist()
        self.network_num_samples = [n.item() for n in network_num_samples]

        if self.rank == 0:
            print("data distribution:", self.network_num_samples)

        # tables are indexed by: self.base_idxs[node_rank] + local_sample_idx
        if self.disable_mem_saving:
            table_dtype = torch.float32 # bfloat16
        else:
            # memory saving by packbits
            proj_dim = proj_dim // 8
            table_dtype = torch.uint8

        if self.disable_proj:
            # disable random projection, i.e., self.W = I
            assert self.disable_mem_saving, "cannot save memory without random projection"
            table_dtype = torch.float32
            proj_dim = source_dim
            self.W = None
        
        # each agent only store the local cache table
        table_size = self.network_num_samples[self.rank]
        table_device, self.table_pin_memory = device, False
        if self.finite_n_aug is None:
            table_dim = (table_size, proj_dim)
        else:
            table_dim = (table_size, self.finite_n_aug, proj_dim)
        
        self.img_table = torch.empty(table_dim, dtype=table_dtype, device=table_device, pin_memory=self.table_pin_memory)
        if not self.img_only:
            self.text_table = torch.empty(table_dim, dtype=table_dtype, device=table_device, pin_memory=self.table_pin_memory)

        # list of local index sorted by staleness in reverse order
        self.staleness_queue = list(range(self.network_num_samples[self.rank]))
        shuffle(self.staleness_queue)
        # self.init_tableloaders()
    

    def to_table_idx(self, local_idx):
        # local_idx == table_idx because we cache the whole local dataset
        return local_idx
    
    def to_table_idx_unbound(self, local_idx):
        # local_idx == table_idx because we cache the whole local dataset
        return local_idx
    
    def to_global_idx(self, rank, local_idx):
        # convert (rank, local_idx) to global index
        return self.base_idxs[rank] + local_idx


    def to_rank_local_idx(self, global_idx):
        # convert global index to (rank, local_idx)
        rank = bisect.bisect_right(self.base_idxs, global_idx) - 1
        local_idx = global_idx - self.base_idxs[rank]
        return rank, local_idx


    def update_cache_table(self, idxs, image_embeddings, text_embeddings):
        if not image_embeddings is None:
            proj_img = self.project(image_embeddings).to(self.img_table.device)
            self.img_table[idxs] = proj_img
        if not text_embeddings is None:
            proj_text = self.project(text_embeddings).to(self.text_table.device)
            self.text_table[idxs] = proj_text


    def project(self, vecs):
        if self.disable_proj:
            return vecs
        
        # random project vec [m, d] to binary [m, p]
        bin_signs = vecs @ self.W >= 0
        
        if self.disable_mem_saving:
            # without memory saving
            bin_signs = bin_signs.to(torch.float32)
        else:
            # with memory saving
            # convert binary [m, p] to uint8 [m, p//8]
            # bin_signs = torch.from_numpy(np.packbits(bin_signs.cpu(), axis=-1))
            # bin_signs = difflogic.PackBitsTensor(bin_signs.T, bit_count=8).T
            raise NotImplementedError("packbits on cuda not implemented")
        return bin_signs
    


    def compute_weights(self, aug_queries, queries_local_idx, trans_batch):
        """
        compute p_{ij} for i in aug_queries, j in queries_local_idx
        """
        queries_table_entry = self.img_table[queries_local_idx.to(self.img_table.device)].to(aug_queries.device)
        proj_aug_queries = self.project(aug_queries)
        network_proj_aug_queries = all_gather_matrices(proj_aug_queries, self.device)

        # nominator
        network_proj_aug_queries = [mat for mat in network_proj_aug_queries]
        queries_table_entry = queries_table_entry
        score_buffer = [torch.zeros((len(aug_queries), len(neigh_q)//trans_batch), device=self.device) for neigh_q in network_proj_aug_queries]
        for r, net_proj_aug_query in enumerate(network_proj_aug_queries):
            r_scores = net_proj_aug_query @ queries_table_entry.T
            r_scores = r_scores if self.disable_proj else estimate_inner_prod(r_scores, queries_table_entry.shape[-1])
            buf = score_buffer if self.rank == r else None
            dist.gather(r_scores, buf, dst=r)
        
        nominator_scores = self.beta * torch.cat(score_buffer, axis=1) # [pos_batch * transform_batch, pos_batch * transform_batch]

        return nominator_scores


    def negative_sample_img(self, neg_batch_size, queries, queries_local_idx):
        proj_img = self.project(queries)
        pos_scores = torch.sum(queries * queries, dim=1)
        neg_samples, cumu_weights = self.gumbel_max_sampling_distributed(self.img_table, proj_img, queries_local_idx, neg_batch_size, self.device, pos_scores)
        return neg_samples, cumu_weights.detach()

    # def negative_sample_text(self, neg_batch_size, queries, queries_local_idx):
    #     proj_text = self.project(queries)
    #     neg_samples, pos_inner, normalizing_logsumexp = self.gumbel_max_sampling_distributed(self.text_table, proj_text, queries_local_idx, neg_batch_size, self.device)
    #     pos_weight = torch.exp(pos_inner - normalizing_logsumexp)
    #     return neg_samples, pos_weight.clone().detach()

    def from_extended_idx_to_idx_aug_idx(self, ext_idx, n_aug):
        aug_idx = ext_idx % n_aug
        idx = torch.floor(ext_idx / n_aug)
        return idx, aug_idx



    def gumbel_max_sampling_distributed(self, table_loader, queries, queries_local_idx, neg_batch_size, dev, pos_scores=None):
        """
            perform gumbel max sampling over distributed gumbel scores, return global indices
        """
        network_queries = all_gather_matrices(queries, dev)

        network_n_queries = [len(q) for q in network_queries]
        pre_i, post_i = sum(network_n_queries[:self.rank]), sum(network_n_queries[:self.rank+1])
        query_filter_idx = torch.empty(sum(network_n_queries), dtype=torch.int64, device=dev).fill_(-1)
        query_filter_idx[pre_i : post_i] = queries_local_idx

        network_queries = torch.cat(network_queries, dim=0) # [n_queries, dim]
        if self.finite_n_aug is None:
            local_table_idx, gumbel_vals, local_max_scores, normalizing_sumexp, a_shift = self.gumbel_max_sampling_batched(table_loader, network_queries, neg_batch_size, query_filter_idx, pre_i, post_i, pos_scores)
        else:
            local_table_idx, gumbel_vals, local_max_scores, normalizing_sumexp, a_shift = self.gumbel_max_sampling_batched(table_loader.reshape(-1, table_loader.shape[-1]), network_queries, neg_batch_size, query_filter_idx, pre_i, post_i, pos_scores)
            local_table_idx, aug_idx = self.from_extended_idx_to_idx_aug_idx(local_table_idx, self.finite_n_aug)

        global_idx = self.to_global_idx(self.rank, local_table_idx)
        global_idx = global_idx.contiguous()
        gumbel_vals = gumbel_vals.contiguous()
        local_max_scores = local_max_scores.contiguous()

        buffer_global_idx, buffer_vals, buffer_scores = \
            [torch.empty_like(global_idx) for _ in range(self.world_size)], \
            [torch.empty_like(gumbel_vals) for _ in range(self.world_size)], \
            [torch.empty_like(local_max_scores) for _ in range(self.world_size)]
        

        # get global sumexp
        dist.all_reduce(normalizing_sumexp, op=dist.ReduceOp.SUM) # [world_n_queries]
        

        # gather global intermediate gumbel (index, value) scores
        dist.all_gather(buffer_global_idx, global_idx)
        dist.all_gather(buffer_vals, gumbel_vals)
        dist.all_gather(buffer_scores, local_max_scores)

        if self.finite_n_aug is not None:
            buffer_aug_idx = [torch.empty_like(aug_idx) for _ in range(self.world_size)]
            dist.all_gather(buffer_aug_idx, aug_idx)

        # filter the local positive sample part
        network_indices = [idxs[pre_i : post_i] for idxs in buffer_global_idx]
        network_gumbal_vals = [vals[pre_i : post_i] for vals in buffer_vals]
        network_max_scores = [sc[pre_i : post_i] for sc in buffer_scores]
            
        # now each agent has the intermediate gumbel (index, value) scores, sampled among all negatives in all cache tables
        network_indices = torch.stack(network_indices) # [world_size, pos_batch_size, neg_batch_size]
        network_gumbal_vals = torch.stack(network_gumbal_vals) # [world_size, pos_batch_size, neg_batch_size]
        network_max_scores = torch.stack(network_max_scores) # [world_size, pos_batch_size, neg_batch_size]

        if self.finite_n_aug is not None:
            network_aug_idx = [aidx[pre_i : post_i] for aidx in buffer_aug_idx]
            network_aug_idx = torch.stack(network_aug_idx) # [world_size, pos_batch_size, neg_batch_size]
        
        # compute positive pairs' scores
        if pos_scores is None:
            # pos_score will be estimated by projected embeddings
            pos_bit_inner = torch.sum(queries * queries, dim=1)
            pos_scores = pos_bit_inner if self.disable_proj else estimate_inner_prod(pos_bit_inner, queries.shape[-1]) # [pos_batch_size]


        _, max_idxs = torch.max(network_gumbal_vals, 0) # max along the first axis (world_size axis), [pos_batch_size, neg_batch_size]

        _, pos_bs, neg_bs = network_indices.shape
        sample_indices = torch.empty((pos_bs, neg_bs), dtype=torch.int64, device=dev)
        sample_scores = torch.empty((pos_bs, neg_bs), dtype=network_max_scores.dtype, device=dev)
        if self.finite_n_aug is not None:
            sample_aug_indices = torch.empty((pos_bs, neg_bs), dtype=torch.int64, device=dev)
        for i in range(pos_bs):
            for j in range(neg_bs):
                sample_indices[i,j] = network_indices[max_idxs[i,j], i,j]
                sample_scores[i,j] = network_max_scores[max_idxs[i,j], i,j]
                if self.finite_n_aug is not None:
                    sample_aug_indices[i,j] = network_aug_idx[max_idxs[i,j], i,j]

        beta_pos_scores = self.beta * pos_scores
        if isinstance(self, StreamingCache):
            beta_pos_scores += self.logalpha_tensor
            raise NotImplementedError("pls check how to reweight samples without replacemnet in streaming cache.")

        # compute exp(score_ii)
        if self.expnorm_trick:
            exp_pos_scores = torch.exp(beta_pos_scores - a_shift[pre_i: post_i])
        else:
            exp_pos_scores = torch.exp(beta_pos_scores)
        
        # compute exp(score_iJ), J the sample index
        if self.expnorm_trick:
            sample_exp = torch.exp(self.beta * sample_scores - a_shift[pre_i: post_i].unsqueeze(1))
        else:
            sample_exp = torch.exp(self.beta * sample_scores)
        sample_exp = torch.cat((exp_pos_scores.unsqueeze(1), sample_exp), dim=1) # append pos sample's score in first column

        # compute \sum_{J \in [neg_samples]} p_{iJ}
        torch.use_deterministic_algorithms(False)
        cumu_weights = torch.cumsum(sample_exp, dim=1)
        torch.use_deterministic_algorithms(True)

        cumu_weights = cumu_weights / normalizing_sumexp[pre_i : post_i].unsqueeze(1)

        cumu_weights = cumu_weights[:, :-1] # drop the last column because it will not be used
        if self.finite_n_aug is not None:
            return (sample_indices, sample_aug_indices), cumu_weights # ([pos_batch_size, neg_batch_size], [pos_batch_size, neg_batch_size]), [pos_batch_size, neg_batch_size]
        else:
            return sample_indices, cumu_weights # [pos_batch_size, neg_batch_size], [pos_batch_size, neg_batch_size]


    def gumbel_max_sampling_batched(self, table, queries, neg_batch_size, queries_filter_idx, pre_i, post_i, local_pos_scores=None):
        """ 
            draw [pos_batch_size, neg_batch_size] negative sample indices 
        """
        pos_batch_size = len(queries)
        neg_infty = torch.tensor(float("-inf"), dtype=torch.float32, device=self.device)
        
        # compute inner product of bool
        # we implement as matrix multiplication of float32 due to pytorch lack of support to uint8 matrix mul.
        table_idx = torch.arange(len(table), device=self.device)
        bit_inner = queries @ table.to(queries.dtype).T # [pos_batch_size, table_size]
        
        approx_inner_prod = bit_inner if self.disable_proj else estimate_inner_prod(bit_inner, table.shape[-1])

        # https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        if self.expnorm_trick:
            _scores = self.beta * approx_inner_prod
            _max_scores, _ = torch.max(_scores, -1) # [pos_batch_size]
            dist.all_reduce(_max_scores, op=dist.ReduceOp.MAX)
            a_shift = _max_scores
        else:
            a_shift = torch.zeros(pos_batch_size, device=self.device)
        # normalizing_sumexp = torch.sum(torch.exp(self.beta * approx_inner_prod.to(torch.float64) - a_shift.unsqueeze(1)), dim=1) 
        normalizing_sumexp = torch.sum(torch.exp(self.beta * approx_inner_prod - a_shift.unsqueeze(1)), dim=1) 

        # compute gumbel scores; draw Gumbel(0,1) by https://github.com/pytorch/pytorch/issues/97851.
        # [pos_batch_size, table_size] + [neg_batch_size, pos_batch_size, table_size]
        # -> [1, pos_batch_size, table_size] + [neg_batch_size, pos_batch_size, table_size]
        # -> [repeat neg_batch_size, pos_batch_size, table_size] + [neg_batch_size, pos_batch_size, table_size]
        score = self.beta * approx_inner_prod \
                - torch.empty((neg_batch_size, pos_batch_size, len(table_idx)), memory_format=torch.legacy_contiguous_format, device=self.device).exponential_().log() # [neg_batch_size, pos_batch_size, table_size]; see https://stackoverflow.com/questions/51371070/how-does-pytorch-broadcasting-work.


        # lower the score of the positive pair (x_i, x_i) so that we do not sample (x_i, x_i)
        queries_table_idx = self.to_table_idx_unbound(queries_filter_idx).to(self.device)
        if self.finite_n_aug is None:
            pos_batch_dim_idx, gumbel_batch_dim_idx = torch.where(queries_table_idx.unsqueeze(1) == table_idx)
        else:
            # lower the scores of all its augmentations
            pos_batch_dim_idx, gumbel_batch_dim_idx = torch.where(queries_table_idx.unsqueeze(1) == torch.floor(table_idx / self.finite_n_aug) )
        score[:, pos_batch_dim_idx, gumbel_batch_dim_idx] += neg_infty

        # remove the score of the positive pair (x_i, x_i) from normalizing_sumexp because 1. we have exact values for local_pos_scores 2. it is weighted differently in streaming cache denominator.
        normalizing_sumexp[pos_batch_dim_idx] -= torch.exp( self.beta * approx_inner_prod[pos_batch_dim_idx, gumbel_batch_dim_idx] - a_shift[pos_batch_dim_idx])
    
        if local_pos_scores is None:
            # pos_score will be estimated by projected embeddings
            pos_inner = torch.sum(queries * queries, dim=1) # [pos_batch_size]
            local_pos_scores = pos_inner if self.disable_proj else estimate_inner_prod(pos_inner, table.shape[-1])
            
        # add the score of the positive pair (x_i, x_i) using local_pos_scores 
        if isinstance(self, StreamingCache):
            normalizing_sumexp[pre_i: post_i] += torch.exp( self.logalpha_tensor + self.beta * local_pos_scores - a_shift[pre_i: post_i] )
        else:
            normalizing_sumexp[pre_i: post_i] += torch.exp( self.beta * local_pos_scores - a_shift[pre_i: post_i] )

        max_val = torch.zeros((neg_batch_size, pos_batch_size), dtype=score.dtype, device=score.device)
        max_table_idxs = torch.zeros((neg_batch_size, pos_batch_size), dtype=torch.int64, device=score.device)
        max_scores = torch.zeros((neg_batch_size, pos_batch_size), dtype=score.dtype, device=score.device)
        # perform distributed gumbel-max sampling without replacement
        buffer_vals = [torch.empty(pos_batch_size, dtype=score.dtype, device=self.device) for _ in range(self.world_size)]
        for j in range(neg_batch_size):
            val, idxs = torch.max(score[j], -1) # max along the table_size dimension, gives [pos_batch_size]
            max_val[j] = val
            max_table_idxs[j] = idxs
            max_scores[j] = approx_inner_prod[torch.arange(pos_batch_size), idxs]

            dist.all_gather(buffer_vals, val) # [world_size, pos_batch_size]
            gathered_vals = torch.stack(buffer_vals)
            _, max_ranks = torch.max(gathered_vals, 0) # max along the first axis (world_size axis), [pos_batch_size]
            where_max = torch.where(max_ranks == self.rank)[0]
            score[:,where_max,idxs[where_max]] += neg_infty # remove the score of sampled index to perform sampling without replacement
        
        return max_table_idxs.T, max_val.T, max_scores.T, normalizing_sumexp, a_shift # return [pos_batch_size, neg_batch_size], [pos_batch_size, neg_batch_size], [pos_batch_size, neg_batch_size], [pos_batch_size], [pos_batch_size]
    
    

class StreamingCache(EmbeddingCache):
    def __init__(self, source_dim, proj_dim, network_num_samples, alpha, beta, device=torch.device("cpu"), gumbel_batch_size=8192, img_only=False,
                 disable_proj=True, disable_mem_saving=True, expnorm_trick=True):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.alpha = alpha # cache fraction
        self.logalpha_tensor = torch.tensor(alpha, device=device).log()
        self.beta = beta
        self.gumbel_batch_size = gumbel_batch_size
        self.img_only = img_only
        self.disable_proj = disable_proj
        self.disable_mem_saving = disable_mem_saving
        self.expnorm_trick = expnorm_trick
        self.device = device
        torch.manual_seed(0)
        self.W = torch.randn(source_dim, proj_dim).to(device)

        # statistics of network dataset
        base_idxs = torch.cumsum(torch.tensor(network_num_samples), dim=0)
        self.base_idxs = torch.cat((torch.tensor([0]), base_idxs))[:-1].tolist()
        self.network_num_samples = [n.item() for n in network_num_samples]

        if self.rank == 0:
            print("data distribution:", self.network_num_samples)

        # tables are indexed by: self.base_idxs[node_rank] + local_sample_idx
        if self.disable_mem_saving:
            table_dtype = torch.float32 # bfloat16
        else:
            # memory saving by packbits
            proj_dim = proj_dim // 8
            table_dtype = torch.uint8
        
        if self.disable_proj:
            # disable random projection, i.e., self.W = I
            assert self.disable_mem_saving, "cannot save memory without random projection"
            table_dtype = torch.float32
            proj_dim = source_dim
            self.W = None
            
        # each agent only store the local cache table
        table_size = int(self.alpha * network_num_samples[self.rank])
        # table_device, self.table_pin_memory = torch.device("cpu"), True
        table_device, self.table_pin_memory = device, False
        
        self.img_table = torch.empty((table_size, proj_dim), dtype=table_dtype, device=table_device, pin_memory=self.table_pin_memory)
        if not self.img_only:
            self.text_table = torch.empty((table_size, proj_dim), dtype=table_dtype, device=table_device, pin_memory=self.table_pin_memory)

        shuffled_idx = torch.randperm(self.network_num_samples[self.rank]).tolist()

        self.cached_idx = shuffled_idx[: self.img_table.shape[0] ]
        self.uncached_idx = shuffled_idx[self.img_table.shape[0] : ]
        self.refresh_mapping()
        # self.init_tableloaders()


    
    def refresh_mapping(self):
        # the reverse index map of self.cached_idx
        # self.cached_map: local_idx -> table_idx
        self.cached_map = {self.cached_idx[i]: i for i in range(len(self.cached_idx))}


    def to_table_idx(self, local_idx):
        # convert local_idx to table_idx
        return torch.tensor([self.cached_map[i] for i in local_idx.tolist()], dtype=torch.int64, device=self.device)
    
    def to_table_idx_unbound(self, local_idx):
        # convert local_idx to table_idx, return -1 when local_idx is not in cache
        return torch.tensor([self.cached_map.get(i, -1) for i in local_idx.tolist()], dtype=torch.int64, device=self.device)


    def to_global_idx(self, rank, table_idx):
        # convert (rank, table_idx) to global index
        if rank == self.rank:
            local_idx = torch.tensor(self.cached_idx, device=self.device)[table_idx]
            return self.base_idxs[rank] + local_idx
        else:
            raise ValueError("cannot do this in streaming cache.")

    
    def drop_rows(self, n_rows):
        # drop the first n rows of table
        self.img_table = self.img_table[ n_rows: ]
        if not self.img_only:
            self.text_table = self.text_table[ n_rows: ]
        self.cached_idx = self.cached_idx[ n_rows: ]
    

    def append_rows(self, idxs, imgs_embds, text_embds):
        # put new rows at the end of table
        if not imgs_embds is None:
            proj_img = self.project(imgs_embds).to(self.img_table.device)
            self.img_table = torch.cat([self.img_table, proj_img])
        if not text_embds is None:
            proj_text = self.project(text_embds).to(self.text_table.device)
            self.text_table = torch.cat([self.text_table, proj_text])
        self.cached_idx += idxs.tolist()
    
    def update_cache_table(self, idxs, image_embeddings, text_embeddings):
        if not image_embeddings is None:
            proj_img = self.project(image_embeddings).to(self.img_table.device)
            table_idx = self.to_table_idx_unbound(idxs)
            mask = table_idx != -1
            self.img_table[table_idx[mask]] = proj_img[mask]
        if not text_embeddings is None:
            proj_text = self.project(text_embeddings).to(self.text_table.device)
            table_idx = self.to_table_idx_unbound(idxs)
            mask = table_idx != -1
            self.text_table[table_idx[mask]] = proj_text[mask]