import torch
from torch import autograd
import torch.distributed as dist
from utils import model_inference
from communication import all_gather_layer, all_gather_vectors, all_gather_matrices

import bisect

def uniform_samples(self_idx, max_idx, dev, skip_self=False):
    new_samples = torch.randint(0, max_idx, (len(self_idx),), device=dev)
    if skip_self:
        while torch.any(new_samples == self_idx):
            new_samples[ new_samples == self_idx ] = torch.randint(0, max_idx, (sum( new_samples == self_idx ),), device=dev)
    return new_samples


def sample_from_distribution(discrete_dist, dev):
    out_tensor = torch.empty((len(discrete_dist), 1), dtype=torch.int64, device=dev)
    torch.multinomial(discrete_dist, 1, out=out_tensor)
    return out_tensor.flatten()

class SortedList():
    def __init__(self, max_size=10):
        self.ls = []
        self.members = set()
        self.max_size = max_size
    
    def find(self, idx):
        for ls_idx, (_, _j) in self.ls:
            if _j == idx:
                return ls_idx
    
    def insert(self, tup):
        # an item is removed from list if
            # it is lowest score and the list is full
        if tup[1] not in self.members:
            self.members.add(tup[1])
            bisect.insort_right(self.ls, tup)
            if len(self.ls) > self.max_size:
                _, i = self.ls.pop(0) # pop the low score
                self.members.remove(i)
        else:
            list_idx = self.find(tup[1])
            self.ls.pop(list_idx)
            self.members.remove(list_idx)
            self.insert(tup)

    def argmax(self):
        return self.ls[-1][1]


class MCMC_BatchController():
    def __init__(self, network_num_samples, rank, world_size, dev, finite_n_aug=None):
        self.rank = rank
        self.world_size = world_size
        self.device = dev

        # statistics of network dataset
        base_idxs = torch.cumsum(torch.tensor(network_num_samples), dim=0)
        self.base_idxs = torch.cat((torch.tensor([0]), base_idxs))[:-1].tolist()
        self.network_num_samples = [n.item() for n in network_num_samples]
        self.global_num_samples = sum(self.network_num_samples)

        local_num_samples = self.network_num_samples[rank]
        if finite_n_aug is None:
            self.scores = torch.ones(local_num_samples) * torch.finfo(torch.float32).smallest_normal
            self.last_sample_global_idx = -torch.ones(local_num_samples, dtype=torch.int64)

        else:
            # for finite support case
            self.state_score = torch.ones((local_num_samples, finite_n_aug)) * torch.finfo(torch.float32).smallest_normal
            self.states = -torch.ones((local_num_samples, finite_n_aug, 2), dtype=torch.int64)

    
    def to_rank_local_idx(self, global_idx):
        # convert global index to (rank, local_idx)
        rank = bisect.bisect_right(self.base_idxs, global_idx) - 1
        local_idx = global_idx - self.base_idxs[rank]
        return rank, local_idx

    def to_global_idx(self, rank, local_idx):
        # convert (rank, local_idx) to global index
        return self.base_idxs[rank] + local_idx
    
    def mh_step(self, queries_local_idx, negative_sample_scores, batch_idx, init_chain, burn_in):
        # please call mh_step twice, each for every augmentation. this corresponds to reusing the same markov chain for two augmentation.
        pos_batch_size, neg_batch_size = negative_sample_scores.shape
        _rand = torch.empty(pos_batch_size)
        if init_chain is None:
            neg_batch_idx_counter = -torch.ones(pos_batch_size, dtype=torch.int64)
        else:
            neg_batch_idx_counter = init_chain
        accepted_pairs = []
        local_score_table = self.scores[queries_local_idx]

        # neg_batch_idx_counter = neg_batch_idx_counter.cpu()
        # negative_sample_scores = negative_sample_scores.cpu()
        num_accepted_samples = []
        for j in range(neg_batch_size):
            new_sample_scores = negative_sample_scores[:, j]
            alphas = new_sample_scores / local_score_table
            accepted = _rand.uniform_() < alphas

            local_score_table[accepted] = new_sample_scores[accepted]
            neg_batch_idx_counter[accepted] = j
            skip_mask = new_sample_scores == 0 # this only happens for positive pairs, where score is set to 0. exp(x) cannot go 0 unless x -> -infty, while our features are bounded.
            if j > burn_in:
                accepted_pairs.append( torch.stack((batch_idx, neg_batch_idx_counter))[:,~skip_mask] ) # append( [2, neg_b] )
                num_accepted_samples.append(accepted)
        self.scores[queries_local_idx] = local_score_table
        accp_tensor = torch.hstack(accepted_pairs).T
        self.acceptance_rate = torch.mean(torch.cat(num_accepted_samples).to(torch.float32))
        return accp_tensor, neg_batch_idx_counter


    def negative_sample_img(self, queries, aug_queries, queries_local_idx, pos_batch, trans_batch, burn_in, beta, timer, ep, rank):
        assert trans_batch == 2, "this function is implemented for 2 augmentations only."
        assert queries is None, "this function does not use non-augmentation features"
        # all gather the augmented output embeddings
        with timer("sampling:comm", epoch=ep, rank=rank):
            bs = len(aug_queries)
            network_aug_features_1 = torch.cat(all_gather_layer.apply(aug_queries[:bs//2])) 
            network_aug_features_2 = torch.cat(all_gather_layer.apply(aug_queries[bs//2:]))
            network_aug_features = torch.cat([network_aug_features_1, network_aug_features_2])
            network_bs = network_aug_features.shape[0]

        # all gather the indicies
            comm_dev = queries_local_idx.device
            global_idx = self.to_global_idx(self.rank, queries_local_idx)
            recv_global_idx = [torch.zeros_like(global_idx, device=global_idx.device) for _ in range(self.world_size)]
            dist.all_gather(recv_global_idx, global_idx)
            network_global_idx = torch.cat(recv_global_idx)

        # change here
            network_global_idx = torch.hstack((network_global_idx, network_global_idx)).cpu() # repeat the indices for 2 augmentations
            queries_local_idx = queries_local_idx.cpu()
        with timer("sampling:comp_1", epoch=ep, rank=rank):
            # similarity_matrix contains the backward-able inner products
            # change here
            # similarity_matrix = torch.matmul(network_aug_features, network_features.T)
            similarity_matrix = torch.matmul(network_aug_features, network_aug_features.T)

            # remove positive pair
            mask = torch.eye(network_bs//2, dtype=torch.bool, device=self.device)

            # change here
            # similarity_matrix[:network_bs//2][mask] = 0 # constant 0 carries no gradient
            # similarity_matrix[network_bs//2:][mask] = 0 # constant 0 carries no gradient

            similarity_matrix[:network_bs//2,:network_bs//2][mask] = 0 # constant 0 carries no gradient
            similarity_matrix[:network_bs//2,network_bs//2:][mask] = 0 # constant 0 carries no gradient
            similarity_matrix[network_bs//2:,:network_bs//2][mask] = 0 # constant 0 carries no gradient
            similarity_matrix[network_bs//2:,network_bs//2:][mask] = 0 # constant 0 carries no gradient

        
        network_pos_batch = network_bs // trans_batch
        # Metropolis-Hasting over the in-batch negative samples
        # e.g.: for each positive sample, output [0,0,2,3,3,3,6,7] means that [0, 2, 3, 6, 7] are the accepted negatives (indicies are in-batch)
        with torch.no_grad():
            with timer("sampling:samp", epoch=ep, rank=rank):
                chain = None
                all_neg_pairs = [] # pairs of (batch_idx, batch_idx), i.e., in-batch negative pairs. indices are global batch index
                last_state_queue = []
                similarity_scores = torch.exp(beta * similarity_matrix).cpu()

                similarity_scores[:network_bs//2,:network_bs//2][mask] = 0 # score 0 must be rejected
                similarity_scores[:network_bs//2,network_bs//2:][mask] = 0 # score 0 must be rejected
                similarity_scores[network_bs//2:,:network_bs//2][mask] = 0 # score 0 must be rejected
                similarity_scores[network_bs//2:,network_bs//2:][mask] = 0 # score 0 must be rejected

                for j in range(trans_batch):
                    st, end = j*network_pos_batch + self.rank*pos_batch, j*network_pos_batch + (self.rank+1)*pos_batch
                    neg_pairs, chain = self.mh_step(queries_local_idx, 
                                                    similarity_scores[st:end], 
                                                    torch.arange(st, end),
                                                    chain, burn_in)
                    _oob = neg_pairs[:,1] == -1
                    _who_use_oob_last_state = neg_pairs[_oob][:,0]
                    _who_use_oob_last_state = _who_use_oob_last_state - st + j * pos_batch # to local aug batch index
                    last_state_queue.append(_who_use_oob_last_state)
                    all_neg_pairs.append(neg_pairs)
                # torch.use_deterministic_algorithms(False)
                oob_last_state_freq = torch.bincount( torch.cat(last_state_queue) )
                # torch.use_deterministic_algorithms(True)
                last_states = self.last_sample_global_idx[queries_local_idx]
                oob_tup = (torch.cat([last_states for _ in range(trans_batch)]), oob_last_state_freq)
                all_neg_pairs = torch.cat(all_neg_pairs)


                # update the markov chain state
                update_mask = chain != -1 # -1 means this chain does not update, i.e., all samples are rejected
                last_chain_state = network_global_idx[ chain[update_mask] ]
                self.last_sample_global_idx[ queries_local_idx[update_mask] ] = last_chain_state # the last chain state is stored for the next time we access this chain

            with timer("sampling:samp_comm", epoch=ep, rank=rank):
            # gather Metropolis-Hasting results
                all_neg_pairs = all_neg_pairs.to(comm_dev)
                # recv_accepted_idx = [torch.empty_like(all_neg_pairs) for _ in range(self.world_size)]
                # dist.all_gather(recv_accepted_idx, all_neg_pairs)
                global_batch_negs = all_gather_matrices(all_neg_pairs, comm_dev)
                global_batch_negs = torch.cat(global_batch_negs)
                global_batch_negs = global_batch_negs[global_batch_negs[:,1] != -1]

        with timer("sampling:reorder", epoch=ep, rank=rank):
            # reconstruct the Metropolis-Hasting results by reindexing and form the backward-able mcmc_similarity_matrix
            mcmc_similarity_matrix = similarity_matrix[ (global_batch_negs[:,0], global_batch_negs[:,1]) ]

        return mcmc_similarity_matrix, oob_tup

    def finite_state_mh_step(self, queries_local_idx, queries_aug_idx, negative_sample_scores, global_batch_idx, burn_in):
        pos_batch_size, neg_batch_size = negative_sample_scores.shape
        _rand = torch.empty(pos_batch_size)
        neg_batch_idx_counter = -torch.ones(pos_batch_size, dtype=torch.int64)
        accepted_pairs = []
        local_score_table = self.state_score[ [queries_local_idx, queries_aug_idx] ]

        # local_score_table = local_score_table.cpu()
        # negative_sample_scores = negative_sample_scores.cpu()
        num_accepted_samples = []
        for j in range(neg_batch_size):
            new_sample_scores = negative_sample_scores[:, j]
            alphas = new_sample_scores / local_score_table
            accepted = _rand.uniform_() < alphas

            local_score_table[accepted] = new_sample_scores[accepted]
            neg_batch_idx_counter[accepted] = j
            if j > burn_in:
                accepted_pairs.append( torch.stack((global_batch_idx, neg_batch_idx_counter)) )
                num_accepted_samples.append(accepted)
        self.state_score[ [queries_local_idx, queries_aug_idx] ] = local_score_table
        accp_tensor = torch.hstack(accepted_pairs).T
        self.acceptance_rate = torch.mean(torch.cat(num_accepted_samples).to(torch.float32))
        return accp_tensor, neg_batch_idx_counter
    

    def out_of_batch_compute(self, model, aug_dataset, network_states, feature_dim):
        """1. locally compute the responsible requests; 2. reorder into the batch index order."""
        with autograd.detect_anomaly():
            network_global_idx = network_states[:,0]
            network_local_rank_idx = torch.tensor( [self.to_rank_local_idx(idx) for idx in network_global_idx] )
            
            index_matrix = torch.cat( (network_local_rank_idx, network_states[:,1].unsqueeze(1)), axis=1) # contains [rank, local_idx, aug_idx]

            local_part = torch.where( index_matrix[:,0] == self.rank )[0]
            requested_local_idx = index_matrix[local_part][:,1].tolist()
            requested_aug_idx = index_matrix[local_part][:,2].tolist()

            if len(local_part) > 0:
                requested_img = torch.stack( [aug_dataset.get_by_aug_idx(local_i, aug_i) for local_i, aug_i in zip(requested_local_idx, requested_aug_idx) ] )
                _, requested_features = model_inference(model, requested_img, self.device)
            else:
                requested_features = torch.zeros( (0, feature_dim), device=self.device)
            if torch.any( torch.isnan(requested_features ) ):
                print("before", local_part)
                print("before", requested_features)
                exit()
            network_idx_remap = all_gather_vectors(local_part, self.device)
            network_features = all_gather_layer.apply( requested_features )
            if torch.any( torch.isnan(requested_features ) ):
                print("after", local_part)
                print("after", requested_features)
                exit()

            reordered_features = torch.empty((len(network_states), feature_dim), device=self.device)
            for r in range(self.world_size):
                reordered_features[ network_idx_remap[r] ] = network_features[r]
            
            return reordered_features


    def negative_sample_img_finite_aug(self, model, aug_dataset, queries, aug_queries, queries_local_idx, queries_aug_idx, pos_batch, trans_batch, burn_in, beta, timer, ep, rank):
        # roadmap: 
        #   1. compute features
        #   2. share feature and compute similarity matrix
        #   3. request out-of-batch inference (img_idx, aug_idx)
        #   4. burn-in with ratio r
        assert trans_batch == 2, "this function is implemented for 2 augmentations only."
        assert queries is None, "this function does not use non-augmentation features"

        # all gather the augmented output embeddings
        with timer("sampling:comm", epoch=ep, rank=rank):
            bs = len(aug_queries)
            network_aug_features_1 = torch.cat(all_gather_layer.apply(aug_queries[:bs//2])) 
            network_aug_features_2 = torch.cat(all_gather_layer.apply(aug_queries[bs//2:]))
            network_aug_features = torch.cat([network_aug_features_1, network_aug_features_2])
            network_bs = network_aug_features.shape[0]

            # compute the indices of the local part
            network_pos_batch = network_bs // trans_batch
            st, end = 0*network_pos_batch + self.rank*pos_batch, 0*network_pos_batch + (self.rank+1)*pos_batch
            aug_1_idx = torch.arange(st, end)
            st, end = 1*network_pos_batch + self.rank*pos_batch, 1*network_pos_batch + (self.rank+1)*pos_batch
            aug_2_idx = torch.arange(st, end)
            network_local_row_idx = torch.cat((aug_1_idx, aug_2_idx)).cpu()


            # all gather the indicies
            comm_dev = queries_local_idx.device
            global_idx = self.to_global_idx(self.rank, queries_local_idx)
            recv_global_idx = [torch.zeros_like(global_idx, device=global_idx.device) for _ in range(self.world_size)]
            dist.all_gather(recv_global_idx, global_idx)
            network_global_idx = torch.cat(recv_global_idx)

            network_aug_idx_1 = torch.cat(all_gather_vectors(queries_aug_idx[:bs//2], comm_dev))
            network_aug_idx_2 = torch.cat(all_gather_vectors(queries_aug_idx[bs//2:], comm_dev))
            network_aug_idx = torch.cat([network_aug_idx_1, network_aug_idx_2])
            network_aug_idx = network_aug_idx.cpu()

            network_global_idx = torch.cat((network_global_idx, network_global_idx)).cpu() # repeat the indices for 2 augmentations
            queries_local_idx = queries_local_idx.cpu()
            queries_aug_idx = queries_aug_idx.cpu()

        with timer("sampling:oob_compute", epoch=ep, rank=rank):
            network_current_state_1 = torch.cat(all_gather_matrices(self.states[ (queries_local_idx, queries_aug_idx[:bs//2]) ], comm_dev ))
            network_current_state_2 = torch.cat(all_gather_matrices(self.states[ (queries_local_idx, queries_aug_idx[bs//2:]) ], comm_dev ))
            network_states = torch.cat([network_current_state_1, network_current_state_2]).cpu()

            noninit_mask = network_states[:,0] == -1 # these chains are not initialized, so there's no need to compute its initial state sample.
            if len(noninit_mask) < network_bs:
                init_chain_features = self.out_of_batch_compute(model, aug_dataset, network_states[~noninit_mask], aug_queries.shape[-1])
                init_similarity = torch.sum(init_chain_features * network_aug_features[~noninit_mask], dim=1)
                init_similarity_scores = torch.exp(beta * init_similarity).cpu()
                self.state_score[ (torch.cat((queries_local_idx, queries_local_idx))[~noninit_mask], queries_aug_idx) ] = init_similarity_scores[~noninit_mask[network_local_row_idx]]


        with timer("sampling:comp_1", epoch=ep, rank=rank):
            # similarity_matrix contains the backward-able inner products
            similarity_matrix = torch.matmul(network_aug_features, network_aug_features.T)

            # remove positive pair
            mask = torch.eye(network_bs//2, dtype=torch.bool)
            large_mask = torch.kron(torch.ones((2,2)), mask).to(torch.bool)

            trunc_similarity_matrix = similarity_matrix[~large_mask].reshape(network_bs, -1) # dropping the positive pairs

            # create truncated index matrix, of pairs (idx, aug_idx) in shape (global_batch_size, truc_neg_batch_size, 2)
            network_index_matrix = torch.stack( (network_global_idx, network_aug_idx) ).T
            network_index_matrix = network_index_matrix.unsqueeze(0).expand(network_bs, -1, -1)
            network_index_matrix = network_index_matrix[~large_mask].reshape(network_bs, -1, 2) # dropping the positive pairs
            

        # Metropolis-Hasting over the in-batch negative samples
        # e.g.: for each positive sample, output [0,0,2,3,3,3,6,7] means that [0, 2, 3, 6, 7] are the accepted negatives (indicies are in-batch)
        with timer("sampling:samp", epoch=ep, rank=rank):
            with torch.no_grad():
                trunc_similarity_scores = torch.exp(beta * trunc_similarity_matrix).cpu()

                # get pairs of (batch_idx, batch_idx), i.e., in-batch negative pairs. indices are global batch index
                all_neg_pairs, chain = self.finite_state_mh_step( 
                                                            torch.cat((queries_local_idx, queries_local_idx)), 
                                                            network_aug_idx[network_local_row_idx],
                                                            trunc_similarity_scores[network_local_row_idx], 
                                                            network_local_row_idx,
                                                            burn_in)
            

            # append the init. chain state's pair at the last column
            init_states = self.states[ (torch.cat((queries_local_idx, queries_local_idx)), queries_aug_idx) ]
            init_states = init_states.unsqueeze(1)

            local_index_matrix = network_index_matrix[network_local_row_idx]
            network_index_matrix = torch.cat( (local_index_matrix, init_states), axis=1 )


            if len(noninit_mask) < network_bs:
                new_col = torch.zeros(network_bs, device=self.device)
                new_col[~noninit_mask] = init_similarity
                new_col = new_col.unsqueeze(1)
                trunc_similarity_matrix = torch.cat( (trunc_similarity_matrix, new_col), axis=1 )

            # update the local markov chain state
            latest_chain_state = network_index_matrix[ (torch.arange(len(network_local_row_idx)), chain) ] # chain == -1 means use the init chain state.
            self.states[ (torch.cat((queries_local_idx, queries_local_idx)), queries_aug_idx) ] = latest_chain_state # the last chain state is stored for the next time we access this chain
            # state_scores are not updated until the next revisit that we update with new scores on new model parameters.

        with timer("sampling:samp_comm", epoch=ep, rank=rank):
            # gather Metropolis-Hasting results
            all_neg_pairs = all_neg_pairs.to(comm_dev).contiguous()
            recv_accepted_idx = [torch.empty_like(all_neg_pairs) for _ in range(self.world_size)]
            dist.all_gather(recv_accepted_idx, all_neg_pairs)
            global_batch_negs = torch.cat(recv_accepted_idx)

        with timer("sampling:reorder", epoch=ep, rank=rank):
            # reconstruct the Metropolis-Hasting results by reindexing and form the backward-able mcmc_similarity_matrix
            mcmc_similarity_matrix = trunc_similarity_matrix[ (global_batch_negs[:,0], global_batch_negs[:,1]) ]

        return mcmc_similarity_matrix, None

# --- Timer summary ------------------------
#   Event   |  Count | Average time |  Frac.
# - communication.1                |  36363 |     0.01628s |   0.8%
# - communication.2                |  36363 |     0.14528s |   7.2%
# - communication.3                |  36363 |     0.24720s |  12.2%
# - communication.gradient         |  36363 |     0.30437s |  15.0%
# - computation.1a                 |  36363 |     0.04309s |   2.1%
# - computation.1b                 |  36363 |     0.00066s |   0.0%
# - computation.2                  |  36363 |     0.00652s |   0.3%
# - computation.3                  |  36363 |     0.00071s |   0.0%
# - computation.gradient           |  36363 |     0.19200s |   9.5%
# - dataloading                    |  36363 |     0.18850s |   9.3%
# - evaluation                     |     92 |   135.48945s |  17.1%
# - sampling+in-batch-neg          |  36363 |     0.53371s |  26.3%
# -------------------------------------------
# - total_averaged_time           |  36363 |   136.21165s |
# -------------------------------------------