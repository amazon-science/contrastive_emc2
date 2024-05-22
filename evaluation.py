import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from utils import model_inference, aug_image_inference, sync_gradient, save_current_gradient_to_host_memory, restore_gradient_to_model
from linear_eval import linear_evaluation
from communication import gather_matrices, gather_vectors, all_gather_layer
import time
from math import ceil

def remove_diagonal_blocks(matrix, k):
    # remove k x k diagonal blocks, i.e., from matrix \in [n, n] to [n, n-k].
    assert matrix.shape[0] == matrix.shape[1] and matrix.shape[1] % k == 0, "matrix shape {} cannot remove diagonal blocks".format(matrix.shape)

    result = []
    iterations = matrix.shape[0] // k
    for i in range(iterations):
        front_block = matrix[i*k : (i+1)*k, 0 : i*k]
        end_block =  matrix[i*k : (i+1)*k, (i+1)*k :]
        result.append(torch.cat([front_block, end_block], dim=1)) # list of [k, n-k]
    
    return torch.cat(result, dim=0)

def get_diagonal_blocks(matrix, k):
    # get k x k diagonal blocks, i.e., from matrix \in [n, n] to [n, n-k].
    assert matrix.shape[0] == matrix.shape[1] and matrix.shape[1] % k == 0, "matrix shape {} cannot get diagonal blocks".format(matrix.shape)
    result = []
    iterations = matrix.shape[0] // k
    for i in range(iterations):
        diag_block = matrix[i*k : (i+1)*k, i*k : (i+1)*k]
        result.append(diag_block) # list of [k, n-k]
    return result
    
        
def compute_global_loss(model, data, dev, config):
    assert config["finite_aug"], "exact gradient is only computable in finite augmentation setup."
    training_aug_batch_size = data.aug_batch_size
    data.aug_batch_size = data.n_augmentations
    dataloader = DataLoader(data, batch_size=config["pos_batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=config["pin_memory"])
    # compute embeddings of all images
    local_image_embds = []
    for _, aug_images, _, _, _ in dataloader:
        aug_image_features = aug_image_inference(model, aug_images, dev) # [aug_batch_size, pos_batch_size, embd_size]
        local_image_embds.append(  torch.permute(aug_image_features, (1,0,2)) ) # list of [pos_batch_size, aug_batch_size, embd_size]


    local_image_embds = torch.cat(local_image_embds, dim=0) # [n_samples, aug_batch_size, embd_size]    
    network_image_embds = torch.cat(all_gather_layer.apply(local_image_embds), dim=0) # [global_n_samples, aug_batch_size, embd_size]

    network_image_embds *= config["beta"]

    n_samples, aug_size, embd_size = network_image_embds.shape
    network_image_embds = network_image_embds.view(1, n_samples*aug_size, embd_size).squeeze(0) # [global_n_samples*aug_batch_size, embd_size]

    similarity_matrix = network_image_embds @ network_image_embds.T # [n, n], n = n_samples*aug_size
    negative_scores = remove_diagonal_blocks(similarity_matrix, aug_size) # [n, n-aug_size]
    positive_scores = get_diagonal_blocks(similarity_matrix, aug_size) # list of [aug_size, aug_size]

    mask = torch.eye(aug_size, dtype=torch.bool)
    positive_scores = torch.cat([m[~mask].flatten() for m in positive_scores]).unsqueeze(1) # [n_pos_scores, 1]
    neg_size = negative_scores.shape
    negative_scores = negative_scores.unsqueeze(0) # [1, n, n]
    negative_scores_expand = negative_scores.expand(aug_size-1, *neg_size)
    negative_scores_expand = negative_scores_expand.reshape(n_samples*aug_size*(aug_size-1), neg_size[-1]) # [n_pos_scores, n_neg_scores]

    ce_matrix = torch.cat([positive_scores, negative_scores_expand], dim=1)
    labels = torch.zeros(n_samples*aug_size*(aug_size-1), device=dev, dtype=torch.long)

    CE = torch.nn.CrossEntropyLoss().to(dev)

    loss = CE(ce_matrix, labels)

    data.aug_batch_size = training_aug_batch_size
    return loss

def compute_gradient_error(optimizer, world_size, model, data, dev, config):
    sync_gradient(optimizer, world_size)
    grad = save_current_gradient_to_host_memory(optimizer, world_size)

    optimizer.zero_grad()
    gloss = compute_global_loss(model, data, dev, config)
    gloss.backward()
    sync_gradient(optimizer, world_size)
    true_grad = save_current_gradient_to_host_memory(optimizer, world_size)

    grad_err = sum([torch.norm(g - tg)**2 for g, tg in zip(grad, true_grad)])
    optimizer.zero_grad()
    restore_gradient_to_model(optimizer, grad)
    return grad_err, gloss


def compute_gradient_norm(optimizer, world_size, model, data, dev, config):
    optimizer.zero_grad()
    gloss = compute_global_loss(model, data, dev, config)
    gloss.backward()
    sync_gradient(optimizer, world_size)
    true_grad = save_current_gradient_to_host_memory(optimizer, world_size)

    grad_norm = sum([torch.norm(tg)**2 for tg in true_grad])
    optimizer.zero_grad()
    return grad_norm, gloss


def global_contrastive_loss_simclr(rank, world_size, global_num_samples, model, dataloader, dev, config):
    # simclr means negative pairs are pairs of augmented images
    # we implement evaluation by sending embeddings
    local_pos_scores = 0

    # compute embeddings of all images
    local_image_embds = []
    for _, aug_images, _, _, _ in dataloader:
        aug_image_features = aug_image_inference(model, aug_images, dev) # [aug_batch_size, pos_batch_size, embd_size]
        local_pos_scores += -config["beta"] * torch.sum(aug_image_features[0] * aug_image_features[1]) # [pos_batch_size]
        local_image_embds.append(aug_image_features[0])
        local_image_embds.append(aug_image_features[1])
    
    # gather pos loss at rank 0
    # local_pos_scores = torch.tensor(local_pos_scores, device=dev)
    dist.reduce(local_pos_scores, 0, op=dist.ReduceOp.SUM)
    if rank == 0:
        global_pos_loss = local_pos_scores / global_num_samples

    local_image_embds = torch.cat(local_image_embds, axis=0)

    for ag in range(world_size):
        # exchange aug_image embeddings
        if ag == rank:
            n_samples = torch.tensor(len(local_image_embds), dtype=torch.int64, device=dev)
        else:
            n_samples = torch.tensor(0, dtype=torch.int64, device=dev)
        dist.broadcast(n_samples, ag)
        if ag == rank:
            target_image_features = local_image_embds
        else:
            target_image_features = torch.empty((n_samples, config["dim"]), dtype=local_image_embds.dtype, device=dev)
        dist.broadcast(target_image_features, ag)

        # compute the negative pair log-sum-exp
        shift = config["beta"] # https://lips.cs.princeton.edu/computing-log-sum-exp/
        neg_sums = torch.zeros(len(target_image_features), device=dev)
        bs = 256
        for i in range( ceil(len(local_image_embds) / 256) ):
            neg_exps = torch.exp( config["beta"] * target_image_features @ local_image_embds[i*bs: (i+1)*bs].T - shift)
            neg_sums += torch.sum(neg_exps, axis=1) 
        dist.reduce(neg_sums, ag, op=dist.ReduceOp.SUM) # global sum
        if ag == rank:
            local_neg_loss = sum( shift + torch.log(neg_sums) )
    
    # gather neg loss at rank 0
    dist.reduce(local_neg_loss, 0, op=dist.ReduceOp.SUM)
    if rank == 0:
        global_neg_loss = local_neg_loss / global_num_samples
    
    if rank == 0:
        return (global_pos_loss + global_neg_loss).item()
    else:
        return None


def global_contrastive_loss(rank, world_size, global_num_samples, model, dataloader, rawdataloader, dev, config):
    # we implement evaluation by sending embeddings
    local_loss = 0

    # compute embeddings of all images
    local_image_embds = []
    for image, _, _, _, _ in rawdataloader:
        _, image_features = model_inference(model, image, dev)
        local_image_embds.append(image_features)
    local_image_embds = torch.cat(local_image_embds, axis=0)

    # sync. number of batches to run
    n_batches = torch.tensor(len(dataloader), dtype=torch.int64, device=dev)
    dist.all_reduce(n_batches, op=dist.ReduceOp.MAX)

    eval_iter = iter(dataloader)
    for _ in range(n_batches.item()):
        try:
            image, aug_images, _, idx, _ = next(eval_iter)
        except StopIteration:
            # this node ran out of samples to evaluate but other nodes still have unevaluated samples
            idx = []
        
        if len(idx) > 0:
            aug_image_features = aug_image_inference(model, aug_images, dev) # [aug_batch_size, pos_batch_size, embd_size]
            _, image_features = model_inference(model, image, dev)
            repeat_image_features = image_features.unsqueeze(0).expand(aug_image_features.shape[0], -1, -1) # [aug_batch_size, pos_batch_size, embd_size]
            pos_scores = -config["beta"] * torch.sum(repeat_image_features * aug_image_features, axis=-1) # [aug_batch, pos_batch_size]
            pos_scores = torch.mean(pos_scores, axis=0) # [pos_batch_size]
        else:
            image_features = torch.empty((0, config["dim"]), device=dev)
            pos_scores = None

        # exchange aug_image embeddings
        aug_image_features = aug_image_features.view(-1, config["dim"])
        n_aug_samples = torch.tensor(len(aug_image_features), dtype=torch.int64, device=dev)
        round_samples = [ torch.tensor(0, dtype=torch.int64, device=dev) for _ in range(world_size) ]
        dist.all_gather(round_samples, n_aug_samples)
        target_image_features = [ torch.empty((m, config["dim"]), dtype=aug_image_features.dtype, device=dev) for m in round_samples ]
        dist.all_gather(target_image_features, aug_image_features)
        target_image_features = torch.cat(target_image_features) # [sum(round_samples), embd_size]

        shift = config["beta"] # https://lips.cs.princeton.edu/computing-log-sum-exp/
        # neg_exps = torch.exp( config["beta"] * target_image_features @ local_image_embds.T - shift).to(torch.float64) # [sum(round_samples), local_num_samples]
        neg_exps = torch.exp( config["beta"] * target_image_features @ local_image_embds.T - shift) # [sum(round_samples), local_num_samples]
        neg_sums = torch.sum(neg_exps, axis=1) # [sum(round_samples)], local sum
        dist.all_reduce(neg_sums, op=dist.ReduceOp.SUM) # global sum

        if len(idx) > 0:
            idx_splits = [0] + torch.cumsum(torch.stack(round_samples), 0).tolist()
            local_neg_sums = neg_sums[idx_splits[rank]: idx_splits[rank+1]] # pick neg_sums of the corresponding local aug positive samples, [pos_batch_size * aug_batch]
            local_loss += sum( pos_scores ) + sum( shift + torch.log(local_neg_sums) ) / config["transform_batch_size"] # sum over postives
        
    dist.reduce(local_loss, 0, op=dist.ReduceOp.SUM)
    if rank == 0:
        local_loss /= global_num_samples
        local_loss = local_loss.item()
        return local_loss
    else:
        return None


def get_frozen_data(le_rawdataloader, model, device):
    frozen_data = []
    for image, _, label, _, _ in le_rawdataloader:
        img_embds, _ = model_inference(model, image, device)
        img_embds = img_embds.clone().detach().cpu()
        for emb, lb in zip(img_embds, label):
            frozen_data.append((emb, lb))
    return frozen_data


def gather_frozen_data(frozen_data, dev):
    local_images_embds = torch.stack([d[0] for d in frozen_data]).to(dev)
    local_images_labels = torch.stack([d[1] for d in frozen_data]).to(dev)
    
    frozen_data_buffer = gather_matrices(local_images_embds, 0, dev)
    frozen_label_buffer = gather_vectors(local_images_labels, 0, dev)
    return frozen_data_buffer, frozen_label_buffer


def knn_acc(train_data, test_data, n_class):
    """ reference to https://github.com/Annusha/temperature_schedules/blob/main/notebooks/simclr_tau_cifar10.ipynb """
    train_features = torch.stack([x[0] for x in train_data])
    test_features = torch.stack([x[0] for x in test_data])

    train_labels = torch.tensor([x[1] for x in train_data])
    test_labels = torch.tensor([x[1] for x in test_data])

    pairwise_distance = torch.cdist(test_features, train_features)
    pred_rank = torch.argsort(pairwise_distance, dim=1)

    acc_1nn_sum = 0
    for cl in range(n_class):
        pred_1nn = pred_rank[test_labels == cl, 0] # prediction of data with label cl
        acc_1nn_sum += (train_labels[pred_1nn] == cl).sum()
    acc_1nn = acc_1nn_sum / len(test_labels)

    knn = 10
    acc_knn_sum = 0
    for cl in range(n_class):
        pred_knn = pred_rank[test_labels == cl, :knn] # 10-nn prediction of data with label cl
        pred_freq = train_labels[pred_knn]
        pred_cls, _ = torch.mode(pred_freq) # find the most frequent class among the knn 
        acc_knn_sum += (pred_cls == cl).sum()
    acc_knn = acc_knn_sum / len(test_labels)

    return acc_1nn.item(), acc_knn.item()



def evaluate(it, ep, rank, world_size, global_num_samples, model, data, dataloader, rawdataloader, le_rawdataloader, le_rawtestdataloader, knn_dataloader, dev, config, extra_stats={}):
    eval_start_time = time.time()
    model.eval()

    with torch.no_grad():
        # we implement evaluation by sending embeddings
        # simclr_loss = global_contrastive_loss_simclr(rank, world_size, global_num_samples, model, dataloader, dev, config) if config["transform_batch_size"] == 2 else None
        # loss = global_contrastive_loss(rank, world_size, global_num_samples, model, dataloader, rawdataloader, dev, config)
        simclr_loss = None
        loss = None

    # linear evaluation
    with torch.no_grad():
        frozen_data = get_frozen_data(le_rawdataloader, model, dev)
        frozen_test_data = get_frozen_data(le_rawtestdataloader, model, dev) if le_rawtestdataloader is not None else None
        frozen_knn_data = get_frozen_data(knn_dataloader, model, dev) if knn_dataloader is not None else None

    frozen_data_buffer, frozen_label_buffer = gather_frozen_data(frozen_data, dev)
    if le_rawtestdataloader is not None:
        frozen_test_data_buffer, frozen_test_label_buffer = gather_frozen_data(frozen_test_data, dev)
    if knn_dataloader is not None:
        frozen_knn_data_buffer, frozen_knn_label_buffer = gather_frozen_data(frozen_knn_data, dev)
    if rank == 0:
        frozen_data_buffer = [b.cpu() for b in frozen_data_buffer]
        frozen_label_buffer = [b.cpu() for b in frozen_label_buffer]
        frozen_data_buffer, frozen_label_buffer = torch.cat(frozen_data_buffer), torch.cat(frozen_label_buffer)
        frozen_data = [(d, l) for d, l in zip(frozen_data_buffer, frozen_label_buffer)]
        
        if le_rawtestdataloader is not None:
            frozen_test_data_buffer = [b.cpu() for b in frozen_test_data_buffer]
            frozen_test_label_buffer = [b.cpu() for b in frozen_test_label_buffer]
            frozen_test_data_buffer, frozen_test_label_buffer = torch.cat(frozen_test_data_buffer), torch.cat(frozen_test_label_buffer)
            frozen_test_data = [(d, l) for d, l in zip(frozen_test_data_buffer, frozen_test_label_buffer)]
        
        if knn_dataloader is not None:
            frozen_knn_data_buffer = [b.cpu() for b in frozen_knn_data_buffer]
            frozen_knn_label_buffer = [b.cpu() for b in frozen_knn_label_buffer]
            frozen_knn_data_buffer, frozen_knn_label_buffer = torch.cat(frozen_knn_data_buffer), torch.cat(frozen_knn_label_buffer)
            knn_frozen_data = [(d, l) for d, l in zip(frozen_knn_data_buffer, frozen_knn_label_buffer)]
        else:
            knn_frozen_data = frozen_data

        print("entering linear_evaluation")
        metrics = linear_evaluation(frozen_data, frozen_test_data, data.num_classes, weight_decay=config["linear_eval_weight_decay"], device=dev, distributed=False)
        acc_1nn, acc_10nn = knn_acc(knn_frozen_data, frozen_test_data, data.num_classes)
    config["accum_eval_time"] += time.time() - eval_start_time
    print("evaluation done")
    if rank == 0:
        import wandb
        wandb.log({"iteration": it, "epoch": ep, "accum_training_time": time.time() - config["program_start_time"] - config["accum_eval_time"], "loss": loss, "simclr_loss": simclr_loss, "acc_1nn": acc_1nn, "acc_10nn": acc_10nn, **metrics, **extra_stats})
        print("loss = {}; acc_1nn: {}; acc_10nn: {}; {}".format(simclr_loss, acc_1nn, acc_10nn, metrics))