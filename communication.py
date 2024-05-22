from enum import IntEnum
import torch
import torch.distributed as dist

class COMM(IntEnum):
    INIT_NEG_PAIR = 101 # receive (src_rank, a, local_idx, emb_type, emb_a) from this channel/tag, and respond to src_rank with emb_{local_idx}. locally compute emb_a @ jacobian( emb_{local_idx} )
    REPLY_NEG_PAIR = 102 # receive reply (src_rank, a, local_idx, emb_type, emb_{local_idx}) from this channel/tag. locally compute jacobian( emb_a ) @ emb_{local_idx}

class EMB_TYPE(IntEnum):
    IMG = 201
    TEXT = 202

class MSG_TYPE(IntEnum):
    IDX = 301
    REQ_IDX = 302
    TYPE_SIG = 303
    EMB = 304


def all_gather_tensors(tensor, ax, device):
    # gather along ax, i.e., each agent can have different shape on ax.
    world_size = dist.get_world_size()
    n_samples = torch.tensor(tensor.shape[ax], dtype=torch.int64, device=device)
    network_n_samples = [ torch.tensor(0, dtype=torch.int64, device=device) for _ in range(world_size) ]
    dist.all_gather(network_n_samples, n_samples)
    max_n_samples = torch.max(torch.stack(network_n_samples))

    padded_shape = torch.tensor(tensor.shape)
    padded_shape[ax] = max_n_samples
    padded_tensor = torch.zeros(padded_shape, dtype=tensor.dtype, device=device)
    if ax == 0:
        padded_tensor[0:n_samples] = tensor
    elif ax == 1:
        padded_tensor[:, 0:n_samples, :] = tensor
    elif ax == 2:
        padded_tensor[:, :, 0:n_samples] = tensor
    else:
        raise ValueError("ax = {} is not supported.".format(ax))


    network_tensors = [ torch.empty_like(padded_tensor, device=device) for _ in range(world_size) ]
    dist.all_gather(network_tensors, padded_tensor)
    if ax == 0:
        network_tensors = [mat[0:n] for mat, n in zip(network_tensors, network_n_samples)]
    elif ax == 1:
        network_tensors = [mat[:, 0:n, :] for mat, n in zip(network_tensors, network_n_samples)]
    elif ax == 2:
        network_tensors = [mat[:, :, 0:n] for mat, n in zip(network_tensors, network_n_samples)]

    return network_tensors


def all_gather_matrices(matrix, device):
    world_size = dist.get_world_size()
    n_samples = torch.tensor(len(matrix), dtype=torch.int64, device=device)
    network_n_samples = [ torch.tensor(0, dtype=torch.int64, device=device) for _ in range(world_size) ]
    dist.all_gather(network_n_samples, n_samples)
    max_n_samples = torch.max(torch.stack(network_n_samples))

    padded_matrix = torch.zeros((max_n_samples, matrix.shape[-1]), dtype=matrix.dtype, device=device)
    padded_matrix[:n_samples] = matrix

    network_matrices = [ torch.empty_like(padded_matrix, device=device) for _ in range(world_size) ]
    dist.all_gather(network_matrices, padded_matrix)
    network_matrices = [mat[:n] for mat, n in zip(network_matrices, network_n_samples)]

    return network_matrices


def all_gather_vectors(vector, device):
    world_size = dist.get_world_size()
    n_samples = torch.tensor(len(vector), dtype=torch.int64, device=device)
    network_n_samples = [ torch.tensor(0, dtype=torch.int64, device=device) for _ in range(world_size) ]
    dist.all_gather(network_n_samples, n_samples)
    max_n_samples = torch.max(torch.stack(network_n_samples))

    padded_vector = torch.zeros(max_n_samples, dtype=vector.dtype, device=device)
    padded_vector[:n_samples] = vector

    network_vectors = [ torch.empty_like(padded_vector, device=device) for _ in range(world_size) ]
    dist.all_gather(network_vectors, padded_vector)
    network_vectors = [vec[:n] for vec, n in zip(network_vectors, network_n_samples)]

    return network_vectors


def gather_matrices(matrix, dst, device):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    n_samples = torch.tensor(len(matrix), dtype=torch.int64, device=device)
    network_n_samples = [ torch.tensor(0, dtype=torch.int64, device=device) for _ in range(world_size) ]
    dist.all_gather(network_n_samples, n_samples)
    max_n_samples = torch.max(torch.stack(network_n_samples))

    padded_matrix = torch.zeros((max_n_samples, matrix.shape[-1]), dtype=matrix.dtype, device=device)
    padded_matrix[:n_samples] = matrix

    network_matrices = [ torch.empty_like(padded_matrix, device=device) for _ in range(world_size) ] if rank == dst else None
    dist.gather(padded_matrix, network_matrices)
    if rank == dst:
        network_matrices = [mat[:n] for mat, n in zip(network_matrices, network_n_samples)]

    return network_matrices


def gather_vectors(vector, dst, device):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    n_samples = torch.tensor(len(vector), dtype=torch.int64, device=device)
    network_n_samples = [ torch.tensor(0, dtype=torch.int64, device=device) for _ in range(world_size) ]
    dist.all_gather(network_n_samples, n_samples)
    max_n_samples = torch.max(torch.stack(network_n_samples))

    padded_vector = torch.zeros(max_n_samples, dtype=vector.dtype, device=device)
    padded_vector[:n_samples] = vector

    network_vectors = [ torch.empty_like(padded_vector, device=device) for _ in range(world_size) ] if rank == dst else None
    dist.gather(padded_vector, network_vectors)
    if rank == dst:
        network_vectors = [mat[:n] for mat, n in zip(network_vectors, network_n_samples)]

    return network_vectors


class all_gather_layer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        # allows input to have different size at axis=0.
        n_samples = torch.tensor(len(input), dtype=torch.int64, device=input.device)
        network_n_samples = [ torch.tensor(0, dtype=torch.int64, device=input.device) for _ in range(dist.get_world_size()) ]
        dist.all_gather(network_n_samples, n_samples)
        max_n_samples = torch.max(torch.stack(network_n_samples))

        padded_shape = torch.tensor(input.shape)
        padded_shape[0] = max_n_samples
        padded_shape = tuple(padded_shape.tolist())
        padded_matrix = torch.zeros(padded_shape, dtype=input.dtype, device=input.device)
        padded_matrix[:n_samples] = input

        ctx.save_for_backward(input)
        output = [torch.zeros_like(padded_matrix) for _ in range(dist.get_world_size())]
        dist.all_gather(output, padded_matrix)
        output = [mat[:n] for mat, n in zip(output, network_n_samples)]
        
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out
    
