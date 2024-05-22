import torch
import torch.distributed as dist
from tqdm import tqdm
# references
# https://github.com/Spijkervet/SimCLR/blob/cd85c4366d2e6ac1b0a16798b76ac0a2c8a94e58/linear_evaluation.py
# https://github.com/Optimization-AI/SogCLR/blob/PyTorch/lincls.py

def linear_evaluation(data, test_data, num_classes, lr=0.001, batch_size=1024, epoch=100, weight_decay=0, device=torch.device("cpu"), distributed=False):
    embedding_dimension = data[0][0].shape[-1]

    linear_model = torch.nn.Linear(embedding_dimension, num_classes, bias=True).to(device)
    if distributed:
        if torch.cuda.is_available():
            linear_model = torch.nn.parallel.DistributedDataParallel(linear_model, device_ids=[device])
        else:
            linear_model = torch.nn.parallel.DistributedDataParallel(linear_model, device_ids=None)
        world_size = dist.get_world_size()
        batch_size = batch_size // world_size
    
    optimizer = torch.optim.SGD(linear_model.parameters(), lr, momentum=0.9, weight_decay=weight_decay) # from SogCLR

    if distributed:
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    num_samples = torch.tensor(len(data), dtype=torch.int64).to(device)
    if distributed:
        dist.all_reduce(num_samples, op=dist.ReduceOp.SUM)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
   
    if test_data is not None:
        num_test_samples = torch.tensor(len(test_data), dtype=torch.int64).to(device)
        if distributed:
            dist.all_reduce(num_test_samples, op=dist.ReduceOp.SUM)
        testdataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    linear_model.train()
    for ep in range(epoch):
        for image_embds, labels in dataloader:
            optimizer.zero_grad()
            image_embds, labels = image_embds.to(device), labels.to(device)
            output = linear_model(image_embds)
            loss = criterion(output, labels)
            
            if distributed:
                n_samples = torch.tensor(len(labels), device=device)
                dist.all_reduce(n_samples, op=dist.ReduceOp.SUM)
                loss *=  world_size / n_samples # when DDP average, loss is divided by world_size
            loss.backward()
            optimizer.step()

    # compute accuracy
    metrics = {}
    linear_model.eval()
    acc = 0
    for image_embds, labels in dataloader:
        image_embds, labels = image_embds.to(device), labels.to(device)
        pred = linear_model(image_embds)
        pred_class = torch.argmax(pred, dim=1)
        acc += sum(pred_class == labels)
    
    if distributed:
        acc = torch.tensor(acc, device=device)
        dist.all_reduce(acc, op=dist.ReduceOp.SUM)
    
    metrics["acc"] = ( acc / num_samples ).item()

    if test_data is not None:
        test_acc = 0
        for image_embds, labels in testdataloader:
            image_embds, labels = image_embds.to(device), labels.to(device)
            pred = linear_model(image_embds)
            pred_class = torch.argmax(pred, dim=1)
            test_acc += sum(pred_class == labels)
        
        if distributed:
            test_acc = torch.tensor(test_acc, device=device)
            dist.all_reduce(test_acc, op=dist.ReduceOp.SUM)

        metrics["test_acc"] = ( test_acc / num_test_samples ).item()

    return metrics

