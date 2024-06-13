import torch
import torch.distributed as dist
import os
import torch.multiprocessing as mp
import torch.nn as nn
# On each spawned worker
def worker(rank):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=2)
    torch.cuda.set_device(rank)
    model = nn.Linear(1, 1, bias=False).to(rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], output_device=rank
    )
    # Rank 1 gets one more input than rank 0.
    inputs = [torch.tensor([1]).float() for _ in range(10 + rank)]
    with model.join():
        for _ in range(5):
            for inp in inputs:
                loss = model(inp).sum()
                loss.backward()
    # Without the join() API, the below synchronization will hang
    # blocking for rank 1's allreduce to complete.
    torch.cuda.synchronize(device=rank)
def main():
    world_size = torch.cuda.device_count()
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    torch.multiprocessing.spawn(worker, args=(), nprocs=world_size, join=True)

if __name__ == "__main__":
    print("t")
    main()