import os
from dataclasses import dataclass
import torch
# simple launch
# python train.py
# DDP launch for e.g 8GPU
# torchrun --standalone --nproc_per_node=8 train.py
# run the training loop
from torch.distributed import init_process_group, destroy_process_group


@dataclass
class DistributedContext:
    """Everything the training script needs to know about which process/device it is running on."""
    ddp: bool
    ddp_rank: int
    ddp_local_rank: int
    ddp_world_size: int
    device: str
    device_type: str
    master_process: bool


def setup_distributed():
    # set up DDP (distributed data parallel).
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # attempt to autodetect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")

    # autocast wants "cuda", not "cuda:0" (which is what device is under torchrun)
    device_type = "cuda" if device.startswith("cuda") else device

    return DistributedContext(
        ddp=ddp,
        ddp_rank=ddp_rank,
        ddp_local_rank=ddp_local_rank,
        ddp_world_size=ddp_world_size,
        device=device,
        device_type=device_type,
        master_process=master_process,
    )


def cleanup_distributed(ddp):
    if ddp:
        destroy_process_group()
