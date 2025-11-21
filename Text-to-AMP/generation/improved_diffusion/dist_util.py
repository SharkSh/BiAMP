import io
import os
import torch as th
import torch.distributed as dist

def setup_dist():
    if dist.is_initialized():
        return
    print("PyTorch version:", th.__version__)
    print("CUDA available:", th.cuda.is_available())
    print("CUDA version:", th.version.cuda)
    print("CUDA device count:", th.cuda.device_count())

    if th.cuda.is_available():
        print("CUDA device name:", th.cuda.get_device_name(0))
    else:
        print("No GPU detected. Check your environment and hardware.")

    backend = "nccl" if th.cuda.is_available() else "gloo"

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500") 

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

def dev():
    if th.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])  
        return th.device(f"cuda:{local_rank}")
    return th.device("cpu")

""" Load a PyTorch file without redundant fetches across ranks. Now using torch.distributed instead of MPI. """
def load_state_dict(path, **kwargs):
    rank = dist.get_rank()
    obj_list = [None]

    if rank == 0:
        with open(path, "rb") as f:
            obj_list[0] = f.read()

    # Broadcast the state_dict from rank 0 to all other ranks
    dist.broadcast_object_list(obj_list, src=0)

    return th.load(io.BytesIO(obj_list[0]), **kwargs)

def sync_params(params):
    for p in params:
        with th.no_grad():
            dist.broadcast(p, src=0)

def check():
    rank = int(os.environ["RANK"])
    if dist.is_initialized():
        print(f"✅ Distributed initialized: Rank {dist.get_rank()} / {dist.get_world_size()}")
    else:
        print("❌ Distributed initialization failed!")

    device = dev()
    print(f"🖥️ Rank {rank} initialized with device: {device}")

    dist.barrier()
    print(f"✅ Rank {dist.get_rank()} passed the barrier!")

    tensor = th.tensor([0.0], device=device)
    if rank == 0:
        tensor += 10.0
    dist.broadcast(tensor, src=0)
    print(f"📡 Rank {dist.get_rank()} received tensor: {tensor.item()}")

