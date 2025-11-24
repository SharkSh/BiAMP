import io
import os
import torch as th
import torch.distributed as dist

""" 设置 PyTorch 分布式进程组（不依赖 MPI）。"""
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

    # 从环境变量获取 `RANK` 和 `WORLD_SIZE`
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 设定主进程的地址和端口
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500") 

    # 初始化 PyTorch 分布式进程组
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

""" 获取当前 GPU 设备。"""
def dev():
    if th.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])  # `torchrun` 传递的 LOCAL_RANK
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

""" 确保每张卡上模型参数一致. """
def sync_params(params):
    for p in params:
        with th.no_grad():
            dist.broadcast(p, src=0)

""" 检查分布式训练有没有正常启动 """
def check():
    rank = int(os.environ["RANK"])
    # **检查 1: 进程是否正确启动**
    if dist.is_initialized():
        print(f"✅ Distributed initialized: Rank {dist.get_rank()} / {dist.get_world_size()}")
    else:
        print("❌ Distributed initialization failed!")

    # **检查 2: 每个 Rank 打印设备信息**
    device = dev()
    print(f"🖥️ Rank {rank} initialized with device: {device}")

    # **检查 3: 所有 rank 进行 barrier 同步**
    dist.barrier()
    print(f"✅ Rank {dist.get_rank()} passed the barrier!")

    # **检查 4: 测试 tensor 广播**
    tensor = th.tensor([0.0], device=device)
    if rank == 0:
        tensor += 10.0
    dist.broadcast(tensor, src=0)
    print(f"📡 Rank {dist.get_rank()} received tensor: {tensor.item()}")

