import torch
import torch.multiprocessing as mp
import time
import os

def burn_gpu(rank, world_size, matrix_size=8192):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    print(f"[GPU {rank}] starting...")

    # 创建大矩阵（常驻显存）
    a = torch.randn(matrix_size, matrix_size, device=device)
    b = torch.randn(matrix_size, matrix_size, device=device)

    torch.cuda.synchronize()

    step = 0
    while True:
        # 核心计算：矩阵乘（非常吃算力）
        c = torch.matmul(a, b)

        # 防止被优化掉
        _ = c.sum().item()

        step += 1
        if step % 10 == 0:
            print(f"[GPU {rank}] step {step}")

def main():
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs")

    mp.spawn(
        burn_gpu,
        args=(num_gpus,),
        nprocs=num_gpus,
        join=True
    )

if __name__ == "__main__":
    main()