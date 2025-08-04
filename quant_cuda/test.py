import torch
import KVmix 

KVmix.init_cublas()

B, nh, M, K, N = 2, 4, 32, 64, 128
bits = 4
group_size = 32
feat_per_int = 32 // bits
fA = torch.randn(B, nh, M, K, dtype=torch.float16, device='cuda')
qB = torch.randint(0, 16, (B, nh, K, N // feat_per_int), dtype=torch.int32, device='cuda')
scales = torch.randn(B, nh, K, N // group_size, dtype=torch.float16, device='cuda')
zeros = torch.randn(B, nh, K, N // group_size, dtype=torch.float16, device='cuda')

C =  KVmix.gemv_forward_cuda_outer_dim(group_size, fA, qB, scales, zeros, bits)
D =  KVmix.gemv_forward_cuda_outer_dimyb(group_size, fA, qB, scales, zeros, bits)
print("C=",C)  
print("D=",D)

KVmix.destroy_cublas()
