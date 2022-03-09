import torch
import torch.nn as nn


class Fence(nn.Module):
    def __init__(self, k=0.001):
        super().__init__()
        self.k = k
    
    def forward(self, x):
        if self.training:
            return x
        else:
            x = self.k*torch.floor(x/self.k)
        return x


def multiply_incr(W, T, incr, incr_n):

    # return W*T                    # original

    incr = incr + incr_n            # not sparse, but is intended to be sparse
    f_incr =  torch.floor(incr)     # not expensive, I think
    incr_out = W*f_incr             # sparse
    incr = incr - f_incr            # sparse, PRONE TO DRIFT!! : self trem in addition
    T = T + f_incr                  # sparse

    return T, incr, incr_out


if __name__  == '__main__':

    m = 5000 
    n = 5000
    p = 5000

    delta = torch.randn((m,n)).cuda()
    T = torch.randn((m,n)).cuda()
    W = torch.randn((p,m)).cuda()
    delta_n = torch.randn(((m,n))).cuda()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(100):
        T, delta, delta_n = multiply_incr(W, T, delta, delta_n)
    end.record()
    torch.cuda.synchronize()

    print(start.elapsed_time(end))


    start1 = torch.cuda.Event(enable_timing=True)
    end1 = torch.cuda.Event(enable_timing=True)


    start1.record()
    for _ in range(100):
        T = W*T
    end1.record()
    torch.cuda.synchronize()

    print(start1.elapsed_time(end1))




