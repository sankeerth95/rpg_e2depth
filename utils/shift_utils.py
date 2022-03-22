import torch

class Expr:

    def __init__(self):
        self.iteration = 0
        self.inferlist = []

    def setup(self, params={}):
        return NotImplementedError

    def reset(self):
        self.inferlist = []
        self.iteration = 0
        
    def run_iterations(self, start_iter, n_iter):
        return NotImplementedError

    def infer(self):
        return NotImplementedError


# experiment specific utilities

# Similarity experiment: where we compare two tensors:
class SimilarityExpr(Expr):

    @staticmethod    
    def get_diff(t1, t2):
        diff = torch.count_nonzero(torch.abs(t1-t2)).cpu().numpy()
        return int(diff), torch.numel(t1)
    


class SweepDelta(SimilarityExpr):

    def __init__(self):
        super().__init__()
        

    @staticmethod
    def num_operations(tensorlist):
        base_tensor = tensorlist[0]

        ns, Ns = [], []
        for i in range(1, len(tensorlist)):
            tensor = tensorlist[i]
            n, N = SimilarityExpr.get_diff(tensor[0], base_tensor[0])
            ns.append(n)
            Ns.append(N)
            base_tensor = tensor

        return ns, Ns


class StoreIntermediateTensore:

    def __init__(self, modules):
        self.modules = modules

        self.store_tensors = {}
        for module in self.modules:
            self.store_tensors[module] = []

        self.hooks = {}
        print("\n******** Instantiated a StorInter class ******\n")

    def __del__(self):
        self.deregister_hooks()
        self.clear_tensors()

    def register_hooks(self):
        for module in self.modules:
            self.hooks[module] = module.register_forward_hook(self.load_tensors)
    
    def view_hooks(self):
        print(self.hooks)

    def deregister_hooks(self):
        for hook in self.hooks.values():
            hook.remove()
        self.hooks = {}

    def load_tensors(self, module, input, output):
        self.store_tensors[module].append(input)

    def clear_tensors(self):
        for module in self.modules:
            self.store_tensors[module] = []




def report_difference_histograms():
    return NotImplementedError



if __name__ == "__main__":
    import torch
    evflownet = torch.load('../pretrained_models/evflownet_model')
    modules = [
        evflownet.resnet_block[0].res_block[0],
        evflownet.resnet_block[0].res_block[1]
    ]
    su = StoreIntermediateTensore(modules)





