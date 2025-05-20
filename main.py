import torch
import torch.nn as nn

# on 5/17/2025, this one works, using pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
# see https://docs.pytorch.org/get-started/locally/

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
t1 = torch.randn(1, 2)
t2 = torch.randn(1, 2).to(dev)
print(t1)  # tensor([[-0.2678,  1.9252]])
print(t2)  # tensor([[ 0.5117, -3.6247]], device='cuda:0')
t1.to(dev)
print(t1)  # tensor([[-0.2678,  1.9252]])
print(t1.is_cuda)  # False
t1 = t1.to(dev)
print(t1)  # tensor([[-0.2678,  1.9252]], device='cuda:0')
print(t1.is_cuda)  # True


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1, 2)

    def forward(self, x):
        x = self.l1(x)
        return x


model = M()  # not on cuda
model.to(dev)  # is on cuda (all parameters)
print(next(model.parameters()).is_cuda)  # True