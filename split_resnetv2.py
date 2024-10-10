from networks.resnetv2 import *
import torch
from torch import nn
import torch.nn.functional as F
__all__ = ["NetHead{}".format(i) for i in range(10)] + ["NetRemain{}".format(i) for i in range(10)] + ['NetHead9SpeC', 'NetRemain9SpeC']
net = KNOWN_MODELS['BiT-S-R101x1']()


test_x = torch.randn(1, 3, 224, 224)


net(test_x)

class NetHead0(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.root(x)
        x = self.net.body.block1(x)
        x = self.net.body.block2(x)
        x = self.net.body.block3(x)
        residual = x
        x = self.net.body.block4.unit01.gn1(x)
        return x, residual
        
class NetRemain0(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, residual):
        # x = self.net.body.block4.unit01.relu(x)
        x = F.relu(x)
        # print(hasattr(self.net.body.block4.unit01, 'downsample'))
        if hasattr(self.net.body.block4.unit01, 'downsample'):
            residual = self.net.body.block4.unit01.downsample(x)
        x = self.net.body.block4.unit01.conv1(x)
        x = self.net.body.block4.unit01.gn2(x)
        x = self.net.body.block4.unit01.relu(x)
        x = self.net.body.block4.unit01.conv2(x)
        x = self.net.body.block4.unit01.gn3(x)
        x = self.net.body.block4.unit01.relu(x)
        x = self.net.body.block4.unit01.conv3(x)
        x += residual
        x = self.net.body.block4.unit02(x)
        x = self.net.body.block4.unit03(x)
        bh = self.net.before_head(x)
        x = self.net.head(bh)
        assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        return x[..., 0, 0],bh
    

class NetHead1(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.root(x)
        x = self.net.body.block1(x)
        x = self.net.body.block2(x)
        x = self.net.body.block3(x)
        residual = x
        residual = self.net.body.block4.unit01.downsample(residual)
        x = self.net.body.block4.unit01.gn1(x)
        x = self.net.body.block4.unit01.relu(x)
        # print(hasattr(self.net.body.block4.unit01, 'downsample'))
        if hasattr(self.net.body.block4.unit01, 'downsample'):
            residual = self.net.body.block4.unit01.downsample(x)
        x = self.net.body.block4.unit01.conv1(x)
        x = self.net.body.block4.unit01.gn2(x)
        return x, residual
        
class NetRemain1(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, residual):
        x = F.relu(x)
        x = self.net.body.block4.unit01.conv2(x)
        x = self.net.body.block4.unit01.gn3(x)
        x = self.net.body.block4.unit01.relu(x)
        x = self.net.body.block4.unit01.conv3(x)
        x += residual
        x = self.net.body.block4.unit02(x)
        x = self.net.body.block4.unit03(x)
        bh = self.net.before_head(x)
        x = self.net.head(bh)
        assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        return x[..., 0, 0],bh
    

class NetHead2(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.root(x)
        x = self.net.body.block1(x)
        x = self.net.body.block2(x)
        x = self.net.body.block3(x)
        residual = x
        x = self.net.body.block4.unit01.gn1(x)
        x = self.net.body.block4.unit01.relu(x)
        # print(hasattr(self.net.body.block4.unit01, 'downsample'))
        if hasattr(self.net.body.block4.unit01, 'downsample'):
            residual = self.net.body.block4.unit01.downsample(x)
        x = self.net.body.block4.unit01.conv1(x)
        x = self.net.body.block4.unit01.gn2(x)
        x = self.net.body.block4.unit01.relu(x)
        x = self.net.body.block4.unit01.conv2(x)
        x = self.net.body.block4.unit01.gn3(x)
        return x, residual
        
class NetRemain2(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, residual):
        x = F.relu(x)
        x = self.net.body.block4.unit01.conv3(x)
        x += residual
        x = self.net.body.block4.unit02(x)
        x = self.net.body.block4.unit03(x)
        bh = self.net.before_head(x)
        x = self.net.head(bh)
        assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        return x[..., 0, 0],bh
    

class NetHead3(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.root(x)
        x = self.net.body.block1(x)
        x = self.net.body.block2(x)
        x = self.net.body.block3(x)
        x = self.net.body.block4.unit01(x)
        residual = x
        x = self.net.body.block4.unit02.gn1(x)
        return x, residual
        
class NetRemain3(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, residual):
        x = F.relu(x)
        if hasattr(self.net.body.block4.unit02, 'downsample'):
            residual = self.net.body.block4.unit02.downsample(x)
        x = self.net.body.block4.unit02.conv1(x)
        x = self.net.body.block4.unit02.gn2(x)
        x = self.net.body.block4.unit02.relu(x)
        x = self.net.body.block4.unit02.conv2(x)
        x = self.net.body.block4.unit02.gn3(x)
        x = self.net.body.block4.unit02.relu(x)
        x = self.net.body.block4.unit02.conv3(x)
        x += residual
        x = self.net.body.block4.unit03(x)
        bh = self.net.before_head(x)
        x = self.net.head(bh)
        assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        return x[..., 0, 0],bh
    

class NetHead4(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.root(x)
        x = self.net.body.block1(x)
        x = self.net.body.block2(x)
        x = self.net.body.block3(x)
        x = self.net.body.block4.unit01(x)
        residual = x
        x = self.net.body.block4.unit02.gn1(x)
        x = self.net.body.block4.unit02.relu(x)
        if hasattr(self.net.body.block4.unit02, 'downsample'):
            residual = self.net.body.block4.unit02.downsample(x)
        x = self.net.body.block4.unit02.conv1(x)
        x = self.net.body.block4.unit02.gn2(x)
        return x, residual
        
class NetRemain4(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, residual):
        x = F.relu(x)
        x = self.net.body.block4.unit02.conv2(x)
        x = self.net.body.block4.unit02.gn3(x)
        x = self.net.body.block4.unit02.relu(x)
        x = self.net.body.block4.unit02.conv3(x)
        x += residual
        x = self.net.body.block4.unit03(x)
        bh = self.net.before_head(x)
        x = self.net.head(bh)
        assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        return x[..., 0, 0],bh
    

class NetHead5(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.root(x)
        x = self.net.body.block1(x)
        x = self.net.body.block2(x)
        x = self.net.body.block3(x)
        x = self.net.body.block4.unit01(x)
        residual = x
        x = self.net.body.block4.unit02.gn1(x)
        x = self.net.body.block4.unit02.relu(x)
        if hasattr(self.net.body.block4.unit02, 'downsample'):
            residual = self.net.body.block4.unit02.downsample(x)
        x = self.net.body.block4.unit02.conv1(x)
        x = self.net.body.block4.unit02.gn2(x)
        x = self.net.body.block4.unit02.relu(x)
        x = self.net.body.block4.unit02.conv2(x)
        x = self.net.body.block4.unit02.gn3(x)

        return x, residual
        
class NetRemain5(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, residual):
        x = F.relu(x)
        x = self.net.body.block4.unit02.conv3(x)
        x += residual
        x = self.net.body.block4.unit03(x)
        bh = self.net.before_head(x)
        x = self.net.head(bh)
        assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        return x[..., 0, 0],bh
    

class NetHead6(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.root(x)
        x = self.net.body.block1(x)
        x = self.net.body.block2(x)
        x = self.net.body.block3(x)
        x = self.net.body.block4.unit01(x)
        x = self.net.body.block4.unit02(x)
        residual = x
        x = self.net.body.block4.unit03.gn1(x)
        return x, residual
        
class NetRemain6(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, residual):
        x = F.relu(x)
        if hasattr(self.net.body.block4.unit03, 'downsample'):
            residual = self.net.body.block4.unit03.downsample(x)
        x = self.net.body.block4.unit03.conv1(x)
        x = self.net.body.block4.unit03.gn2(x)
        x = self.net.body.block4.unit03.relu(x)
        x = self.net.body.block4.unit03.conv2(x)
        x = self.net.body.block4.unit03.gn3(x)
        x = self.net.body.block4.unit03.relu(x)
        x = self.net.body.block4.unit03.conv3(x)
        x += residual
        bh = self.net.before_head(x)
        x = self.net.head(bh)
        assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        return x[..., 0, 0],bh
    

class NetHead7(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.root(x)
        x = self.net.body.block1(x)
        x = self.net.body.block2(x)
        x = self.net.body.block3(x)
        x = self.net.body.block4.unit01(x)
        x = self.net.body.block4.unit02(x)
        residual = x
        x = self.net.body.block4.unit03.gn1(x)
        x = self.net.body.block4.unit03.relu(x)
        if hasattr(self.net.body.block4.unit03, 'downsample'):
            residual = self.net.body.block4.unit03.downsample(x)
        x = self.net.body.block4.unit03.conv1(x)
        x = self.net.body.block4.unit03.gn2(x)
        return x, residual
        
class NetRemain7(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self,x, residual):
        x = F.relu(x)
        x = self.net.body.block4.unit03.conv2(x)
        x = self.net.body.block4.unit03.gn3(x)
        x = self.net.body.block4.unit03.relu(x)
        x = self.net.body.block4.unit03.conv3(x)
        x += residual
        bh = self.net.before_head(x)
        x = self.net.head(bh)
        assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        return x[..., 0, 0],bh
    

class NetHead8(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.root(x)
        x = self.net.body.block1(x)
        x = self.net.body.block2(x)
        x = self.net.body.block3(x)
        x = self.net.body.block4.unit01(x)
        x = self.net.body.block4.unit02(x)
        residual = x
        x = self.net.body.block4.unit03.gn1(x)
        x = self.net.body.block4.unit03.relu(x)
        if hasattr(self.net.body.block4.unit03, 'downsample'):
            residual = self.net.body.block4.unit03.downsample(x)
        x = self.net.body.block4.unit03.conv1(x)
        x = self.net.body.block4.unit03.gn2(x)
        x = self.net.body.block4.unit03.relu(x)
        x = self.net.body.block4.unit03.conv2(x)
        x = self.net.body.block4.unit03.gn3(x)
        return x, residual
        
class NetRemain8(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self,x, residual):
        
        x = F.relu(x)
        x = self.net.body.block4.unit03.conv3(x)
        x += residual
        bh = self.net.before_head(x)
        x = self.net.head(bh)
        assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        return x[..., 0, 0],bh
    

class NetHead9(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.root(x)
        x = self.net.body.block1(x)
        x = self.net.body.block2(x)
        x = self.net.body.block3(x)
        x = self.net.body.block4(x)
        x = self.net.before_head(x)
        return x
        
class NetRemain9(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self,x):
        bh = x
        x = self.net.head(bh)
        assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        return x[..., 0, 0],bh
    

class NetHead9SpeC(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.root(x)
        x = self.net.body.block1(x)
        x = self.net.body.block2(x)
        x = self.net.body.block3(x)
        x = self.net.body.block4(x)
        x = self.net.before_head.gn(x)
        return x
        
class NetRemain9SpeC(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self,x):
        x = F.relu(x)
        bh = self.net.before_head.avg(x)
        x = self.net.head(bh)
        assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        return x[..., 0, 0],bh
    

# %%
