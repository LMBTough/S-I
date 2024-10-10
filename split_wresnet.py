
from networks.wresnet import KNOWN_MODELS
import torch.nn as nn
import torch
from torch import nn
import torch.nn.functional as F
import torch
import torch.nn as nn

# All hooks need data type
class Grad_all_hook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.save_grad)
        self.data = torch.Tensor()
        self.feature = torch.Tensor()

    def save_grad(self, module, input, output):
        def _stor_grad(grad):
            self.data = grad.detach()
        output.register_hook(_stor_grad)
        self.feature = output.detach().clone()

    def close(self):
        self.hook.remove()
        
class Grad_feature_hook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.save_grad)
        self.data = torch.Tensor()
        self.feature = torch.Tensor()

    def save_grad(self, module, input, output):
        def _stor_grad(grad):
            self.data = grad.detach()
        output.register_hook(_stor_grad)
        self.feature = output.clone()

    def close(self):
        self.hook.remove()

net = KNOWN_MODELS['wrn_40_2'](num_classes=100)
checkpoint = torch.load("checkpoint/models/cifar100_wrn40_2.pth", map_location='cpu')
net.load_state_dict(checkpoint['state_dict'])


test_x = torch.randn(1, 3, 32,32)

from copy import deepcopy

class NetHead0(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1.layer[0].bn1(x)
        return x
        
class NetRemain0(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = F.relu(x)
        out = self.net.block1.layer[0].conv1(x)
        out = self.net.block1.layer[0].bn2(out)
        out = F.relu(out)
        out = self.net.block1.layer[0].dropout(out)
        out = self.net.block1.layer[0].conv2(out)
        x = torch.add(self.net.block1.layer[0].convShortcut(x), out)
        x = self.net.block1.layer[1](x)
        x = self.net.block1.layer[2](x)
        x = self.net.block1.layer[3](x)
        x = self.net.block1.layer[4](x)
        x = self.net.block1.layer[5](x)
        x = self.net.block2(x)
        x = self.net.block3(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead1(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1.layer[0].bn1(x)
        x = F.relu(x)
        out = self.net.block1.layer[0].conv1(x)
        out = self.net.block1.layer[0].bn2(out)
        return out,x
        
class NetRemain1(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block1.layer[0].dropout(out)
        out = self.net.block1.layer[0].conv2(out)
        x = torch.add( self.net.block1.layer[0].convShortcut(x), out)
        x = self.net.block1.layer[1](x)
        x = self.net.block1.layer[2](x)
        x = self.net.block1.layer[3](x)
        x = self.net.block1.layer[4](x)
        x = self.net.block1.layer[5](x)
        x = self.net.block2(x)
        x = self.net.block3(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead2(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1.layer[0](x)
        out = self.net.block1.layer[1].bn1(x)
        return out,x
        
class NetRemain2(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block1.layer[1].conv1(out)
        out = self.net.block1.layer[1].bn2(out)
        out = F.relu(out)
        out = self.net.block1.layer[1].dropout(out)
        out = self.net.block1.layer[1].conv2(out)
        x = torch.add(x, out)
        x = self.net.block1.layer[2](x)
        x = self.net.block1.layer[3](x)
        x = self.net.block1.layer[4](x)
        x = self.net.block1.layer[5](x)
        x = self.net.block2(x)
        x = self.net.block3(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead3(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1.layer[0](x)
        out = self.net.block1.layer[1].bn1(x)
        out = F.relu(out)
        out = self.net.block1.layer[1].conv1(out)
        out = self.net.block1.layer[1].bn2(out)
        return out,x
        
class NetRemain3(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block1.layer[1].dropout(out)
        out = self.net.block1.layer[1].conv2(out)
        x = torch.add(x, out)
        x = self.net.block1.layer[2](x)
        x = self.net.block1.layer[3](x)
        x = self.net.block1.layer[4](x)
        x = self.net.block1.layer[5](x)
        x = self.net.block2(x)
        x = self.net.block3(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead4(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1.layer[0](x)
        x = self.net.block1.layer[1](x)
        out = self.net.block1.layer[2].bn1(x)
        return out,x
        
class NetRemain4(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block1.layer[2].conv1(out)
        out = self.net.block1.layer[2].bn2(out)
        out = F.relu(out)
        out = self.net.block1.layer[2].dropout(out)
        out = self.net.block1.layer[2].conv2(out)
        x = torch.add(x, out)
        x = self.net.block1.layer[3](x)
        x = self.net.block1.layer[4](x)
        x = self.net.block1.layer[5](x)
        x = self.net.block2(x)
        x = self.net.block3(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead5(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1.layer[0](x)
        x = self.net.block1.layer[1](x)
        out = self.net.block1.layer[2].bn1(x)
        out = F.relu(out)
        out = self.net.block1.layer[2].conv1(out)
        out = self.net.block1.layer[2].bn2(out)
        return out,x
        
class NetRemain5(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block1.layer[2].dropout(out)
        out = self.net.block1.layer[2].conv2(out)
        x = torch.add(x, out)
        x = self.net.block1.layer[3](x)
        x = self.net.block1.layer[4](x)
        x = self.net.block1.layer[5](x)
        x = self.net.block2(x)
        x = self.net.block3(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead6(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1.layer[0](x)
        x = self.net.block1.layer[1](x)
        x = self.net.block1.layer[2](x)
        out = self.net.block1.layer[3].bn1(x)
        return out,x
        
class NetRemain6(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block1.layer[3].conv1(out)
        out = self.net.block1.layer[3].bn2(out)
        out = F.relu(out)
        out = self.net.block1.layer[3].dropout(out)
        out = self.net.block1.layer[3].conv2(out)
        x = torch.add(x, out)
        x = self.net.block1.layer[4](x)
        x = self.net.block1.layer[5](x)
        x = self.net.block2(x)
        x = self.net.block3(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead7(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1.layer[0](x)
        x = self.net.block1.layer[1](x)
        x = self.net.block1.layer[2](x)
        out = self.net.block1.layer[3].bn1(x)
        out = F.relu(out)
        out = self.net.block1.layer[3].conv1(out)
        out = self.net.block1.layer[3].bn2(out)
        return out,x
        
class NetRemain7(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block1.layer[3].dropout(out)
        out = self.net.block1.layer[3].conv2(out)
        x = torch.add(x, out)
        x = self.net.block1.layer[4](x)
        x = self.net.block1.layer[5](x)
        x = self.net.block2(x)
        x = self.net.block3(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x

from copy import deepcopy

class NetHead8(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1.layer[0](x)
        x = self.net.block1.layer[1](x)
        x = self.net.block1.layer[2](x)
        x = self.net.block1.layer[3](x)
        out = self.net.block1.layer[4].bn1(x)
        return out,x
        
class NetRemain8(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block1.layer[4].conv1(out)
        out = self.net.block1.layer[4].bn2(out)
        out = F.relu(out)
        out = self.net.block1.layer[4].dropout(out)
        out = self.net.block1.layer[4].conv2(out)
        x = torch.add(x, out)
        x = self.net.block1.layer[5](x)
        x = self.net.block2(x)
        x = self.net.block3(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead9(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1.layer[0](x)
        x = self.net.block1.layer[1](x)
        x = self.net.block1.layer[2](x)
        x = self.net.block1.layer[3](x)
        out = self.net.block1.layer[4].bn1(x)
        out = F.relu(out)
        out = self.net.block1.layer[4].conv1(out)
        out = self.net.block1.layer[4].bn2(out)
        return out,x
        
class NetRemain9(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block1.layer[4].dropout(out)
        out = self.net.block1.layer[4].conv2(out)
        x = torch.add(x, out)
        x = self.net.block1.layer[5](x)
        x = self.net.block2(x)
        x = self.net.block3(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead10(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1.layer[0](x)
        x = self.net.block1.layer[1](x)
        x = self.net.block1.layer[2](x)
        x = self.net.block1.layer[3](x)
        x = self.net.block1.layer[4](x)
        out = self.net.block1.layer[5].bn1(x)
        return out,x
        
class NetRemain10(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block1.layer[5].conv1(out)
        out = self.net.block1.layer[5].bn2(out)
        out = F.relu(out)
        out = self.net.block1.layer[5].dropout(out)
        out = self.net.block1.layer[5].conv2(out)
        x = torch.add(x, out)
        x = self.net.block2(x)
        x = self.net.block3(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead11(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1.layer[0](x)
        x = self.net.block1.layer[1](x)
        x = self.net.block1.layer[2](x)
        x = self.net.block1.layer[3](x)
        x = self.net.block1.layer[4](x)
        out = self.net.block1.layer[5].bn1(x)
        out = F.relu(out)
        out = self.net.block1.layer[5].conv1(out)
        out = self.net.block1.layer[5].bn2(out)
        return out,x
        
class NetRemain11(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block1.layer[5].dropout(out)
        out = self.net.block1.layer[5].conv2(out)
        x = torch.add(x, out)
        x = self.net.block2(x)
        x = self.net.block3(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead12(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1(x)
        x = self.net.block2.layer[0].bn1(x)
        return x
        
class NetRemain12(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = F.relu(x)
        out = self.net.block2.layer[0].conv1(x)
        out = self.net.block2.layer[0].bn2(out)
        out = F.relu(out)
        out = self.net.block2.layer[0].dropout(out)
        out = self.net.block2.layer[0].conv2(out)
        x = torch.add(self.net.block2.layer[0].convShortcut(x), out)
        x = self.net.block2.layer[1](x)
        x = self.net.block2.layer[2](x)
        x = self.net.block2.layer[3](x)
        x = self.net.block2.layer[4](x)
        x = self.net.block2.layer[5](x)
        x = self.net.block3(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead13(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1(x)
        x = self.net.block2.layer[0].bn1(x)
        x = F.relu(x)
        out = self.net.block2.layer[0].conv1(x)
        out = self.net.block2.layer[0].bn2(out)
        return out,x
        
class NetRemain13(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block2.layer[0].dropout(out)
        out = self.net.block2.layer[0].conv2(out)
        x = torch.add(self.net.block2.layer[0].convShortcut(x), out)
        x = self.net.block2.layer[1](x)
        x = self.net.block2.layer[2](x)
        x = self.net.block2.layer[3](x)
        x = self.net.block2.layer[4](x)
        x = self.net.block2.layer[5](x)
        x = self.net.block3(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead14(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1(x)
        x = self.net.block2.layer[0](x)
        out = self.net.block2.layer[1].bn1(x)
        return out,x
        
class NetRemain14(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block2.layer[1].conv1(out)
        out = self.net.block2.layer[1].bn2(out)
        out = F.relu(out)
        out = self.net.block2.layer[1].dropout(out)
        out = self.net.block2.layer[1].conv2(out)
        x = torch.add(x, out)
        x = self.net.block2.layer[2](x)
        x = self.net.block2.layer[3](x)
        x = self.net.block2.layer[4](x)
        x = self.net.block2.layer[5](x)
        x = self.net.block3(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead15(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1(x)
        x = self.net.block2.layer[0](x)
        out = self.net.block2.layer[1].bn1(x)
        out = F.relu(out)
        out = self.net.block2.layer[1].conv1(out)
        out = self.net.block2.layer[1].bn2(out)
        return out,x
        
class NetRemain15(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        
        out = F.relu(out)
        out = self.net.block2.layer[1].dropout(out)
        out = self.net.block2.layer[1].conv2(out)
        x = torch.add(x, out)
        x = self.net.block2.layer[2](x)
        x = self.net.block2.layer[3](x)
        x = self.net.block2.layer[4](x)
        x = self.net.block2.layer[5](x)
        x = self.net.block3(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead16(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1(x)
        x = self.net.block2.layer[0](x)
        x = self.net.block2.layer[1](x)
        out = self.net.block2.layer[2].bn1(x)
        return out,x
        
class NetRemain16(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block2.layer[2].conv1(out)
        out = self.net.block2.layer[2].bn2(out)
        out = F.relu(out)
        out = self.net.block2.layer[2].dropout(out)
        out = self.net.block2.layer[2].conv2(out)
        x = torch.add(x, out)
        x = self.net.block2.layer[3](x)
        x = self.net.block2.layer[4](x)
        x = self.net.block2.layer[5](x)
        x = self.net.block3(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead17(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1(x)
        x = self.net.block2.layer[0](x)
        x = self.net.block2.layer[1](x)
        out = self.net.block2.layer[2].bn1(x)
        out = F.relu(out)
        out = self.net.block2.layer[2].conv1(out)
        out = self.net.block2.layer[2].bn2(out)
        return out,x
        
class NetRemain17(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block2.layer[2].dropout(out)
        out = self.net.block2.layer[2].conv2(out)
        x = torch.add(x, out)
        x = self.net.block2.layer[3](x)
        x = self.net.block2.layer[4](x)
        x = self.net.block2.layer[5](x)
        x = self.net.block3(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead18(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1(x)
        x = self.net.block2.layer[0](x)
        x = self.net.block2.layer[1](x)
        x = self.net.block2.layer[2](x)
        out = self.net.block2.layer[3].bn1(x)
        return out,x
        
class NetRemain18(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block2.layer[3].conv1(out)
        out = self.net.block2.layer[3].bn2(out)
        out = F.relu(out)
        out = self.net.block2.layer[3].dropout(out)
        out = self.net.block2.layer[3].conv2(out)
        x = torch.add(x, out)
        x = self.net.block2.layer[4](x)
        x = self.net.block2.layer[5](x)
        x = self.net.block3(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead19(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1(x)
        x = self.net.block2.layer[0](x)
        x = self.net.block2.layer[1](x)
        x = self.net.block2.layer[2](x)
        out = self.net.block2.layer[3].bn1(x)
        out = F.relu(out)
        out = self.net.block2.layer[3].conv1(out)
        out = self.net.block2.layer[3].bn2(out)
        return out,x
        
class NetRemain19(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block2.layer[3].dropout(out)
        out = self.net.block2.layer[3].conv2(out)
        x = torch.add(x, out)
        x = self.net.block2.layer[4](x)
        x = self.net.block2.layer[5](x)
        x = self.net.block3(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead20(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1(x)
        x = self.net.block2.layer[0](x)
        x = self.net.block2.layer[1](x)
        x = self.net.block2.layer[2](x)
        x = self.net.block2.layer[3](x)
        out = self.net.block2.layer[4].bn1(x)
        return out,x
        
class NetRemain20(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block2.layer[4].conv1(out)
        out = self.net.block2.layer[4].bn2(out)
        out = F.relu(out)
        out = self.net.block2.layer[4].dropout(out)
        out = self.net.block2.layer[4].conv2(out)
        x = torch.add(x, out)
        x = self.net.block2.layer[5](x)
        x = self.net.block3(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead21(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1(x)
        x = self.net.block2.layer[0](x)
        x = self.net.block2.layer[1](x)
        x = self.net.block2.layer[2](x)
        x = self.net.block2.layer[3](x)
        out = self.net.block2.layer[4].bn1(x)
        out = F.relu(out)
        out = self.net.block2.layer[4].conv1(out)
        out = self.net.block2.layer[4].bn2(out)
        return out,x
        
class NetRemain21(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):

        out = F.relu(out)
        out = self.net.block2.layer[4].dropout(out)
        out = self.net.block2.layer[4].conv2(out)
        x = torch.add(x, out)
        x = self.net.block2.layer[5](x)
        x = self.net.block3(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead22(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1(x)
        x = self.net.block2.layer[0](x)
        x = self.net.block2.layer[1](x)
        x = self.net.block2.layer[2](x)
        x = self.net.block2.layer[3](x)
        x = self.net.block2.layer[4](x)
        out = self.net.block2.layer[5].bn1(x)
        return out,x
        
class NetRemain22(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block2.layer[5].conv1(out)
        out = self.net.block2.layer[5].bn2(out)
        out = F.relu(out)
        out = self.net.block2.layer[5].dropout(out)
        out = self.net.block2.layer[5].conv2(out)
        x = torch.add(x, out)
        x = self.net.block3(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead23(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1(x)
        x = self.net.block2.layer[0](x)
        x = self.net.block2.layer[1](x)
        x = self.net.block2.layer[2](x)
        x = self.net.block2.layer[3](x)
        x = self.net.block2.layer[4](x)
        out = self.net.block2.layer[5].bn1(x)
        out = F.relu(out)
        out = self.net.block2.layer[5].conv1(out)
        out = self.net.block2.layer[5].bn2(out)
        return out,x
        
class NetRemain23(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block2.layer[5].dropout(out)
        out = self.net.block2.layer[5].conv2(out)
        x = torch.add(x, out)
        x = self.net.block3(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead24(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1(x)
        x = self.net.block2(x)
        x = self.net.block3.layer[0].bn1(x)
        return x
        
class NetRemain24(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = F.relu(x)
        out = self.net.block3.layer[0].conv1(x)
        out = self.net.block3.layer[0].bn2(out)
        out = F.relu(out)
        out = self.net.block3.layer[0].dropout(out)
        out = self.net.block3.layer[0].conv2(out)
        x = torch.add(self.net.block3.layer[0].convShortcut(x), out)
        x = self.net.block3.layer[1](x)
        x = self.net.block3.layer[2](x)
        x = self.net.block3.layer[3](x)
        x = self.net.block3.layer[4](x)
        x = self.net.block3.layer[5](x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead25(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1(x)
        x = self.net.block2(x)
        x = self.net.block3.layer[0].bn1(x)
        x = F.relu(x)
        out = self.net.block3.layer[0].conv1(x)
        out = self.net.block3.layer[0].bn2(out)
        return out,x
        
class NetRemain25(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block3.layer[0].dropout(out)
        out = self.net.block3.layer[0].conv2(out)
        x = torch.add(self.net.block3.layer[0].convShortcut(x), out)
        x = self.net.block3.layer[1](x)
        x = self.net.block3.layer[2](x)
        x = self.net.block3.layer[3](x)
        x = self.net.block3.layer[4](x)
        x = self.net.block3.layer[5](x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead26(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1(x)
        x = self.net.block2(x)
        x = self.net.block3.layer[0](x)
        out = self.net.block3.layer[1].bn1(x)
        return out,x
        
class NetRemain26(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block3.layer[1].conv1(out)
        out = self.net.block3.layer[1].bn2(out)
        out = F.relu(out)
        out = self.net.block3.layer[1].dropout(out)
        out = self.net.block3.layer[1].conv2(out)
        x = torch.add(x, out)
        x = self.net.block3.layer[2](x)
        x = self.net.block3.layer[3](x)
        x = self.net.block3.layer[4](x)
        x = self.net.block3.layer[5](x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead27(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1(x)
        x = self.net.block2(x)
        x = self.net.block3.layer[0](x)
        out = self.net.block3.layer[1].bn1(x)
        out = F.relu(out)
        out = self.net.block3.layer[1].conv1(out)
        out = self.net.block3.layer[1].bn2(out)
        return out,x
        
class NetRemain27(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block3.layer[1].dropout(out)
        out = self.net.block3.layer[1].conv2(out)
        x = torch.add(x, out)
        x = self.net.block3.layer[2](x)
        x = self.net.block3.layer[3](x)
        x = self.net.block3.layer[4](x)
        x = self.net.block3.layer[5](x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead28(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1(x)
        x = self.net.block2(x)
        x = self.net.block3.layer[0](x)
        x = self.net.block3.layer[1](x)
        out = self.net.block3.layer[2].bn1(x)
        return out,x
        
class NetRemain28(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block3.layer[2].conv1(out)
        out = self.net.block3.layer[2].bn2(out)
        out = F.relu(out)
        out = self.net.block3.layer[2].dropout(out)
        out = self.net.block3.layer[2].conv2(out)
        x = torch.add(x, out)
        x = self.net.block3.layer[3](x)
        x = self.net.block3.layer[4](x)
        x = self.net.block3.layer[5](x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead29(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1(x)
        x = self.net.block2(x)
        x = self.net.block3.layer[0](x)
        x = self.net.block3.layer[1](x)
        out = self.net.block3.layer[2].bn1(x)
        out = F.relu(out)
        out = self.net.block3.layer[2].conv1(out)
        out = self.net.block3.layer[2].bn2(out)
        return out,x
        
class NetRemain29(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block3.layer[2].dropout(out)
        out = self.net.block3.layer[2].conv2(out)
        x = torch.add(x, out)
        x = self.net.block3.layer[3](x)
        x = self.net.block3.layer[4](x)
        x = self.net.block3.layer[5](x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead30(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1(x)
        x = self.net.block2(x)
        x = self.net.block3.layer[0](x)
        x = self.net.block3.layer[1](x)
        x = self.net.block3.layer[2](x)
        out = self.net.block3.layer[3].bn1(x)
        return out,x
        
class NetRemain30(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block3.layer[3].conv1(out)
        out = self.net.block3.layer[3].bn2(out)
        out = F.relu(out)
        out = self.net.block3.layer[3].dropout(out)
        out = self.net.block3.layer[3].conv2(out)
        x = torch.add(x, out)
        x = self.net.block3.layer[4](x)
        x = self.net.block3.layer[5](x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead31(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1(x)
        x = self.net.block2(x)
        x = self.net.block3.layer[0](x)
        x = self.net.block3.layer[1](x)
        x = self.net.block3.layer[2](x)
        out = self.net.block3.layer[3].bn1(x)
        out = F.relu(out)
        out = self.net.block3.layer[3].conv1(out)
        out = self.net.block3.layer[3].bn2(out)
        return out,x
        
class NetRemain31(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block3.layer[3].dropout(out)
        out = self.net.block3.layer[3].conv2(out)
        x = torch.add(x, out)
        x = self.net.block3.layer[4](x)
        x = self.net.block3.layer[5](x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x

from copy import deepcopy

class NetHead32(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1(x)
        x = self.net.block2(x)
        x = self.net.block3.layer[0](x)
        x = self.net.block3.layer[1](x)
        x = self.net.block3.layer[2](x)
        x = self.net.block3.layer[3](x)
        out = self.net.block3.layer[4].bn1(x)
        return out,x
        
class NetRemain32(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block3.layer[4].conv1(out)
        out = self.net.block3.layer[4].bn2(out)
        out = F.relu(out)
        out = self.net.block3.layer[4].dropout(out)
        out = self.net.block3.layer[4].conv2(out)
        x = torch.add(x, out)
        x = self.net.block3.layer[5](x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead33(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1(x)
        x = self.net.block2(x)
        x = self.net.block3.layer[0](x)
        x = self.net.block3.layer[1](x)
        x = self.net.block3.layer[2](x)
        x = self.net.block3.layer[3](x)
        out = self.net.block3.layer[4].bn1(x)
        out = F.relu(out)
        out = self.net.block3.layer[4].conv1(out)
        out = self.net.block3.layer[4].bn2(out)
        return out,x
        
class NetRemain33(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block3.layer[4].dropout(out)
        out = self.net.block3.layer[4].conv2(out)
        x = torch.add(x, out)
        x = self.net.block3.layer[5](x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead34(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1(x)
        x = self.net.block2(x)
        x = self.net.block3.layer[0](x)
        x = self.net.block3.layer[1](x)
        x = self.net.block3.layer[2](x)
        x = self.net.block3.layer[3](x)
        x = self.net.block3.layer[4](x)
        out = self.net.block3.layer[5].bn1(x)
        return out,x
        
class NetRemain34(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block3.layer[5].conv1(out)
        out = self.net.block3.layer[5].bn2(out)
        out = F.relu(out)
        out = self.net.block3.layer[5].dropout(out)
        out = self.net.block3.layer[5].conv2(out)
        x = torch.add(x, out)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead35(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1(x)
        x = self.net.block2(x)
        x = self.net.block3.layer[0](x)
        x = self.net.block3.layer[1](x)
        x = self.net.block3.layer[2](x)
        x = self.net.block3.layer[3](x)
        x = self.net.block3.layer[4](x)
        out = self.net.block3.layer[5].bn1(x)
        out = F.relu(out)
        out = self.net.block3.layer[5].conv1(out)
        out = self.net.block3.layer[5].bn2(out)
        return out,x
        
class NetRemain35(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, out, x):
        out = F.relu(out)
        out = self.net.block3.layer[5].dropout(out)
        out = self.net.block3.layer[5].conv2(out)
        x = torch.add(x, out)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

from copy import deepcopy

class NetHead36(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1(x)
        x = self.net.block2(x)
        x = self.net.block3.layer[0](x)
        x = self.net.block3.layer[1](x)
        x = self.net.block3.layer[2](x)
        x = self.net.block3.layer[3](x)
        x = self.net.block3.layer[4](x)
        x = self.net.block3.layer[5](x)
        x = self.net.bn1(x)
        return x
        
class NetRemain36(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x
    

class NetHead37(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.block1(x)
        x = self.net.block2(x)
        x = self.net.block3.layer[0](x)
        x = self.net.block3.layer[1](x)
        x = self.net.block3.layer[2](x)
        x = self.net.block3.layer[3](x)
        x = self.net.block3.layer[4](x)
        x = self.net.block3.layer[5](x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.AdaptAvgPool(x)
        return x
        
class NetRemain37(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x