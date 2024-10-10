from networks.resnet import resnet34
import torch
from torch import nn
import torch.nn.functional as F

__all__ = ["NetHead{}".format(i) for i in range(36)] + ["NetRemain{}".format(i) for i in range(36)]

test_x = torch.randn(1, 3, 32,32)


net = resnet34()

net(test_x)

from copy import deepcopy

class NetHead0(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        
        return x
        
class NetRemain0(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead1(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        last_layer_x = F.relu(x)
        x = self.net.layer1[0].conv1(last_layer_x)
        x = self.net.layer1[0].bn1(x)
        return x,last_layer_x
    
class NetRemain1(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        x = F.relu(x)
        x = self.net.layer1[0].conv2(x)
        x = self.net.layer1[0].bn2(x)
        x = self.net.layer1[0].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer1[1](x)
        x = self.net.layer1[2](x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead2(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        last_layer_x = F.relu(x)
        x = self.net.layer1[0].conv1(last_layer_x)
        x = self.net.layer1[0].bn1(x)
        x = F.relu(x)
        x = self.net.layer1[0].conv2(x)
        x = self.net.layer1[0].bn2(x)
        return x,last_layer_x
    
class NetRemain2(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        x = self.net.layer1[0].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer1[1](x)
        x = self.net.layer1[2](x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead3(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        last_layer_x = self.net.layer1[0](x)
        x = self.net.layer1[1].conv1(last_layer_x)
        x = self.net.layer1[1].bn1(x)
        return x,last_layer_x
    
class NetRemain3(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        
        x = F.relu(x)
        x = self.net.layer1[1].conv2(x)
        x = self.net.layer1[1].bn2(x)
        x = self.net.layer1[1].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer1[2](x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead4(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        last_layer_x = self.net.layer1[0](x)
        x = self.net.layer1[1].conv1(last_layer_x)
        x = self.net.layer1[1].bn1(x)
        x = F.relu(x)
        x = self.net.layer1[1].conv2(x)
        x = self.net.layer1[1].bn2(x)
        return x,last_layer_x
    
class NetRemain4(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        
        x = self.net.layer1[1].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer1[2](x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead5(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1[0](x)
        last_layer_x = self.net.layer1[1](x)
        x = self.net.layer1[2].conv1(last_layer_x)
        x = self.net.layer1[2].bn1(x)
        return x,last_layer_x
    
class NetRemain5(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        
        x = F.relu(x)
        x = self.net.layer1[2].conv2(x)
        x = self.net.layer1[2].bn2(x)
        x = self.net.layer1[2].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead6(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1[0](x)
        last_layer_x = self.net.layer1[1](x)
        x = self.net.layer1[2].conv1(last_layer_x)
        x = self.net.layer1[2].bn1(x)
        x = F.relu(x)
        x = self.net.layer1[2].conv2(x)
        x = self.net.layer1[2].bn2(x)
        return x,last_layer_x
    
class NetRemain6(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        
        x = self.net.layer1[2].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead7(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        last_layer_x = self.net.layer1(x)
        x = self.net.layer2[0].conv1(last_layer_x)
        x = self.net.layer2[0].bn1(x)
        return x,last_layer_x
    
class NetRemain7(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        
        x = F.relu(x)
        x = self.net.layer2[0].conv2(x)
        x = self.net.layer2[0].bn2(x)
        x = self.net.layer2[0].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer2[1](x)
        x = self.net.layer2[2](x)
        x = self.net.layer2[3](x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead8(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        last_layer_x = self.net.layer1(x)
        x = self.net.layer2[0].conv1(last_layer_x)
        x = self.net.layer2[0].bn1(x)
        x = F.relu(x)
        x = self.net.layer2[0].conv2(x)
        x = self.net.layer2[0].bn2(x)
        return x,last_layer_x
    
class NetRemain8(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        
        x = self.net.layer2[0].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer2[1](x)
        x = self.net.layer2[2](x)
        x = self.net.layer2[3](x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead9(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        last_layer_x = self.net.layer1(x)
        x = self.net.layer2[0].conv1(last_layer_x)
        x = self.net.layer2[0].bn1(x)
        x = F.relu(x)
        x = self.net.layer2[0].conv2(x)
        x = self.net.layer2[0].bn2(x)
        st_x = self.net.layer2[0].shortcut[0](last_layer_x)
        st_x = self.net.layer2[0].shortcut[1](st_x)        
        return st_x,x
    
class NetRemain9(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, st_x,x):
        x = st_x + x
        x = F.relu(x)
        x = self.net.layer2[1](x)
        x = self.net.layer2[2](x)
        x = self.net.layer2[3](x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead10(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1(x)
        last_layer_x = self.net.layer2[0](x)
        x = self.net.layer2[1].conv1(last_layer_x)
        x = self.net.layer2[1].bn1(x)
        return x,last_layer_x
    
class NetRemain10(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        x = F.relu(x)
        x = self.net.layer2[1].conv2(x)
        x = self.net.layer2[1].bn2(x)
        x = self.net.layer2[1].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer2[2](x)
        x = self.net.layer2[3](x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead11(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1(x)
        last_layer_x = self.net.layer2[0](x)
        x = self.net.layer2[1].conv1(last_layer_x)
        x = self.net.layer2[1].bn1(x)
        x = F.relu(x)
        x = self.net.layer2[1].conv2(x)
        x = self.net.layer2[1].bn2(x)
        return x,last_layer_x
    
class NetRemain11(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        x = self.net.layer2[1].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer2[2](x)
        x = self.net.layer2[3](x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead12(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1(x)
        x = self.net.layer2[0](x)
        last_layer_x = self.net.layer2[1](x)
        x = self.net.layer2[2].conv1(last_layer_x)
        x = self.net.layer2[2].bn1(x)
        return x,last_layer_x
    
class NetRemain12(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        x = F.relu(x)
        x = self.net.layer2[2].conv2(x)
        x = self.net.layer2[2].bn2(x)
        x = self.net.layer2[2].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer2[3](x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead13(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1(x)
        x = self.net.layer2[0](x)
        last_layer_x = self.net.layer2[1](x)
        x = self.net.layer2[2].conv1(last_layer_x)
        x = self.net.layer2[2].bn1(x)
        x = F.relu(x)
        x = self.net.layer2[2].conv2(x)
        x = self.net.layer2[2].bn2(x)
        return x,last_layer_x
    
class NetRemain13(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        x = self.net.layer2[2].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer2[3](x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead14(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1(x)
        x = self.net.layer2[0](x)
        x = self.net.layer2[1](x)
        last_layer_x = self.net.layer2[2](x)
        x = self.net.layer2[3].conv1(last_layer_x)
        x = self.net.layer2[3].bn1(x)
        return x,last_layer_x
    
class NetRemain14(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        x = F.relu(x)
        x = self.net.layer2[3].conv2(x)
        x = self.net.layer2[3].bn2(x)
        x = self.net.layer2[3].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead15(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1(x)
        x = self.net.layer2[0](x)
        x = self.net.layer2[1](x)
        last_layer_x = self.net.layer2[2](x)
        x = self.net.layer2[3].conv1(last_layer_x)
        x = self.net.layer2[3].bn1(x)
        x = F.relu(x)
        x = self.net.layer2[3].conv2(x)
        x = self.net.layer2[3].bn2(x)
        return x,last_layer_x
    
class NetRemain15(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        x = self.net.layer2[3].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead16(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1(x)
        last_layer_x = self.net.layer2(x)
        x = self.net.layer3[0].conv1(last_layer_x)
        x = self.net.layer3[0].bn1(x)
        return x,last_layer_x
    
class NetRemain16(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        x = F.relu(x)
        x = self.net.layer3[0].conv2(x)
        x = self.net.layer3[0].bn2(x)
        x = self.net.layer3[0].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer3[1](x)
        x = self.net.layer3[2](x)
        x = self.net.layer3[3](x)
        x = self.net.layer3[4](x)
        x = self.net.layer3[5](x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead17(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1(x)
        last_layer_x = self.net.layer2(x)
        x = self.net.layer3[0].conv1(last_layer_x)
        x = self.net.layer3[0].bn1(x)
        x = F.relu(x)
        x = self.net.layer3[0].conv2(x)
        x = self.net.layer3[0].bn2(x)
        return x,last_layer_x
    
class NetRemain17(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        x = self.net.layer3[0].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer3[1](x)
        x = self.net.layer3[2](x)
        x = self.net.layer3[3](x)
        x = self.net.layer3[4](x)
        x = self.net.layer3[5](x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead18(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1(x)
        last_layer_x = self.net.layer2(x)
        x = self.net.layer3[0].conv1(last_layer_x)
        x = self.net.layer3[0].bn1(x)
        x = F.relu(x)
        x = self.net.layer3[0].conv2(x)
        x = self.net.layer3[0].bn2(x)
        st_x = self.net.layer3[0].shortcut[0](last_layer_x)
        st_x = self.net.layer3[0].shortcut[1](st_x)
        return st_x,x
    
class NetRemain18(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, st_x, x):
        x = st_x + x
        x = F.relu(x)
        x = self.net.layer3[1](x)
        x = self.net.layer3[2](x)
        x = self.net.layer3[3](x)
        x = self.net.layer3[4](x)
        x = self.net.layer3[5](x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead19(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        last_layer_x = self.net.layer3[0](x)
        x = self.net.layer3[1].conv1(last_layer_x)
        x = self.net.layer3[1].bn1(x)
        return x,last_layer_x
    
class NetRemain19(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        x = F.relu(x)
        x = self.net.layer3[1].conv2(x)
        x = self.net.layer3[1].bn2(x)
        x = self.net.layer3[1].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer3[2](x)
        x = self.net.layer3[3](x)
        x = self.net.layer3[4](x)
        x = self.net.layer3[5](x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead20(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        last_layer_x = self.net.layer3[0](x)
        x = self.net.layer3[1].conv1(last_layer_x)
        x = self.net.layer3[1].bn1(x)
        x = F.relu(x)
        x = self.net.layer3[1].conv2(x)
        x = self.net.layer3[1].bn2(x)
        return x,last_layer_x
    
class NetRemain20(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        x = self.net.layer3[1].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer3[2](x)
        x = self.net.layer3[3](x)
        x = self.net.layer3[4](x)
        x = self.net.layer3[5](x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead21(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3[0](x)
        last_layer_x = self.net.layer3[1](x)
        x = self.net.layer3[2].conv1(last_layer_x)
        x = self.net.layer3[2].bn1(x)
        return x,last_layer_x
    
class NetRemain21(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        x = F.relu(x)
        x = self.net.layer3[2].conv2(x)
        x = self.net.layer3[2].bn2(x)
        x = self.net.layer3[2].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer3[3](x)
        x = self.net.layer3[4](x)
        x = self.net.layer3[5](x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead22(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3[0](x)
        last_layer_x = self.net.layer3[1](x)
        x = self.net.layer3[2].conv1(last_layer_x)
        x = self.net.layer3[2].bn1(x)
        x = F.relu(x)
        x = self.net.layer3[2].conv2(x)
        x = self.net.layer3[2].bn2(x)
        return x,last_layer_x
    
class NetRemain22(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        x = self.net.layer3[2].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer3[3](x)
        x = self.net.layer3[4](x)
        x = self.net.layer3[5](x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead23(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3[0](x)
        x = self.net.layer3[1](x)
        last_layer_x = self.net.layer3[2](x)
        x = self.net.layer3[3].conv1(last_layer_x)
        x = self.net.layer3[3].bn1(x)
        return x,last_layer_x
    
class NetRemain23(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        x = F.relu(x)
        x = self.net.layer3[3].conv2(x)
        x = self.net.layer3[3].bn2(x)
        x = self.net.layer3[3].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer3[4](x)
        x = self.net.layer3[5](x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x


class NetHead24(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3[0](x)
        x = self.net.layer3[1](x)
        last_layer_x = self.net.layer3[2](x)
        x = self.net.layer3[3].conv1(last_layer_x)
        x = self.net.layer3[3].bn1(x)
        x = F.relu(x)
        x = self.net.layer3[3].conv2(x)
        x = self.net.layer3[3].bn2(x)
        return x,last_layer_x
    
class NetRemain24(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        x = self.net.layer3[3].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer3[4](x)
        x = self.net.layer3[5](x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead25(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3[0](x)
        x = self.net.layer3[1](x)
        x = self.net.layer3[2](x)
        last_layer_x = self.net.layer3[3](x)
        x = self.net.layer3[4].conv1(last_layer_x)
        x = self.net.layer3[4].bn1(x)
        return x,last_layer_x
    
class NetRemain25(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        x = F.relu(x)
        x = self.net.layer3[4].conv2(x)
        x = self.net.layer3[4].bn2(x)
        x = self.net.layer3[4].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer3[5](x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead26(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3[0](x)
        x = self.net.layer3[1](x)
        x = self.net.layer3[2](x)
        last_layer_x = self.net.layer3[3](x)
        x = self.net.layer3[4].conv1(last_layer_x)
        x = self.net.layer3[4].bn1(x)
        x = F.relu(x)
        x = self.net.layer3[4].conv2(x)
        x = self.net.layer3[4].bn2(x)
        return x,last_layer_x
    
class NetRemain26(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        x = self.net.layer3[4].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer3[5](x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead27(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3[0](x)
        x = self.net.layer3[1](x)
        x = self.net.layer3[2](x)
        x = self.net.layer3[3](x)
        last_layer_x = self.net.layer3[4](x)
        x = self.net.layer3[5].conv1(last_layer_x)
        x = self.net.layer3[5].bn1(x)
        return x,last_layer_x
    
class NetRemain27(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        x = F.relu(x)
        x = self.net.layer3[5].conv2(x)
        x = self.net.layer3[5].bn2(x)
        x = self.net.layer3[5].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead28(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3[0](x)
        x = self.net.layer3[1](x)
        x = self.net.layer3[2](x)
        x = self.net.layer3[3](x)
        last_layer_x = self.net.layer3[4](x)
        x = self.net.layer3[5].conv1(last_layer_x)
        x = self.net.layer3[5].bn1(x)
        x = F.relu(x)
        x = self.net.layer3[5].conv2(x)
        x = self.net.layer3[5].bn2(x)
        return x,last_layer_x
    
class NetRemain28(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        x = self.net.layer3[5].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead29(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        last_layer_x = self.net.layer3(x)
        x = self.net.layer4[0].conv1(last_layer_x)
        x = self.net.layer4[0].bn1(x)
        return x,last_layer_x
    
class NetRemain29(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        x = F.relu(x)
        x = self.net.layer4[0].conv2(x)
        x = self.net.layer4[0].bn2(x)
        x = self.net.layer4[0].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer4[1](x)
        x = self.net.layer4[2](x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead30(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        last_layer_x = self.net.layer3(x)
        x = self.net.layer4[0].conv1(last_layer_x)
        x = self.net.layer4[0].bn1(x)
        x = F.relu(x)
        x = self.net.layer4[0].conv2(x)
        x = self.net.layer4[0].bn2(x)
        return x,last_layer_x
    
class NetRemain30(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        x = self.net.layer4[0].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer4[1](x)
        x = self.net.layer4[2](x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead31(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        last_layer_x = self.net.layer3(x)
        x = self.net.layer4[0].conv1(last_layer_x)
        x = self.net.layer4[0].bn1(x)
        x = F.relu(x)
        x = self.net.layer4[0].conv2(x)
        x = self.net.layer4[0].bn2(x)
        st_x = self.net.layer4[0].shortcut[0](last_layer_x)
        st_x = self.net.layer4[0].shortcut[1](st_x)        
        return st_x,x
    
class NetRemain31(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, st_x,x):
        x =  st_x + x
        x = F.relu(x)
        x = self.net.layer4[1](x)
        x = self.net.layer4[2](x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead32(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        last_layer_x = self.net.layer4[0](x)
        x = self.net.layer4[1].conv1(last_layer_x)
        x = self.net.layer4[1].bn1(x)
        return x,last_layer_x
    
class NetRemain32(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        x = F.relu(x)
        x = self.net.layer4[1].conv2(x)
        x = self.net.layer4[1].bn2(x)
        x = self.net.layer4[1].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer4[2](x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead33(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        last_layer_x = self.net.layer4[0](x)
        x = self.net.layer4[1].conv1(last_layer_x)
        x = self.net.layer4[1].bn1(x)
        x = F.relu(x)
        x = self.net.layer4[1].conv2(x)
        x = self.net.layer4[1].bn2(x)
        return x,last_layer_x
    
class NetRemain33(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        x = self.net.layer4[1].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = self.net.layer4[2](x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead34(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4[0](x)
        last_layer_x = self.net.layer4[1](x)
        x = self.net.layer4[2].conv1(last_layer_x)
        x = self.net.layer4[2].bn1(x)
        return x,last_layer_x
    
class NetRemain34(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x,last_layer_x):
        x = F.relu(x)
        x = self.net.layer4[2].conv2(x)
        x = self.net.layer4[2].bn2(x)
        x = self.net.layer4[2].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x
    

class NetHead35(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = F.relu(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4[0](x)
        last_layer_x = self.net.layer4[1](x)
        x = self.net.layer4[2].conv1(last_layer_x)
        x = self.net.layer4[2].bn1(x)
        x = F.relu(x)
        x = self.net.layer4[2].conv2(x)
        x = self.net.layer4[2].bn2(x)
        return x,last_layer_x
    
class NetRemain35(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, last_layer_x):
        x = self.net.layer4[2].shortcut(last_layer_x) + x
        x = F.relu(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.net.linear(x)
        return x