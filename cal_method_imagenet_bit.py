import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
# torch.cuda.set_device(args.cuda)

def get_square(gradients):
    if gradients[0].dim() == 1:
        gradients = [grad.unsqueeze(-1) for grad in gradients]
    gradients = torch.cat(gradients, dim=1)
    gradients = torch.pow(gradients, 2)
    var = gradients.mean(dim=(-1))
    return var


# def cal_zero(net, input, device=None, hooks=None):
   
#     input,target = input
#     input = input.to(device)
#     target = target.to(device)
#     gradients = None
#     last_outputs = None
#     last_grads = None
#     print(len(hooks))
#     raise ValueError
#     for _ in range(5):
#         input = Variable(input, requires_grad=True)
#         net.zero_grad()
#         y = net(input)
#         y.max(dim=1).values.sum().backward()
#         if last_grads is None and last_outputs is None:
#             last_grads = [hook.data for hook in hooks]
#             last_outputs = [hook.feature for hook in hooks]
#         else:
#             grads = [hook.data for hook in hooks]
#             outputs = [hook.feature for hook in hooks]
#             if gradients is None:
#                 gradients = [(o - ol) * gl for o, ol, gl in zip(outputs, last_outputs, last_grads)]
#             else:
#                 gradients = [(o - ol) * gl + g for o, ol, gl, g in zip(outputs, last_outputs, last_grads, gradients)]
#             last_grads = grads
#             last_outputs = outputs
#         input = input + 1/255 * input.grad.sign()
#         input = torch.clamp(input, 0, 1)
#     # gradients = [hook.data for hook in hooks]
#     # gradients = [torch.where(grad != 0, torch.ones_like(grad), torch.zeros_like(grad)) for grad in gradients]
#     scores = [grad.mean(dim=(-1, -2)) for grad in gradients]
#     square_scores = get_square(scores)
#     return square_scores


# from split_resnet import *
from split_resnetv2 import *
    


def cal_zero(net, input, device=None, hooks=None, loss_method="CE",alpha=1/255, method='multiply'):
    input,target = input
    input, target = input.to(device), target.to(device)
    target = None

    # y = net(input)
    # y.max(dim=1).values.sum().backward()
    # gradients_old = [hook.data for hook in hooks]

    gradients = list()
    
    namespaces = globals()
    namespaces.update(locals())
    # all_models = [
    #     eval(f"NetHead{i}(net), NetRemain{i}(net)", namespaces) for i in range(36)
    # ]
    
    # all_idx = list(range(36))
    all_models = [
        (eval(f"NetHead{i}(net), NetRemain{i}(net)", namespaces)) for i in range(10)
    ]
    if hooks[-1].module_name == 'GroupNorm':
        all_models[-1] = (NetHead9SpeC(net), NetRemain9SpeC(net))
    all_idx = list(range(10))
    original_input = input
    for idx, (head, remain) in zip(all_idx, all_models):
        # print(idx)
        net.zero_grad()
        target = None
        input = head(original_input)
        if isinstance(input,tuple):
            input,last_layer_x = input
        else:
            last_layer_x = None
        gradients_bn = None
        last_outputs = None
        last_grads = None
        for _ in range(2):
            input = Variable(input, requires_grad=True)
            net.zero_grad()
            if last_layer_x is not None:
                y = remain(input, last_layer_x)
            else:
                y = remain(input)
            # loss = y.max(dim=1).values.sum()
            loss = y.max(dim=1).values
            gd = torch.autograd.grad(loss, input, create_graph=True,grad_outputs=torch.ones_like(loss))[0]
            
            # print((gradients_old[idx] - gd).sum().item())
            
            if last_grads is None and last_outputs is None:
                last_grads = gd.detach()
                last_outputs = input.detach().clone()
            else:
                outputs = input.detach().clone()
                last_grads = gd.detach()
                last_outputs = outputs

            if loss_method == "CE":
                ce = nn.CrossEntropyLoss(reduction='none')
                if target is None:
                    target = y.argmax(dim=-1)
                loss = ce(y, target)
                grad = torch.autograd.grad(loss, input, create_graph=True,grad_outputs=torch.ones_like(loss))[0]
            elif loss_method == "MAX":
                grad = gd
            input = input + alpha * grad.sign()
            # input = input + 0.001 * grad.sign()

            # if gradients_bn is None:
            #     gradients_bn = torch.where(last_grads != 0, torch.ones_like(last_grads), torch.zeros_like(last_grads))
            # else:
            #     gradients_bn += torch.where(last_grads != 0, torch.ones_like(last_grads), torch.zeros_like(last_grads))
                # gradients_bn *= torch.where(last_grads != 0, torch.ones_like(last_grads), torch.zeros_like(last_grads))
                
            # if gradients_bn is None:
            #     gradients_bn = torch.where(last_grads != 0, torch.ones_like(last_grads), torch.zeros_like(last_grads))
            # else:
            #     if method == 'multiply':
            #         gradients_bn *= torch.where(last_grads != 0, torch.ones_like(last_grads), torch.zeros_like(last_grads))
            #     elif method == 'add':
            #         gradients_bn += torch.where(last_grads != 0, torch.ones_like(last_grads), torch.zeros_like(last_grads))
            if gradients_bn is None:
                gradients_bn = last_grads
            else:
                if method == 'multiply':
                    gradients_bn *= last_grads
                elif method == 'add':
                    gradients_bn += last_grads
        gradients_bn = torch.where(gradients_bn != 0, torch.ones_like(gradients_bn), torch.zeros_like(gradients_bn))
        gradients.append(gradients_bn)
    
    # raise ValueError
    scores = [grad.mean(dim=(-1, -2)) for grad in gradients]
    square_scores = get_square(scores)
    return square_scores





# def cal_zero(net, input, device=None, hooks=None):
#     net.zero_grad()
#     input, target = input
#     input, target = input.to(device), target.to(device)
#     y = net(input)
#     # print(y.argmax(-1).eq(target).float().mean())
#     # raise ValueError
#     y.max(dim=1).values.sum().backward()
#     gradients = [hook.data for hook in hooks]

#     gradients = [torch.where(grad != 0, torch.ones_like(grad), torch.zeros_like(grad)) for grad in gradients]
    
#     scores = [grad.mean(dim=(-1, -2)) for grad in gradients]
#     square_scores = get_square(scores)
#     return square_scores


def cal_grad_value(net, input, device=None, hooks=None, loss_method="CE",alpha=1/255, method='multiply'):
    input, target = input
    input, target = input.to(device), target.to(device)
    net.zero_grad()
    # y = net(input)
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()    
    namespaces = globals()
    namespaces.update(locals())

    all_models = [
        (eval(f"NetHead{i}(net), NetRemain{i}(net)", namespaces)) for i in range(10)
    ]
    if hooks[-1].module_name == 'GroupNorm':
        all_models[-1] = (NetHead9SpeC(net), NetRemain9SpeC(net))
    original_input = input

    head, remain = all_models[-1]
    net.zero_grad()
    target = None
    input = head(original_input)
    if isinstance(input,tuple):
        input,last_layer_x = input
    else:
        last_layer_x = None
    gradients_bn = None
    last_outputs = None
    last_grads = None
    for _ in range(2):
        input = Variable(input, requires_grad=True)
        net.zero_grad()
        if last_layer_x is not None:
            y,bh = remain(input, last_layer_x)
        else:
            y,bh = remain(input)
        loss = logsoftmax(y)
        gd = torch.autograd.grad(loss, input, create_graph=True,grad_outputs=torch.ones_like(loss))[0]

        
        if last_grads is None and last_outputs is None:
            last_grads = gd.detach()
            last_outputs = input.detach().clone()
        else:
            outputs = input.detach().clone()
            last_grads = gd.detach()
            last_outputs = outputs

        if loss_method == "CE":
            ce = nn.CrossEntropyLoss(reduction='none')
            if target is None:
                target = y.argmax(dim=-1)
            loss = ce(y, target)
            grad = torch.autograd.grad(loss, input, create_graph=True,grad_outputs=torch.ones_like(loss))[0]
        elif loss_method == "MAX":
            grad = gd
        input = input + alpha * grad.sign()

        if gradients_bn is None:
            gradients_bn = last_grads
        else:
            if method == 'multiply':
                gradients_bn *= last_grads
            elif method == 'add':
                gradients_bn += last_grads
    
    # loss = logsoftmax(y)
    # loss.sum().backward(retain_graph=True)
    # before_head_grad = hooks[-1].data.mean(dim=(-1, -2))
    before_head_grad = gradients_bn.mean(dim=(-1, -2))
    output_component = torch.sqrt(torch.abs(before_head_grad).mean(dim=1))
    output_component = output_component.unsqueeze(dim=1)


    gradients = list()
    all_idx = list(range(9))
    for idx, (head, remain) in zip(all_idx, all_models[:-1]):
        # print(idx)
        net.zero_grad()
        target = None
        input = head(original_input)
        if isinstance(input,tuple):
            input,last_layer_x = input
        else:
            last_layer_x = None
        gradients_bn = None
        last_outputs = None
        last_grads = None
        for _ in range(2):
            input = Variable(input, requires_grad=True)
            net.zero_grad()
            if last_layer_x is not None:
                y,bh = remain(input, last_layer_x)
            else:
                y,bh = remain(input)
            loss = bh
            gd = torch.autograd.grad(loss, input, create_graph=True,grad_outputs=torch.ones_like(loss))[0]
            
            
            if last_grads is None and last_outputs is None:
                last_grads = gd.detach()
                last_outputs = input.detach().clone()
            else:
                outputs = input.detach().clone()
                last_grads = gd.detach()
                last_outputs = outputs

            if loss_method == "CE":
                ce = nn.CrossEntropyLoss(reduction='none')
                if target is None:
                    target = y.argmax(dim=-1)
                loss = ce(y, target)
                grad = torch.autograd.grad(loss, input, create_graph=True,grad_outputs=torch.ones_like(loss))[0]
            elif loss_method == "MAX":
                grad = gd
            input = input + alpha * grad.sign()
            if gradients_bn is None:
                gradients_bn = last_grads
            else:
                if method == 'multiply':
                    gradients_bn *= last_grads
                elif method == 'add':
                    gradients_bn += last_grads
        gradients.append(gradients_bn)
    # loss = net.before_head_data
    # loss.sum().backward()
    # gradients = [hook.data for hook in hooks]
    # gradients = gradients[:-1]
    gradients = [grad.mean(dim=(-1, -2)) for grad in gradients]
    inner_component = torch.abs(torch.cat(gradients, dim=1))
    score = torch.pow(inner_component / output_component, 2).mean(dim=1)


    return score.detach()














# from hook import *

# def cal_zero(net, input, device=None, hooks=None):
#     net.zero_grad()
#     hooks1,hooks2 = hooks
#     input, target = input
#     ori_input, target = input.to(device), target.to(device)
#     gradients = list()
#     for idx in range(len(hooks1)):
#         new_input = None
#         grads_bn = None
#         for _ in range(5):
#             y = net(ori_input).max(dim=1).values
#             if new_input is None:
#                 new_input = hooks2[idx].data
#                 grad = torch.autograd.grad(y, new_input, create_graph=True,grad_outputs=torch.ones_like(y))[0]
#             else:
#                 grad = torch.autograd.grad(y, new_input, create_graph=True,grad_outputs=torch.ones_like(y),allow_unused=True)[0]
#                 new_input = new_input + grad.sign() * 1/255
#                 new_input = torch.autograd.Variable(new_input, requires_grad=True)
#                 module = hooks2[idx].module
#                 hooks2[idx] = Feature_mod_hook(module, new_input)
#             if grads_bn is None:
#                 grads_bn = torch.where(grad != 0, torch.ones_like(grad), torch.zeros_like(grad))
#             else:
#                 grads_bn += torch.where(grad != 0, torch.ones_like(grad), torch.zeros_like(grad))
            
#         gradients.append(grads_bn)
#         hooks2[idx] = Feature_mod_hook(module)
        
#     scores = [grad.mean(dim=(-1, -2)) for grad in gradients]
#     square_scores = get_square(scores)
#     return square_scores