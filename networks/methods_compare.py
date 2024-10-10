import torch
import torch.nn as nn
# from networks.resnet import *
# from networks.resnetv2 import *
# from networks.wresnet import *
from networks import resnet
from networks import resnetv2
from networks import wresnet
from metrics import cal_metric, print_results, print_all_results
import time
from collections import OrderedDict
from cal_method import *
from hook import *
from torch.autograd import Variable
import numpy as np
import os
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def get_square(gradients):
    if gradients[0].dim() == 1:
        gradients = [grad.unsqueeze(-1) for grad in gradients]
    gradients = torch.cat(gradients, dim=1)
    gradients = torch.pow(gradients, 2)
    var = gradients.mean(dim=(-1))
    return var

from sklearn.linear_model import LogisticRegressionCV

class Methods():
    def __init__(self, opt):
        super(Methods, self).__init__()
        self.opt = opt
        self.device = torch.device('cuda:'+f'{opt.cuda}') if torch.cuda.is_available() else 'cpu'
        device = self.device
        if opt.model_arch == 'resnet':
            self.model = resnet.KNOWN_MODELS[opt.model_name](num_classes=opt.num_classes)
            checkpoint = torch.load(opt.model_path, map_location='cpu')
        elif opt.model_arch == 'wresnet':
            self.model = wresnet.KNOWN_MODELS[opt.model_name](num_classes=opt.num_classes)
            checkpoint = torch.load(opt.model_path, map_location='cpu')
        elif opt.model_arch == 'resnetv2':
            self.model = resnetv2.KNOWN_MODELS[opt.model_name](head_size=opt.num_classes)
            checkpoint = torch.load(opt.model_path, map_location='cpu')
        
        if opt.model_arch == 'resnetv2':
            self.model.load_state_dict_custom(checkpoint['model'])
        elif 'state_dict' in checkpoint.keys():
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(device)
        self.KNOWN_METHODS = OrderedDict([
            ('cal_zero', lambda *a, **kw: cal_zero(*a, **kw)),
            ('cal_grad_value', lambda *a, **kw: cal_grad_value(*a, **kw)),
        ])
        
        
    def cal_gaia_score(self, device, dataset, hooks, dataset_name=''):       
        score = torch.tensor([], device=device)
        for data,_ in dataset:
            data = data.to(device)
            score = torch.cat([score, -self.KNOWN_METHODS[self.opt.cal_method](self.model, data, device, hooks)], 0)
        return score
    
    def get_score_gaia(self, id_dataset, ood_dataset, ood_name, device):
        
        if self.opt.hook == 'bn':
            hooks = get_bn_hooks(self.model, self.opt.model_name)
        elif self.opt.hook == 'before_head':
            hooks = get_beforehead_hooks(self.model, self.opt.model_name, self.opt.cal_method, ood_name)
            
        start_time = time.time()
        print('compute in-distribution dataset...') 
        know = self.cal_gaia_score(device, id_dataset, hooks, 'id_dataset')
        know = know.cpu().numpy()
        
        print('compute ood dataset '+ood_name+'...')
        novel = self.cal_gaia_score(device, ood_dataset, hooks, ood_name)
        novel = np.array(novel.cpu().tolist())
        
        end_time = time.time()
        result = cal_metric(know, novel)
        
        print('process result '+ood_name+' total images num: '+str(len(know) + len(novel))+'...')
        print('Computation cost: '+ str((end_time-start_time)/3600) + ' h')
        
        print_results(result, ood_name, "ours")
        return result
    
    
    def cal_msp_score(self,dataset):
        confs = []
        m = torch.nn.Softmax(dim=-1).cuda()
        for b, (x, y) in enumerate(dataset):
            with torch.no_grad():
                x = x.cuda()
                # compute output, measure accuracy and record loss.
                logits = self.model(x)

                conf, _ = torch.max(m(logits), dim=-1)
                confs.extend(conf.data.cpu().numpy())
        return np.array(confs)

    def get_score_msp(self, id_dataset, ood_dataset, ood_name, device):
        
        start_time = time.time()
        print('compute in-distribution dataset...') 
        know = self.cal_msp_score(id_dataset)
        
        print('compute ood dataset '+ood_name+'...')
        novel = self.cal_msp_score(ood_dataset)
        
        end_time = time.time()
        result = cal_metric(know, novel)
        
        print('process result '+ood_name+' total images num: '+str(len(know) + len(novel))+'...')
        print('Computation cost: '+ str((end_time-start_time)/3600) + ' h')
        
        print_results(result, ood_name, "ours")
        return result
        
    def cal_odin_score(self,dataset, epsilon, temper):
        criterion = torch.nn.CrossEntropyLoss().cuda()
        confs = []
        for b, (x, y) in enumerate(dataset):
            x = Variable(x.cuda(), requires_grad=True)
            outputs = self.model(x)

            maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
            outputs = outputs / temper

            labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
            loss = criterion(outputs, labels)
            loss.backward()

            # Normalizing the gradient to binary in {0, 1}
            gradient = torch.ge(x.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2

            # Adding small perturbations to images
            tempInputs = torch.add(x.data, -epsilon, gradient)
            outputs = self.model(Variable(tempInputs))
            outputs = outputs / temper
            # Calculating the confidence after adding perturbations
            nnOutputs = outputs.data.cpu()
            nnOutputs = nnOutputs.numpy()
            nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
            nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

            confs.extend(np.max(nnOutputs, axis=1))

        return np.array(confs)
    
    
    def get_score_odin(self, id_dataset, ood_dataset, ood_name, device):
        
        start_time = time.time()
        print('compute in-distribution dataset...') 
        know = self.cal_odin_score(id_dataset,0,1000)
        
        print('compute ood dataset '+ood_name+'...')
        novel = self.cal_odin_score(ood_dataset,0,1000)
        
        end_time = time.time()
        result = cal_metric(know, novel)
        
        print('process result '+ood_name+' total images num: '+str(len(know) + len(novel))+'...')
        print('Computation cost: '+ str((end_time-start_time)/3600) + ' h')
        
        print_results(result, ood_name, "ours")
        return result
    
    def cal_energy_score(self,dataset, temper=1):
        confs = []
        for b, (x, y) in enumerate(dataset):
            with torch.no_grad():
                x = x.cuda()
                # compute output, measure accuracy and record loss.
                logits = self.model(x)

                conf = temper * torch.logsumexp(logits / temper, dim=1)
                confs.extend(conf.data.cpu().numpy())
        return np.array(confs)
    
    def get_score_energy(self, id_dataset, ood_dataset, ood_name, device):
        start_time = time.time()
        print('compute in-distribution dataset...') 
        know = self.cal_energy_score(id_dataset,1)
        
        print('compute ood dataset '+ood_name+'...')
        novel = self.cal_energy_score(ood_dataset,1)
        
        end_time = time.time()
        result = cal_metric(know, novel)
        
        print('process result '+ood_name+' total images num: '+str(len(know) + len(novel))+'...')
        print('Computation cost: '+ str((end_time-start_time)/3600) + ' h')
        
        print_results(result, ood_name, "ours")
        return result
    
    def cal_gradnorm_score(self,dataset,temperature,num_classes):
        confs = []
        logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
        for b, (x, y) in enumerate(dataset):
            inputs = Variable(x.cuda(), requires_grad=True)

            self.model.zero_grad()
            outputs = self.model(inputs)
            targets = torch.ones((inputs.shape[0], num_classes)).cuda()
            outputs = outputs / temperature
            loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))

            loss.backward()

            # layer_grad = self.model.head.conv.weight.grad.data
            # layer_grad = self.model.layer4[1].conv2.weight.grad.data
            layer_grad = self.model.block3.layer[1].conv2.weight.grad.data
            layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
            confs.append(layer_grad_norm)

        return np.array(confs)
    

    def get_score_gradnorm(self, id_dataset, ood_dataset, ood_name, device):
        start_time = time.time()
        print('compute in-distribution dataset...') 
        know = self.cal_gradnorm_score(id_dataset,1,self.opt.num_classes)
        
        print('compute ood dataset '+ood_name+'...')
        novel = self.cal_gradnorm_score(ood_dataset,1,self.opt.num_classes)
        
        end_time = time.time()
        result = cal_metric(know, novel)
        
        print('process result '+ood_name+' total images num: '+str(len(know) + len(novel))+'...')
        print('Computation cost: '+ str((end_time-start_time)/3600) + ' h')
        
        print_results(result, ood_name, "ours")
        return result




    def get_score_rankfeat(self, id_dataset, ood_dataset, ood_name, device):
        start_time = time.time()
        print('compute in-distribution dataset...') 
        know = self.cal_rankfeat_score(id_dataset,1)
        
        print('compute ood dataset '+ood_name+'...')
        novel = self.cal_rankfeat_score(ood_dataset,1)
        
        end_time = time.time()
        result = cal_metric(know, novel)
        
        print('process result '+ood_name+' total images num: '+str(len(know) + len(novel))+'...')
        print('Computation cost: '+ str((end_time-start_time)/3600) + ' h')
        
        print_results(result, ood_name, "ours")
        return result

    # def cal_rankfeat_score(self,data_loader,temperature=1):
    #     # for resnet34
    #     model = self.model
    #     confs = []
    #     from tqdm import tqdm
    #     for b, (x, y) in tqdm(enumerate(data_loader), total=len(data_loader)):

    #         inputs = x.cuda()

    #         #Logit of Block 4 feature
    #         feat1 = model.intermediate_forward(inputs,layer_index=4)
    #         B, C, H, W = feat1.size()
    #         feat1 = feat1.view(B, C, H * W)
    #         u,s,v = torch.linalg.svd(feat1,full_matrices=False)
    #         feat1 = feat1 - s[:,0:1].unsqueeze(2)*u[:,:,0:1].bmm(v[:,0:1,:])
    #         #if you want to use PI for acceleration, comment the above 2 lines and uncomment the line below
    #         #feat1 = feat1 - power_iteration(feat1, iter=20)
    #         feat1 = feat1.view(B,C,H,W)
    #         feat1 = F.avg_pool2d(feat1, 4)
    #         feat1 = feat1.view(feat1.size(0), -1)
    #         logits1 = model.linear(feat1)

    #         # Logit of Block 4 feature
    #         feat2 = model.intermediate_forward(inputs, layer_index=3)
    #         B, C, H, W = feat2.size()
    #         feat2 = feat2.view(B, C, H * W)
    #         u, s, v = torch.linalg.svd(feat2,full_matrices=False)
    #         feat2 = feat2 - s[:, 0:1].unsqueeze(2) * u[:, :, 0:1].bmm(v[:, 0:1, :])
    #         #if you want to use PI for acceleration, comment the above 2 lines and uncomment the line below
    #         #feat2 = feat2 - power_iteration(feat2, iter=20)
    #         feat2 = feat2.view(B, C, H, W)
    #         feat2 = model.layer4(feat2)
    #         feat2 = F.avg_pool2d(feat2, 4)
    #         feat2 = feat2.view(feat2.size(0), -1)
    #         logits2 = model.linear(feat2)

    #         #Fusion at the logit space
    #         logits = (logits1+logits2) / 2
    #         conf = temperature * torch.logsumexp(logits / temperature, dim=1)
    #         confs.extend(conf.data.cpu().numpy())

    #     return np.array(confs)
    

    def cal_rankfeat_score(self,data_loader,temperature=1):
        # for wrn_40_2
        model = self.model
        confs = []
        from tqdm import tqdm
        for b, (x, y) in tqdm(enumerate(data_loader), total=len(data_loader)):

            inputs = x.cuda()

            #Logit of Block 4 feature
            feat1 = model.intermediate_forward(inputs,layer_index=3)
            B, C, H, W = feat1.size()
            feat1 = feat1.view(B, C, H * W)
            u,s,v = torch.linalg.svd(feat1,full_matrices=False)
            feat1 = feat1 - s[:,0:1].unsqueeze(2)*u[:,:,0:1].bmm(v[:,0:1,:])
            #if you want to use PI for acceleration, comment the above 2 lines and uncomment the line below
            #feat1 = feat1 - power_iteration(feat1, iter=20)
            feat1 = feat1.view(B,C,H,W)
            # feat1 = F.avg_pool2d(feat1, 4)
            feat1 = model.relu(model.bn1(feat1))
            feat1 = model.AdaptAvgPool(feat1)
            feat1 = feat1.view(feat1.size(0), -1)
            logits1 = model.fc(feat1)

            # Logit of Block 4 feature
            feat2 = model.intermediate_forward(inputs, layer_index=2)
            B, C, H, W = feat2.size()
            feat2 = feat2.view(B, C, H * W)
            u, s, v = torch.linalg.svd(feat2,full_matrices=False)
            feat2 = feat2 - s[:, 0:1].unsqueeze(2) * u[:, :, 0:1].bmm(v[:, 0:1, :])
            #if you want to use PI for acceleration, comment the above 2 lines and uncomment the line below
            #feat2 = feat2 - power_iteration(feat2, iter=20)
            feat2 = feat2.view(B, C, H, W)
            # feat2 = model.layer4(feat2)
            feat2 = model.block3(feat2)
            feat2 = model.relu(model.bn1(feat2))
            feat2 = model.AdaptAvgPool(feat2)
            feat2 = feat2.view(feat2.size(0), -1)
            logits2 = model.fc(feat2)

            #Fusion at the logit space
            logits = (logits1+logits2) / 2
            conf = temperature * torch.logsumexp(logits / temperature, dim=1)
            confs.extend(conf.data.cpu().numpy())

        return np.array(confs)

    def get_score_react(self, id_dataset, ood_dataset, ood_name, device):
        start_time = time.time()
        print('compute in-distribution dataset...') 
        know = self.cal_react_score(id_dataset,1)
        
        print('compute ood dataset '+ood_name+'...')
        novel = self.cal_react_score(ood_dataset,1)
        
        end_time = time.time()
        result = cal_metric(know, novel)
        
        print('process result '+ood_name+' total images num: '+str(len(know) + len(novel))+'...')
        print('Computation cost: '+ str((end_time-start_time)/3600) + ' h')
        
        print_results(result, ood_name, "ours")
        return result

    # def cal_react_score(self,data_loader,temperature=1):
    #     # for resnet34
    #     model = self.model
    #     confs = []
    #     for b, (x, y) in enumerate(data_loader):
    #         inputs = x.cuda()
    #         feat = model.intermediate_forward(inputs,layer_index=4)
    #         feat = F.avg_pool2d(feat, 4)
    #         feat = feat.view(feat.size(0), -1)
    #         feat = torch.clip(feat,max=1.25)
    #         logits = model.linear(feat)
    #         conf = temperature * torch.logsumexp(logits / temperature, dim=1)
    #         confs.extend(conf.data.cpu().numpy())
    #     return np.array(confs)
    
    def cal_react_score(self,data_loader,temperature=1):
        # for wrn_40_2
        model = self.model
        confs = []
        for b, (x, y) in enumerate(data_loader):
            inputs = x.cuda()
            feat = model.intermediate_forward(inputs,layer_index=3)
            feat = model.relu(model.bn1(feat))
            feat = model.AdaptAvgPool(feat)
            feat = feat.view(feat.size(0), -1)
            feat = torch.clip(feat,max=1.25)
            logits = model.fc(feat)
            conf = temperature * torch.logsumexp(logits / temperature, dim=1)
            confs.extend(conf.data.cpu().numpy())
        return np.array(confs)
    

            
    def get_scores(self, id_dataset, ood_name, ood_datasets,method="gaia"):
        device = self.device

        self.model.eval()

        results = []
        for idx_ood in range(len(ood_name)):
            if method == "gaia":
                result = self.get_score_gaia(id_dataset, ood_datasets[idx_ood], ood_name[idx_ood], device)
            elif method == "msp":
                result = self.get_score_msp(id_dataset, ood_datasets[idx_ood], ood_name[idx_ood], device)
            elif method == "odin":
                result = self.get_score_odin(id_dataset, ood_datasets[idx_ood], ood_name[idx_ood], device)
            elif method == "energy":
                result = self.get_score_energy(id_dataset, ood_datasets[idx_ood], ood_name[idx_ood], device)
            elif method == "ash":
                result = self.get_score_energy(id_dataset, ood_datasets[idx_ood], ood_name[idx_ood], device)
            elif method == "gradnorm":
                result = self.get_score_gradnorm(id_dataset, ood_datasets[idx_ood], ood_name[idx_ood], device)
            elif method == "rankfeat":
                result = self.get_score_rankfeat(id_dataset, ood_datasets[idx_ood], ood_name[idx_ood], device)
            elif method == "react":
                result = self.get_score_react(id_dataset, ood_datasets[idx_ood], ood_name[idx_ood], device)
            results.append(result)
        
        print_all_results(results, ood_name, method)