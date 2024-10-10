import torch
import torch.nn as nn

from networks import resnet
from networks import resnetv2
from networks import wresnet
from metrics import cal_metric, print_results, print_all_results
import time
from collections import OrderedDict
from hook import *
import numpy as np

class Methods():
    def __init__(self, opt):
        super(Methods, self).__init__()
        self.opt = opt
        if self.opt.type == "S-I":
            if self.opt.model_name == "resnet34":
                from cal_method_cifar100_resnet34 import cal_zero, cal_grad_value
            elif self.opt.model_name == 'BiT-S-R101x1':
                from cal_method_imagenet_bit import cal_zero, cal_grad_value
            elif self.opt.model_name == "wrn_40_2":
                from cal_method_cifar100_wrn import cal_zero, cal_grad_value
        elif self.opt.type == "baseline":
            from cal_method import cal_zero, cal_grad_value
        self.device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
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
            if opt.dataset not in ["cifar10", "cifar100"]:
                self.model.load_state_dict_custom(checkpoint['model'])
            else:
                checkpoint = checkpoint['model']
                checkpoint['module.before_head.gn.weight'] = checkpoint['module.head.gn.weight']
                checkpoint['module.before_head.gn.bias'] = checkpoint['module.head.gn.bias']
                del checkpoint['module.head.gn.weight']
                del checkpoint['module.head.gn.bias']
                self.model.load_state_dict_custom(checkpoint)
        elif 'state_dict' in checkpoint.keys():
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(device)
        self.KNOWN_METHODS = OrderedDict([
            ('cal_zero', lambda *a, **kw: cal_zero(*a, **kw)),
            ('cal_grad_value', lambda *a, **kw: cal_grad_value(*a, **kw)),
        ])
        
        self.know = None
        

    def cal_score(self, device, dataset, hooks, dataset_name=''):
        from tqdm import tqdm     
        score = torch.tensor([], device=device)
        for data in tqdm(dataset, desc=dataset_name):
            score = torch.cat([score, -self.KNOWN_METHODS[self.opt.cal_method](self.model, data, device, hooks,loss_method=self.opt.loss_method,alpha=self.opt.alpha,method=self.opt.method)], 0)
        return score

    def get_score(self, id_dataset, ood_dataset, ood_name, device):
        import os
        if self.opt.hook == 'bn':
            hooks = get_bn_hooks(self.model, self.opt.model_name)
        elif self.opt.hook == 'before_head':
            hooks = get_beforehead_hooks(self.model, self.opt.model_name, self.opt.cal_method, ood_name)
            
        print('compute in-distribution dataset...') 
        
        if self.opt.model_name == 'BiT-S-R101x1' and ood_name == 'textures':
            if not os.path.exists(f"{self.opt.save_dir}/know_textures_{self.opt.idx}.npy"):
                know = self.cal_score(device, id_dataset, hooks, 'id_dataset')
                know = know.cpu().numpy()
                np.save(f"{self.opt.save_dir}/know_textures_{self.opt.idx}.npy", know)
        else:
            if not os.path.exists(f"{self.opt.save_dir}/know_{self.opt.idx}.npy"):
                if self.know is None:
                    know = self.cal_score(device, id_dataset, hooks, 'id_dataset')
                    know = know.cpu().numpy()
                    self.know = know
                else:
                    know = self.know
                np.save(f"{self.opt.save_dir}/know_{self.opt.idx}.npy", know)
        if not os.path.exists(f"{self.opt.save_dir}/novel_{ood_name}_{self.opt.idx}.npy"):
            print('compute ood dataset '+ood_name+'...')
            novel = self.cal_score(device, ood_dataset, hooks, ood_name)
            novel = np.array(novel.cpu().tolist())
            np.save(f"{self.opt.save_dir}/novel_{ood_name}_{self.opt.idx}.npy", novel)
        
        
        
    def get_scores(self, id_dataset, ood_name, ood_datasets):
        device = self.device
        self.model.eval()
        assert ood_name[0] != "textures"
        for idx_ood in range(len(ood_name)):
            self.get_score(id_dataset, ood_datasets[idx_ood], ood_name[idx_ood], device)