import os
import multiprocessing
import argparse

pool = multiprocessing.Pool(processes=1)

# fixed = "python eval.py -dataset cifar10 -model_arch wresnet -model_name wrn_40_2 -num_classes 10 -score GAIA -cal_method cal_zero -hook bn -data_dir ./datasets/ -model_path ./checkpoint/models/cifar10_wrn40_2.pth -batch_size 128 -num_workers 2 --type S-I"

# fixed = "python eval.py -dataset cifar10 -model_arch resnet -model_name resnet34 -num_classes 10 -score GAIA -cal_method cal_zero -hook bn -data_dir ./datasets/ -model_path ./checkpoint/models/cifar10_resnet34.pth -batch_size 128 -num_workers 2 --type baseline"
# fixed = "python eval.py -dataset cifar10 -model_arch resnet -model_name resnet34 -num_classes 10 -score GAIA -cal_method cal_zero -hook bn -data_dir ./datasets/ -model_path ./checkpoint/models/cifar10_resnet34.pth -batch_size 128 -num_workers 2 --type S-I"
# fixed = "python eval.py -dataset cifar10 -model_arch resnetv2 -model_name BiT-S-R101x1 -num_classes 10 -score GAIA -cal_method cal_zero -hook before_head -data_dir ./datasets/ -model_path ./BiT-S-R101x1-flat-finetune_cifar10.pth.tar -batch_size 16 -num_workers 2 --type baseline"
# fixed = "python eval.py -dataset cifar10 -model_arch resnetv2 -model_name BiT-S-R101x1 -num_classes 10 -score GAIA -cal_method cal_zero -hook before_head -data_dir ./datasets/ -model_path ./BiT-S-R101x1-flat-finetune_cifar10.pth.tar -batch_size 16 -num_workers 2 --type S-I"

# fixed = "python eval.py -dataset cifar100 -model_arch resnet -model_name resnet34 -num_classes 100 -score GAIA -cal_method cal_zero -hook bn -data_dir ./datasets/ -model_path ./checkpoint/models/cifar100_resnet34.pth -batch_size 128 -num_workers 2 --type baseline"
# fixed = "python eval.py -dataset cifar100 -model_arch resnet -model_name resnet34 -num_classes 100 -score GAIA -cal_method cal_zero -hook bn -data_dir ./datasets/ -model_path ./checkpoint/models/cifar100_resnet34.pth -batch_size 128 -num_workers 2 --type S-I"
# fixed = "python eval.py -dataset cifar100 -model_arch resnetv2 -model_name BiT-S-R101x1 -num_classes 100 -score GAIA -cal_method cal_zero -hook before_head -data_dir ./datasets/ -model_path ./BiT-S-R101x1-flat-finetune_cifar100.pth.tar -batch_size 16 -num_workers 2 --type baseline"
# fixed = "python eval.py -dataset cifar100 -model_arch resnetv2 -model_name BiT-S-R101x1 -num_classes 100 -score GAIA -cal_method cal_zero -hook before_head -data_dir ./datasets/ -model_path ./BiT-S-R101x1-flat-finetune_cifar100.pth.tar -batch_size 16 -num_workers 2 --type S-I"
# fixed = "python eval.py -dataset cifar100 -model_arch wresnet -model_name wrn_40_2 -num_classes 100 -score GAIA -cal_method cal_zero -hook bn -data_dir ./datasets/ -model_path ./checkpoint/models/cifar100_wrn40_2.pth -batch_size 128 -num_workers 2 --type baseline"
# fixed = "python eval.py -dataset cifar100 -model_arch wresnet -model_name wrn_40_2 -num_classes 100 -score GAIA -cal_method cal_zero -hook bn -data_dir ./datasets/ -model_path ./checkpoint/models/cifar100_wrn40_2.pth -batch_size 128 -num_workers 2 --type S-I"

# fixed = "python eval.py -dataset imagenet -model_arch resnetv2 -model_name BiT-S-R101x1 -num_classes 1000 -score GAIA -cal_method cal_zero -hook before_head -data_dir ./datasets/ -model_path ./BiT-S-R101x1-flat-finetune.pth.tar -batch_size 16 -num_workers 2 --type baseline"
# fixed = "python eval.py -dataset imagenet -model_arch resnetv2 -model_name BiT-S-R101x1 -num_classes 1000 -score GAIA -cal_method cal_zero -hook before_head -data_dir ./datasets/ -model_path ./BiT-S-R101x1-flat-finetune.pth.tar -batch_size 16 -num_workers 2 --type S-I"
# fixed = "python eval.py -dataset imagenet -model_arch resnetv2 -model_name BiT-S-R101x1 -num_classes 1000 -score GAIA -cal_method cal_grad_value -hook before_head -data_dir ./datasets/ -model_path ./BiT-S-R101x1-flat-finetune.pth.tar -batch_size 16 -num_workers 2 --type baseline"
# loss_methods = ["CE", "MAX"]
loss_methods = ["CE"]
methods = ["add"]
alphas = ["0.001"]

cuda_flag = True

command = list()

all_fixed = [
    "python eval.py -dataset imagenet -model_arch resnetv2 -model_name BiT-S-R101x1 -num_classes 1000 -score GAIA -cal_method cal_grad_value -hook before_head -data_dir ./datasets/ -model_path ./BiT-S-R101x1-flat-finetune.pth.tar -batch_size 16 -num_workers 0 --type S-I",
]
for fixed in all_fixed:
    for loss_method in loss_methods:
        for method in methods:
            for alpha in alphas:
                for idx in range(1):
                    command.append(fixed + f" --loss_method {loss_method} --method {method} --alpha {alpha} -cuda {0 if cuda_flag else 1} --idx {idx}")
                    cuda_flag = not cuda_flag

print(command)

print(len(command))

pool.imap(os.system, command)
pool.close()
pool.join()

