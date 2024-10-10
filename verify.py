from metrics import cal_metric, print_results, print_all_results
import numpy as np
import os
from glob import glob

import argparse

argparser = argparse.ArgumentParser()

argparser.add_argument('--path',type=str,help='Path to the file',default='')
argparser.add_argument('--max_idx',type=int,help='Max index',default=5)

args = argparser.parse_args()

know_results = np.concatenate([np.load(os.path.join(args.path,f'know_{i}.npy')) for i in range(args.max_idx)])

results = list()

if "cifar100" in args.path:
    svhn_results = np.concatenate([np.load(i) for i in glob(os.path.join(args.path,'novel_svhn_*'))])
    lsun_results = np.concatenate([np.load(i) for i in glob(os.path.join(args.path,'novel_lsun_*'))])
    tinyimagenet_results = np.concatenate([np.load(i) for i in glob(os.path.join(args.path,'novel_tinyimagenet_*'))])
    places_results = np.concatenate([np.load(i) for i in glob(os.path.join(args.path,'novel_places_*'))])
    textures_results = np.concatenate([np.load(i) for i in glob(os.path.join(args.path,'novel_textures_*'))])

    ood_name = ['svhn','lsun','tinyimagenet','places','textures']

    results = [cal_metric(know_results,svhn_results),cal_metric(know_results,lsun_results),cal_metric(know_results,tinyimagenet_results),cal_metric(know_results,places_results),cal_metric(know_results,textures_results)]

    print_all_results(results,ood_name,'ours')
elif "imagenet" in args.path:
    iNaturalist_results = np.concatenate([np.load(i) for i in glob(os.path.join(args.path,'novel_iNaturalist_*'))])
    places_results = np.concatenate([np.load(i) for i in glob(os.path.join(args.path,'novel_places_*'))])
    sun_results = np.concatenate([np.load(i) for i in glob(os.path.join(args.path,'novel_sun_*'))])
    textures_results = np.concatenate([np.load(i) for i in glob(os.path.join(args.path,'novel_textures_*'))])
    ood_name = ['iNaturalist', 'textures',  'sun', 'places']
    if "BiT-S-R101x1" in args.path:
        know_textures_results = np.concatenate([np.load(i) for i in glob(os.path.join(args.path,'know_textures_*'))])
        results = [cal_metric(know_results,iNaturalist_results),cal_metric(know_textures_results,textures_results),cal_metric(know_results,sun_results),cal_metric(know_results,places_results)]
    else:
        results = [cal_metric(know_results,iNaturalist_results),cal_metric(know_results,textures_results),cal_metric(know_results,sun_results),cal_metric(know_results,places_results)]
    print_all_results(results,ood_name,'ours')
elif "cifar10" in args.path:
    ood_name = ['svhn',  'lsun',  'tinyimagenet',  'places', 'textures']
    
    svhn_results = np.concatenate([np.load(i) for i in glob(os.path.join(args.path,'novel_svhn_*'))])
    lsun_results = np.concatenate([np.load(i) for i in glob(os.path.join(args.path,'novel_lsun_*'))])
    tinyimagenet_results = np.concatenate([np.load(i) for i in glob(os.path.join(args.path,'novel_tinyimagenet_*'))])
    places_results = np.concatenate([np.load(i) for i in glob(os.path.join(args.path,'novel_places_*'))])
    textures_results = np.concatenate([np.load(i) for i in glob(os.path.join(args.path,'novel_textures_*'))])
    if "BiT-S-R101x1" in args.path:
        know_textures_results = np.concatenate([np.load(i) for i in glob(os.path.join(args.path,'know_textures_*'))])
        results = [cal_metric(know_results,svhn_results),cal_metric(know_results,lsun_results),cal_metric(know_results,tinyimagenet_results),cal_metric(know_results,places_results),cal_metric(know_textures_results,textures_results)]
    else:
        results = [cal_metric(know_results,svhn_results),cal_metric(know_results,lsun_results),cal_metric(know_results,tinyimagenet_results),cal_metric(know_results,places_results),cal_metric(know_results,textures_results)]
    print_all_results(results,ood_name,'ours')