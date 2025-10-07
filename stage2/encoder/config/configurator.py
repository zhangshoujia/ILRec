import os
import yaml
import json
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn

def parse_configure(model=None, dataset=None):
    parser = argparse.ArgumentParser(description='RLMRec')
    parser.add_argument('--model', type=str, default='LightGCN', help='Model name')
    parser.add_argument('--dataset', type=str, default='amazon', help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--cuda', type=str, default='0', help='CUDA visible device ids')
    args, _ = parser.parse_known_args()

    # CUDA
    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    # model name
    if model is not None:
        model_name = model.lower()
    elif args.model is not None:
        model_name = args.model.lower()
    else:
        model_name = 'default'
        # print("Read the default (blank) configuration.")

    # dataset
    if dataset is not None:
        args.dataset = dataset

    # find yml file
    if not os.path.exists('./encoder/config/modelconf/{}.yml'.format(model_name)):
        raise Exception("Please create the yaml file for your model first.")

    # read yml file
    with open('./encoder/config/modelconf/{}.yml'.format(model_name), encoding='utf-8') as f:
        config_data = f.read()
        configs = yaml.safe_load(config_data)
        configs['model']['name'] = configs['model']['name'].lower()
        if 'tune' not in configs:
            configs['tune'] = {'enable': False}
        configs['device'] = args.device
        if args.dataset is not None:
            configs['data']['name'] = args.dataset
        if args.seed is not None:
            configs['train']['seed'] = args.seed

        try:
            overrides_str = os.environ.get('RLMREC_TUNING_OVERRIDES')
            if overrides_str:
                overrides = json.loads(overrides_str)
                # Target paths: model root and model.<dataset>
                model_root = configs.setdefault('model', {})
                ds_name = configs.get('data', {}).get('name')
                if ds_name:
                    model_ds = model_root.setdefault(ds_name, {})
                else:
                    model_ds = None

                for k, v in overrides.items():
                    # Write to model root
                    model_root[k] = v
                    # Compatibility for key sup_cl_rate
                    if k == 'sup_cl_weight':
                        model_root['sup_cl_rate'] = v
                    # Write to model.<dataset>
                    if model_ds is not None:
                        model_ds[k] = v
                        if k == 'sup_cl_weight':
                            model_ds['sup_cl_rate'] = v
        except Exception as e:
            print(f"⚠️  Failed to apply tuning overrides from environment variable: {e}")

        data_name = configs['data']['name']
        usrprf_embeds_path = "./data/" + data_name + "/ft_usr_emb.pkl"
        itmprf_embeds_path = "./data/" + data_name + "/ft_itm_emb.pkl"
        configs['sem_graph_path'] = "./data/" + data_name + "/semantic_graph_top" +  str(configs['model']['semantic_topk'])  + "_mat.pkl"
        
        with open(usrprf_embeds_path, 'rb') as f:
            configs['usrprf_embeds'] = pickle.load(f)
        with open(itmprf_embeds_path, 'rb') as f:
            configs['itmprf_embeds'] = pickle.load(f)

        return configs

configs = parse_configure()