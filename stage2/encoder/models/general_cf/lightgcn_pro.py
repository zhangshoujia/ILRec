import math
import torch as t
from torch import nn
import torch.nn.functional as F
import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params, ssl_con_loss, make_torch_adj, load_one_mat, cal_infonce_loss, cal_sup_cl_loss
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_


class LightGCN_pro(BaseModel):


    def __init__(self, data_handler):
        super(LightGCN_pro, self).__init__(data_handler)

        self.data_handler = data_handler
        self.adj = data_handler.torch_adj
        self.keep_rate = configs['model']['keep_rate']
        self.edge_dropper = SpAdjEdgeDrop()
        self.final_embeds = None
        self.is_training = False

        self.layer_num = int(self.hyper_config.get('layer_num', 3))
        self.reg_weight = float(self.hyper_config.get('reg_weight', 0.0))
        self.embedding_size = int(self.hyper_config.get('embedding_size', self.embedding_size))

        self.semantic_topk = int(self.hyper_config.get('semantic_topk', 10))

        self.sup_cl_weight = float(self.hyper_config.get('sup_cl_weight', 0.1))
        self.cv_sup_cl_weight = float(self.hyper_config.get('cv_sup_cl_weight', 0.1))
        self.emb_cl_weight = float(self.hyper_config.get('emb_cl_weight', 0.0))
        self.contrast_temperature = float(self.hyper_config.get('contrast_temperature', 0.1))
        self.sup_contrast_temperature = float(self.hyper_config.get('sup_contrast_temperature', 0.25))

        self.debug_log = bool(self.hyper_config.get('debug_log', False))
        self.max_debug_steps = int(self.hyper_config.get('max_debug_steps', 2))
        self._debug_count = 0


        target_device = configs['device']
        self.usrprf_embeds = t.tensor(configs['usrprf_embeds']).float().to(target_device)
        self.itmprf_embeds = t.tensor(configs['itmprf_embeds']).float().to(target_device)
        
        self.register_buffer('semantic_user_embeds', self.usrprf_embeds)
        self.register_buffer('semantic_item_embeds', self.itmprf_embeds)
        
        self.semantic_embedding_size = self.usrprf_embeds.shape[1]

        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        
        print(f"✓ Semantic embeddings and model embeddings are both on device: {target_device}")

        self._try_load_semantic_graph_mat()

        target_device = configs['device']
        if self.semantic_adj.device != target_device:
            self.semantic_adj = self.semantic_adj.to(target_device)
            print(f"✓ Semantic graph has been moved to the target device: {target_device}")

        self.semantic_mlps = nn.Sequential(
            nn.Linear(self.semantic_embedding_size, (self.embedding_size + self.semantic_embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.embedding_size + self.semantic_embedding_size) // 2, self.embedding_size)
        )

        self._init_weight()

        def _init_weight(self):
            for m in self.semantic_mlps:
                if isinstance(m, nn.Linear):
                    init(m.weight)


        def _try_load_semantic_graph_mat(self):
            """Try to load the precomputed semantic interaction graph using an adjacency matrix."""
            try:
                precomputed_path = configs['sem_graph_path']

                if os.path.exists(precomputed_path):
                    trn_mat = load_one_mat(precomputed_path)
                    self.semantic_adj = make_torch_adj(trn_mat)
                    if self.semantic_adj.device != configs['device']:
                        if self._debug_count < 3:
                            print(f"⚠️ Device mismatch detected. Moving semantic graph from {self.semantic_adj.device} to {configs['device']}")
                        self.semantic_adj = self.semantic_adj.to(configs['device'])

            except Exception as e:
                print(f"⚠️ Failed to load the precomputed semantic graph: {e}")
                print("  Continue using the dynamically constructed semantic graph")

        def _propagate(self, adj, embeds):
            """Graph propagation."""
            try:
                if adj.device != embeds.device:
                    print(f"⚠️ Device mismatch before propagation: adj={adj.device}, embeds={embeds.device}")
                    adj = adj.to(embeds.device)

                if hasattr(adj, '_indices'):
                    indices = adj._indices()
                    if indices.numel() > 0:
                        max_row = indices[0].max().item()
                        max_col = indices[1].max().item()
                        if max_row >= embeds.size(0) or max_col >= embeds.size(0):
                            print(f"⚠️ Adjacency matrix index out of range: max_row={max_row}, max_col={max_col}, embeds_size={embeds.size(0)}")
                            return embeds

                return t.spmm(adj, embeds)
            except Exception as e:
                print(f"⚠️ Graph propagation failed: {e}")
                return embeds

        @staticmethod
        def _avg_l2_norm(x: t.Tensor):
            if x is None:
                return float('nan')
            with t.no_grad():
                return t.norm(x, p=2, dim=1).mean().item()

        def forward(self, adj=None, keep_rate=1.0, batch_data=None):
            if adj is None:
                adj = self.adj
                semantic_adj = self.semantic_adj
            if not self.is_training and self.final_embeds is not None:
                return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]

            embeds = t.cat([self.user_embeds, self.item_embeds], dim=0)
            orig_embeds_list = [embeds]

            seman_embeds = t.cat([self.semantic_user_embeds, self.semantic_item_embeds], dim=0)
            sem_embeds_list = [seman_embeds]

            if self.is_training:
                adj = self.edge_dropper(adj, keep_rate)
                semantic_adj = self.edge_dropper(self.semantic_adj, keep_rate)
            else:
                adj = self.adj
                semantic_adj = self.semantic_adj

            if semantic_adj.device != embeds.device:
                if self._debug_count < 3:
                    print(f"⚠️ Device mismatch detected. Moving semantic graph from {semantic_adj.device} to {embeds.device}")
                semantic_adj = semantic_adj.to(embeds.device)

            if adj.device != embeds.device:
                if self._debug_count < 3:
                    print(f"⚠️ Original graph device mismatch detected. Moving from {adj.device} to {embeds.device}")
                adj = adj.to(embeds.device)

        for layer_idx in range(self.layer_num):
            orig_embeds = self._propagate(adj, orig_embeds_list[-1])
            orig_embeds_list.append(orig_embeds)
            
            sem_embeds = self._propagate(semantic_adj, sem_embeds_list[-1])
            sem_embeds_list.append(sem_embeds)
            
        final_orig_embeds = sum(orig_embeds_list)
        final_sem_embeds = sum(sem_embeds_list)
        
        orig_user_embeds = final_orig_embeds[:self.user_num]
        orig_item_embeds = final_orig_embeds[self.user_num:]
        sem_user_embeds = final_sem_embeds[:self.user_num]
        sem_item_embeds = final_sem_embeds[self.user_num:]
            
        return orig_user_embeds, orig_item_embeds, sem_user_embeds, sem_item_embeds

    def _compute_contrastive_loss(self, orig_interactions, sem_interactions, contrast_temperature):
        if self.interaction_contrast_weight <= 0 or orig_interactions is None or sem_interactions is None:
            return 0.0
        
        contrast_loss = ssl_con_loss(orig_interactions, sem_interactions, contrast_temperature)
        return contrast_loss

    def cal_loss(self, batch_data):
        self.is_training = True
        orig_user_embeds, orig_item_embeds, sem_user_embeds, sem_item_embeds = self.forward(self.adj, self.keep_rate, batch_data)
        sem_user_embeds = self.semantic_mlps(sem_user_embeds)
        sem_item_embeds = self.semantic_mlps(sem_item_embeds)

        orig_user_embeds, orig_item_embeds = F.normalize(orig_user_embeds, dim=-1), F.normalize(orig_item_embeds, dim=-1)
        sem_user_embeds, sem_item_embeds = F.normalize(sem_user_embeds, dim=-1), F.normalize(sem_item_embeds, dim=-1)
        
        ancs, poss, negs = batch_data
        behavior_anc_embeds = orig_user_embeds[ancs]
        behavior_pos_embeds = orig_item_embeds[poss]
        behavior_neg_embeds = orig_item_embeds[negs]

        sem_anc_embeds = sem_user_embeds[ancs]
        sem_pos_embeds = sem_item_embeds[poss]
        sem_neg_embeds = sem_item_embeds[negs]
        
        unique_user_idx = t.unique(t.Tensor(ancs).type(t.long)).cuda()
        unique_item_idx = t.unique(t.Tensor(poss).type(t.long)).cuda()

        unique_user_emb, unique_item_emb = orig_user_embeds[unique_user_idx], orig_item_embeds[unique_item_idx]
        node_emb = t.cat((unique_user_emb, unique_item_emb), dim=0)
        sup_cl_loss = cal_sup_cl_loss(behavior_anc_embeds, behavior_pos_embeds, node_emb, self.sup_contrast_temperature)
        sup_cl_loss *= self.sup_cl_weight
        unique_sem_user_emb, unique_sem_item_emb = sem_user_embeds[unique_user_idx], sem_item_embeds[unique_item_idx]
        sem_node_emb = t.cat((unique_sem_user_emb, unique_sem_item_emb), dim=0)
       
        beh_sem_emb = t.cat((unique_user_emb, unique_sem_item_emb), dim=0)
        sem_beh_emb = t.cat((unique_sem_user_emb, unique_item_emb), dim=0)
        sup_cl_user_loss = cal_sup_cl_loss(behavior_anc_embeds, sem_pos_embeds, beh_sem_emb, self.sup_contrast_temperature)
        sup_cl_item_loss = cal_sup_cl_loss(sem_anc_embeds, behavior_pos_embeds, sem_beh_emb, self.sup_contrast_temperature)

        full_sup_cl_loss = (sup_cl_user_loss + sup_cl_item_loss) * self.cv_sup_cl_weight

        bpr_loss = cal_bpr_loss(behavior_anc_embeds, behavior_pos_embeds, behavior_neg_embeds) / behavior_anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)
        embedding_contrast_loss = cal_infonce_loss(behavior_anc_embeds, sem_anc_embeds, sem_user_embeds, self.contrast_temperature) + \
                                  cal_infonce_loss(behavior_pos_embeds, sem_pos_embeds, sem_pos_embeds, self.contrast_temperature) + \
                                  cal_infonce_loss(behavior_neg_embeds, sem_neg_embeds, sem_neg_embeds, self.contrast_temperature)
        embedding_contrast_loss /= behavior_anc_embeds.shape[0]
        embedding_contrast_loss *= self.emb_cl_weight

        
        loss = bpr_loss + reg_loss + sup_cl_loss + embedding_contrast_loss + full_sup_cl_loss
        losses = {
            'bpr_loss': bpr_loss, 
            'reg_loss': reg_loss, 
            'sup_cl_loss': sup_cl_loss,
            'embedding_contrast_loss': embedding_contrast_loss,
            'full_sup_cl_loss': full_sup_cl_loss
        }
        return loss, losses

    def full_predict(self, batch_data):
        orig_user_embeds, orig_item_embeds, _, _ = self.forward(self.adj, 1.0, None)
        
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = orig_user_embeds[pck_users]
        full_preds = pck_user_embeds @ orig_item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
