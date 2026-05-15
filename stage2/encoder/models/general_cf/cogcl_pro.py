
import copy
import random
import time
from collections import defaultdict

import numpy as np
import torch as t
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
from config.configurator import configs
from torch_geometric.nn import MessagePassing
from sklearn.cluster import KMeans
from models.loss_utils import reg_params
import os
from models.loss_utils import reg_params, cal_sup_cl_loss, cal_infonce_loss, make_torch_adj, load_one_mat

from torch_geometric.nn.conv.gcn_conv import gcn_norm
init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class COGCL_pro(BaseModel):
    def __init__(self, data_handler):
        super(COGCL_pro, self).__init__(data_handler)

        self.n_users = self.user_num
        self.n_items = self.item_num

        R = data_handler._create_sparse_interaction_matrix().tocoo()

        self._user = torch.from_numpy(R.row).long()
        self._item = torch.from_numpy(R.col).long()

        row = self._user                                 
        col = self._item + self.n_users                   
        edge_index_ui = torch.stack([row, col], dim=0)    
        edge_index_iu = torch.stack([col, row], dim=0)   
        edge_index = torch.cat([edge_index_ui, edge_index_iu], dim=1)
        edge_index = edge_index.to().cuda()

        self.adj = data_handler.torch_adj

        edge_weight = torch.ones(edge_index.size(1)).cuda()
        num_nodes = self.n_users + self.n_items

        from torch_geometric.nn.conv.gcn_conv import gcn_norm
        self.edge_index, self.edge_weight = gcn_norm(
            edge_index, edge_weight, num_nodes, add_self_loops=False
        )

        self.dataset = data_handler          
        if not hasattr(self.dataset, 'code_similar_users'):
            self.dataset.code_similar_users = [list() for _ in range(self.n_users)]
        if not hasattr(self.dataset, 'code_similar_items'):
            self.dataset.code_similar_items = [list() for _ in range(self.n_items)]

        if self.hyper_config.get('sim_cl_weight', 0) > 0 and hasattr(self.dataset, 'get_same_targe_users_items'):
            self.dataset.get_same_targe_users_items()

        self.USER_ID      = configs['train'].get('user_field', 'user')         
        self.ITEM_ID      = configs['train'].get('item_field', 'item')         
        self.NEG_ITEM_ID  = configs['train'].get('neg_item_field', 'neg_item')

        if self.hyper_config['sim_cl_weight'] > 0:
            self.dataset.get_same_targe_users_items()
            
        setattr(self.dataset, 'code_similar_users', [[]] * self.n_users)
        setattr(self.dataset, 'code_similar_items', [[]] * self.n_items)


        self.AUG_USER_ID  = "aug_" + self.USER_ID
        self.AUG_ITEM_ID = "aug_" + self.ITEM_ID



        self.layer_num = configs['model']['layer_num']
        self.embedding_size = configs['model']['embedding_size']

        self.reg_weight = float(self.hyper_config.get('reg_weight', 0.0))

        self.vq_loss_weight = float(self.hyper_config.get('vq_loss_weight', 1.0))
        self.cl_weight      = float(self.hyper_config.get('cl_weight', 0.1))
        self.sim_cl_weight  = float(self.hyper_config.get('sim_cl_weight', 0.0))

        self.user_code_num = self.hyper_config['user_code_num']
        self.item_code_num = self.hyper_config['item_code_num']
        self.user_code_size = self.hyper_config['user_code_size']
        self.item_code_size = self.hyper_config['item_code_size']
        self.n_user_codes = self.user_code_num * self.user_code_size
        self.n_item_codes = self.item_code_num * self.item_code_size

        self.code_dist = self.hyper_config['code_dist']
        self.code_dist_tau = self.hyper_config['code_dist_tau']
        self.code_batch_size = self.hyper_config['code_batch_size']

        self.vq_type = self.hyper_config['vq_type']
        self.vq_ema = self.hyper_config['vq_ema']


        self.require_pow = self.hyper_config['require_pow']


        self.cl_tau = self.hyper_config['cl_tau']

        self.graph_replace_p = self.hyper_config['graph_replace_p']
        self.graph_add_p = self.hyper_config['graph_add_p']

        self.drop_p = self.hyper_config['drop_p']
        self.drop_fwd = self.hyper_config['drop_fwd']

        self.data_aug_delay = self.hyper_config['data_aug_delay']
        self.epoch_num = 0

        self.semantic_topk = int(self.hyper_config.get('semantic_topk', 10))

        self.sup_cl_weight = float(self.hyper_config.get('sup_cl_weight', 0.1))

        self.cv_sup_cl_weight = float(self.hyper_config.get('cv_sup_cl_weight', 0.1))

        self.emb_cl_weight = float(self.hyper_config.get('emb_cl_weight', 0.0))


        self.contrast_temperature = float(self.hyper_config.get('contrast_temperature', 0.1))
        self.sup_contrast_temperature = float(self.hyper_config.get('sup_contrast_temperature', 0.25))


        if self.vq_type.lower() == 'rq':
            self.user_vq = ResidualVectorQuantizer(codebook_num=self.user_code_num, codebook_size=self.user_code_size, 
                                                   codebook_dim=self.embedding_size, dist=self.code_dist, tau=self.code_dist_tau, vq_ema=self.vq_ema)
            self.item_vq = ResidualVectorQuantizer(codebook_num=self.item_code_num, codebook_size=self.item_code_size,
                                                    codebook_dim=self.embedding_size, dist=self.code_dist, tau=self.code_dist_tau, vq_ema=self.vq_ema)
        elif self.vq_type.lower() == 'pq':
            assert self.embedding_size % self.user_code_num == 0
            assert self.embedding_size % self.item_code_num == 0
            self.user_vq = ProductVectorQuantizer(codebook_num=self.user_code_num, codebook_size=self.user_code_size,
                                                  codebook_dim=self.embedding_size//self.user_code_num, dist=self.code_dist, tau=self.code_dist_tau, vq_ema=self.vq_ema)
            self.item_vq = ProductVectorQuantizer(codebook_num=self.item_code_num, codebook_size=self.item_code_size,
                                                  codebook_dim=self.embedding_size//self.item_code_num, dist=self.code_dist, tau=self.code_dist_tau, vq_ema=self.vq_ema)
        else:
            raise NotImplementedError


        init = torch.nn.init.xavier_uniform_

        self.user_embedding = nn.Parameter(init(t.empty(self.n_users,        self.embedding_size)))
        self.item_embedding = nn.Parameter(init(t.empty(self.n_items,        self.embedding_size)))
        self.user_code_embedding = nn.Parameter(init(t.empty(self.n_user_codes, self.embedding_size)))
        self.item_code_embedding = nn.Parameter(init(t.empty(self.n_item_codes, self.embedding_size)))


        self.gcn_conv = LightGCNConv(dim=self.embedding_size)

        self.dropout = torch.nn.Dropout(self.drop_p)


        self.aug_edge_index_1 = None
        self.aug_edge_weight_1 = None
        self.aug_edge_index_2 = None
        self.aug_edge_weight_2 = None

        self.non_graph_aug_edge_index, self.non_graph_aug_edge_weight = None, None
        self.non_graph_aug_edge_index, self.non_graph_aug_edge_weight = self.inter_graph_aug()


        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        self.other_parameter_name = ['restore_user_e', 'restore_item_e']


        self.debug_log = bool(self.hyper_config.get('debug_log', False))
        self.max_debug_steps = int(self.hyper_config.get('max_debug_steps', 2))
        self._debug_count = 0
        target_device = configs['device']
        self.usrprf_embeds = t.tensor(configs['usrprf_embeds']).float().cuda()
        self.itmprf_embeds = t.tensor(configs['itmprf_embeds']).float().cuda()
        self.register_buffer('semantic_user_embeds', self.usrprf_embeds)
        self.register_buffer('semantic_item_embeds', self.itmprf_embeds)
        self.semantic_embedding_size = self.usrprf_embeds.shape[1]

        self.semantic_mlps = nn.Sequential(
            nn.Linear(self.semantic_embedding_size, (self.embedding_size + self.semantic_embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.embedding_size + self.semantic_embedding_size) // 2, self.embedding_size)
        )

        print(f"✓ Semantic embeddings and model embeddings are both on device: {target_device}")

        self._try_load_semantic_graph_mat()


        target_device = configs['device']
        if self.semantic_adj.device != target_device:
            self.semantic_adj = self.semantic_adj.to(target_device)
            print(f"✓ Semantic graph has been moved to the target device: {target_device}")

        self.interaction_user_indices = None
        self.interaction_item_indices = None

        self._init_weight()

        def _init_weight(self):

            for m in self.semantic_mlps:
                if isinstance(m, nn.Linear):
                    init(m.weight)


        def _try_load_semantic_graph_mat(self):
            """Try to load the precomputed semantic interaction graph using an adjacency matrix."""
            try:
                # Get the dataset name

                precomputed_path = configs['sem_graph_path']

                if os.path.exists(precomputed_path):
                    trn_mat = load_one_mat(precomputed_path)
                    self.semantic_adj = make_torch_adj(trn_mat)
                    print(f"✓ Successfully loaded the precomputed semantic graph. Shape: {self.semantic_adj.size()}")
                    print(f"  Number of non-zero elements: {self.semantic_adj._nnz()}")
                    print(f"  Sparsity: {1.0 - (self.semantic_adj._nnz() / (self.semantic_adj.size(0) * self.semantic_adj.size(1))):.4f}")
                    print(f"  Current device: {self.semantic_adj.device}")
                    print(f"  Target device: {configs['device']}")

                    # Ensure that the semantic graph is on the correct device
                    if self.semantic_adj.device != configs['device']:
                        if self._debug_count < 3:  # Print the warning only for the first few times
                            print(f"⚠️ Device mismatch detected. Moving semantic graph from {self.semantic_adj.device} to {configs['device']}")
                        self.semantic_adj = self.semantic_adj.to(configs['device'])

            except Exception as e:
                print(f"⚠️ Failed to load the precomputed semantic graph: {e}")
                print("  Continue using the dynamically constructed semantic graph")

    def make_all_one_adj(self):
        idxs = self.adj._indices()
        vals = t.ones_like(self.adj._values())
        shape = self.adj.shape
        return t.sparse.FloatTensor(idxs, vals, shape).cuda()



    def train(self, mode: bool = True):
        r"""Override train method of base class. The subgraph is reconstructed each time it is called.

        """
        
        if mode and self.epoch_num >= self.data_aug_delay:
            if self.epoch_num == self.data_aug_delay:
                print("Start Data Augmentation")
            self.graph_augment()

        if mode:
            self.epoch_num += 1

        T = super().train(mode=mode)

        return T

    def _to_long(self, x):
        import torch, numpy as np
        if isinstance(x, torch.Tensor):
            return x.long().to().cuda()
        if isinstance(x, (list, tuple, np.ndarray)):
            return torch.as_tensor(x, dtype=torch.long).cuda()
        return torch.tensor(x, dtype=torch.long).cuda

    def _unpack_triplet(self, interaction):
        if isinstance(interaction, (tuple, list)):
            if len(interaction) == 3:
                u, p, n = interaction
                return self._to_long(u), self._to_long(p), self._to_long(n)
            elif len(interaction) == 2:
                u, p = interaction
                return self._to_long(u), self._to_long(p), None
            else:
                raise ValueError(f"unexpected tuple/list len={len(interaction)} for interaction")
        if hasattr(interaction, 'keys'):
            u = interaction.get(getattr(self, 'USER_ID', 'user'))
            p = interaction.get(getattr(self, 'ITEM_ID', 'item'))
            n = interaction.get(getattr(self, 'NEG_ITEM_ID', 'neg_item'), None)
            if u is None or p is None:
                raise KeyError("interaction missing user/item fields")
            return self._to_long(u), self._to_long(p), (None if n is None else self._to_long(n))
        raise TypeError(f"unsupported interaction type: {type(interaction)}")

    def _get_aug_from_batch(self, batch, user, pos_item):

        if hasattr(batch, 'keys'):
            if self.AUG_USER_ID in batch and self.AUG_ITEM_ID in batch:
                au = batch[self.AUG_USER_ID]
                ai = batch[self.AUG_ITEM_ID]
                return self._to_long(au), self._to_long(ai)


        import random, torch
        u_list = user.tolist()
        i_list = pos_item.tolist()


        cu = getattr(self.dataset, 'code_similar_users', None)
        ci = getattr(self.dataset, 'code_similar_items', None)

        aug_u = []
        for u in u_list:
            cand = (cu[u] if cu and len(cu[u]) > 0 else [u])
            aug_u.append(random.choice(cand))
        aug_i = []
        for i in i_list:
            cand = (ci[i] if ci and len(ci[i]) > 0 else [i])
            aug_i.append(random.choice(cand))

        aug_users = torch.as_tensor(aug_u, dtype=torch.long).cuda()
        aug_items = torch.as_tensor(aug_i, dtype=torch.long).cuda()
        return aug_users, aug_items

    def BPRLoss(self, pos_scores, neg_scores):
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))

    def EmbReg(self):
        return (self.user_embedding.pow(2).sum()
                + self.item_embedding.pow(2).sum()
                + self.user_code_embedding.pow(2).sum()
                + self.item_code_embedding.pow(2).sum())


    def graph_augment(self):

        all_user_codes, all_item_codes = self.get_all_codes()

        # 0 for replace, 1 for add
        aug_types = random.choices([0, 1], k=2)
        if self.sim_cl_weight > 0:
            similar_users, similar_items = self.get_share_codes_info(all_user_codes, all_item_codes)
            setattr(self.dataset, 'code_similar_users', similar_users)
            setattr(self.dataset, 'code_similar_items', similar_items)

        all_user_codes = all_user_codes.detach().cpu().numpy()
        all_item_codes = all_item_codes.detach().cpu().numpy()

        self.aug_edge_index_1, self.aug_edge_weight_1 = self.inter_graph_aug(all_user_codes, all_item_codes, aug_types[0])
        self.aug_edge_index_2, self.aug_edge_weight_2 = self.inter_graph_aug(all_user_codes, all_item_codes, aug_types[1])


    @torch.no_grad()
    def get_all_codes(self):

        self.user_vq.eval()
        self.item_vq.eval()

        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        user_all_embeddings = self.restore_user_e
        item_all_embeddings = self.restore_item_e

        start, batch_size = 0, self.code_batch_size
        all_user_codes = []
        while start < self.n_users:
            batch_user_embs = user_all_embeddings[start: start + batch_size]
            _, codes = self.emb_quantize(batch_user_embs, self.user_vq)
            all_user_codes.append(codes)
            start += batch_size
        all_user_codes = torch.cat(all_user_codes, dim=0)
        # user_offset = torch.arange(self.user_code_num, dtype=torch.long, device=self.device) * self.user_code_size
        user_offset = torch.arange(self.user_code_num, dtype=torch.long).cuda() * self.user_code_size
        all_user_codes = all_user_codes + user_offset.unsqueeze(0)

        start, batch_size = 0, self.code_batch_size
        all_item_codes = []
        while start < self.n_items:
            batch_item_embs = item_all_embeddings[start: start + batch_size]
            _, codes = self.emb_quantize(batch_item_embs, self.item_vq)
            all_item_codes.append(codes)
            start += batch_size
        all_item_codes = torch.cat(all_item_codes, dim=0)
        # item_offset = torch.arange(self.item_code_num, dtype=torch.long, device=self.device) * self.item_code_size
        item_offset = torch.arange(self.item_code_num, dtype=torch.long).cuda() * self.item_code_size
        all_item_codes = all_item_codes + item_offset.unsqueeze(0)

        return all_user_codes, all_item_codes

    def get_share_codes_info(self, all_user_codes, all_item_codes):

        start, batch_size = 0, self.code_batch_size
        similar_users = [[]] * self.n_users
        while start < self.n_users:
            batch_ucodes = all_user_codes[start: start + batch_size]
            batch_sim = []
            k = 0
            while k < self.n_users:
                block = all_user_codes[k: k + batch_size]
                # B, 4, 1 == 1, 4, B   ->   B, 4, B   ->  B, B
                sim = (batch_ucodes.unsqueeze(-1) == block.T.unsqueeze(0)).to(torch.int).sum(dim=1)
                batch_sim.append(sim)
                k += batch_size
            # several B, B   ->    B, N
            batch_sim = torch.cat(batch_sim, dim=1)
            batch_sim[:, 0] = 0

            for i, sim in enumerate(batch_sim):
                uid = start + i
                sim_index = torch.where(sim >= self.user_code_num - 1)[0]
                similar_users[uid] = sim_index.cpu().numpy()

            start += batch_size



        start, batch_size = 0, self.code_batch_size
        similar_items = [[]] * self.n_items
        while start < self.n_items:
            batch_icodes = all_item_codes[start: start + batch_size]
            batch_sim = []
            k = 0
            while k < self.n_items:
                block = all_item_codes[k: k + batch_size]
                sim = (batch_icodes.unsqueeze(-1) == block.T.unsqueeze(0)).to(torch.int).sum(dim=1)
                batch_sim.append(sim)
                k += batch_size
            batch_sim = torch.cat(batch_sim, dim=1)
            batch_sim[:, 0] = 0
            for i, sim in enumerate(batch_sim):
                iid = start + i
                sim_index = torch.where(sim >= self.item_code_num - 1)[0]
                similar_items[iid] = sim_index.cpu().numpy()

            start += batch_size

        return similar_users, similar_items

    def inter_graph_aug(self, all_user_codes=None, all_item_codes=None, aug_type=2):



        if all_user_codes is None or all_item_codes is None:
            assert aug_type==2

        if aug_type == 2 and self.non_graph_aug_edge_index is not None and self.non_graph_aug_edge_weight is not None:
            return self.non_graph_aug_edge_index, self.non_graph_aug_edge_weight

        row = self._user
        col = self._item

        all_idx = np.arange(len(row))
        if aug_type == 0: # replace
            aug_num = int(len(all_idx) * self.graph_replace_p)
            user_aug_idx = np.random.choice(len(row), aug_num, replace=False)
            item_aug_idx = np.random.choice(len(col), aug_num, replace=False)
            keep_idx = list( set(all_idx) - set(user_aug_idx) - set(item_aug_idx) )

        elif aug_type==1: # add
            keep_idx = all_idx
            aug_num = int(len(all_idx) * self.graph_add_p)
            user_aug_idx = np.random.choice(len(row), aug_num, replace=False)
            item_aug_idx = np.random.choice(len(col), aug_num, replace=False)
        else: # non graph aug
            keep_idx = all_idx
            user_aug_idx = []
            item_aug_idx = []

        keep_row = row[keep_idx]
        keep_col = col[keep_idx] + self.n_users + self.n_user_codes


        if aug_type == 2:
            aug_inter_row = []
            aug_inter_col = []
        else:
            aug_inter_row = []
            aug_inter_col = []

            aug_row = row[user_aug_idx]
            aug_col = col[user_aug_idx]
            for user, item in zip(aug_row, aug_col):
                item_codes = all_item_codes[item] + self.n_items + self.n_users + self.n_user_codes
                item_codes = item_codes.tolist()
                aug_inter_row.extend([user]* len(item_codes))
                aug_inter_col.extend(item_codes)

            aug_row = row[item_aug_idx]
            aug_col = col[item_aug_idx]
            for user, item in zip(aug_row, aug_col):
                user_codes = all_user_codes[user] + self.n_users
                item = item + self.n_users + self.n_user_codes
                user_codes = user_codes.tolist()
                aug_inter_row.extend(user_codes)
                aug_inter_col.extend([item] * len(user_codes))


        all_row = torch.tensor(keep_row.numpy().tolist() + aug_inter_row).to(keep_row.dtype)
        all_col = torch.tensor(keep_col.numpy().tolist() + aug_inter_col).to(keep_col.dtype)

        edge_index1 = torch.stack([all_row, all_col])
        edge_index2 = torch.stack([all_col, all_row])
        aug_edge_index = torch.cat([edge_index1, edge_index2], dim=1)
        aug_edge_weight = torch.ones(aug_edge_index.size(1))

        num_nodes = self.n_users + self.n_user_codes + self.n_items + self.n_item_codes


        aug_edge_index, aug_edge_weight = gcn_norm(aug_edge_index.to().cuda(),
                                            aug_edge_weight.to().cuda(),
                                            num_nodes, add_self_loops=False)


        return aug_edge_index, aug_edge_weight



    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding          
        item_embeddings = self.item_embedding
        return torch.cat([user_embeddings, item_embeddings], dim=0)


    def get_w_codes_embeddings(self):
        user_embeddings = self.user_embedding
        item_embeddings = self.item_embedding
        user_code_embeddings = self.user_code_embedding
        item_code_embeddings = self.item_code_embedding
        return torch.cat([user_embeddings, user_code_embeddings,
                        item_embeddings, item_code_embeddings], dim=0)

                        
    def forward(self, drop=False):
        embeddings = self.get_ego_embeddings()
        embeddings_list = []

        for layer_idx in range(self.layer_num):
            if drop:
                embeddings = self.dropout(embeddings)
            embeddings = self.gcn_conv(embeddings, self.edge_index, self.edge_weight)
            embeddings_list.append(embeddings)

        all_embeddings = torch.stack(embeddings_list, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            all_embeddings, [self.n_users, self.n_items]
        )

        return user_all_embeddings, item_all_embeddings

    def forward1(self, adj=None, keep_rate=1.0):
        if adj is None:
            adj = self.adj
            semantic_adj = self.semantic_adj

        seman_embeds = t.cat([self.semantic_user_embeds, self.semantic_item_embeds], dim=0)
        sem_embeds_list = [seman_embeds]

        adj = self.adj
        semantic_adj = self.semantic_adj

        for layer_idx in range(self.layer_num):
            sem_embeds = t.spmm(semantic_adj, sem_embeds_list[-1])
            sem_embeds_list.append(sem_embeds)

        final_sem_embeds = sum(sem_embeds_list)
        sem_user_embeds = final_sem_embeds[:self.user_num]
        sem_item_embeds = final_sem_embeds[self.user_num:]
        return sem_user_embeds, sem_item_embeds
    
    def aug_forward(self, aug_edge_index=None, aug_edge_weight=None):

        if aug_edge_index is None or aug_edge_weight is None:
            aug_edge_index = self.non_graph_aug_edge_index
            aug_edge_weight = self.non_graph_aug_edge_weight

        embeddings = self.get_w_codes_embeddings()
        embeddings_list = []

        for layer_idx in range(self.layer_num):
            embeddings = self.dropout(embeddings)
            embeddings = self.gcn_conv(embeddings, aug_edge_index, aug_edge_weight)
            embeddings_list.append(embeddings)

        all_embeddings = torch.stack(embeddings_list, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            all_embeddings, [self.n_users + self.n_user_codes, self.n_items + self.n_item_codes]
        )

        return user_all_embeddings, item_all_embeddings




    def quantize(self, users, items, user_embeddings=None, item_embeddings=None):
        if user_embeddings is None or item_embeddings is None:
            user_embs = self.user_embedding[users] 
            item_embs = self.item_embedding[items]
        else:
            user_embs = user_embeddings[users]
            item_embs = item_embeddings[items]
        user_vq_loss, _ = self.emb_quantize(user_embs, self.user_vq)
        item_vq_loss, _ = self.emb_quantize(item_embs, self.item_vq)
        return user_vq_loss + item_vq_loss
    
    
    def emb_quantize(self, x, vq_layer):

        x = x.detach()

        x_q, mean_com_loss, all_codes = vq_layer(x)

        if self.code_dist.lower() == 'l2':
            recon_loss = F.mse_loss(x_q, x)
        elif self.code_dist.lower() == 'cos':
            recon_scores = torch.matmul(F.normalize(x_q, dim=-1), F.normalize(x, dim=-1).t()) / self.code_dist_tau
            recon_labels = torch.arange(x.size(0), dtype=torch.long).to(x.device)
            recon_loss = F.cross_entropy(recon_scores, recon_labels)
        else:
            raise NotImplementedError

        loss = recon_loss + mean_com_loss

        return loss, all_codes


    def calculate_cl_loss(self, x1, x2):
        assert x1.size(0) == x2.size(0)

        x1, x2 = F.normalize(x1, dim=-1), F.normalize(x2, dim=-1)

        sim_12 = torch.mm(x1, x2.T)
        sim_21 = sim_12.T
        sim_11 = torch.mm(x1, x1.T)
        sim_22 = torch.mm(x2, x2.T)

        sim_12 = torch.exp(sim_12 / self.cl_tau)
        sim_21 = torch.exp(sim_21 / self.cl_tau)
        sim_11 = torch.exp(sim_11 / self.cl_tau)
        sim_22 = torch.exp(sim_22 / self.cl_tau)

        loss_12 = -torch.log(
            sim_12.diag()
            / (sim_12.sum(1) + sim_11.sum(1) - sim_11.diag())
        )
        loss_21 = -torch.log(
            sim_21.diag()
            / (sim_21.sum(1) + sim_22.sum(1) - sim_22.diag())
        )

        loss = 0.5 * (loss_12 + loss_21).mean()

        return loss

    def cal_loss(self, batch_data):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None


        user, pos_item, neg_item = self._unpack_triplet(batch_data)

        orig_user_embeds, orig_item_embeds = self.forward(drop=self.drop_fwd)
        sem_user_embeds, sem_item_embeds = self.forward1(self.adj, 1.0)
        sem_user_embeds = self.semantic_mlps(sem_user_embeds)
        sem_item_embeds = self.semantic_mlps(sem_item_embeds)
        orig_user_embeds, orig_item_embeds = F.normalize(orig_user_embeds, dim=-1), F.normalize(orig_item_embeds, dim=-1)
        sem_user_embeds, sem_item_embeds = F.normalize(sem_user_embeds, dim=-1), F.normalize(sem_item_embeds, dim=-1)
        ancs, poss, negs = self._unpack_triplet(batch_data)
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
        embedding_contrast_loss = cal_infonce_loss(behavior_anc_embeds, sem_anc_embeds, sem_user_embeds, self.contrast_temperature) + \
                    cal_infonce_loss(behavior_pos_embeds, sem_pos_embeds, sem_pos_embeds, self.contrast_temperature) + \
                    cal_infonce_loss(behavior_neg_embeds, sem_neg_embeds, sem_neg_embeds, self.contrast_temperature)
        
        embedding_contrast_loss /= behavior_anc_embeds.shape[0]
        embedding_contrast_loss *= self.emb_cl_weight

        u_embeddings = orig_user_embeds[user]
        pos_embeddings = orig_item_embeds[pos_item]
        neg_embeddings = orig_item_embeds[neg_item]

        pos_id_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_id_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        mf_loss = self.BPRLoss(pos_id_scores, neg_id_scores)

        reg_loss = self.EmbReg()

        user = torch.unique(user, sorted=True)
        pos_item = torch.unique(pos_item, sorted=True)

        vq_loss = self.quantize(user, pos_item, orig_user_embeds, orig_item_embeds)

        user_aug1_embeddings, item_aug1_embeddings = self.aug_forward(self.aug_edge_index_1,self.aug_edge_weight_1)
        user_aug2_embeddings, item_aug2_embeddings = self.aug_forward(self.aug_edge_index_2,self.aug_edge_weight_2)
        u_x1, u_x2 = user_aug1_embeddings[user], user_aug2_embeddings[user]
        i_x1, i_x2 = item_aug1_embeddings[pos_item], item_aug2_embeddings[pos_item]

        cl_loss = self.calculate_cl_loss(u_x1, u_x2) + self.calculate_cl_loss(i_x1, i_x2)

        if self.sim_cl_weight > 0:

            aug_users, aug_items = self._get_aug_from_batch(batch_data, user, pos_item)
            u_x3 = orig_user_embeds[aug_users]
            i_x3 = orig_item_embeds[aug_items]

            sim_cl_loss_1 = self.calculate_cl_loss(u_x1, u_x3) + self.calculate_cl_loss(i_x1, i_x3)
            sim_cl_loss_2 = self.calculate_cl_loss(u_x2, u_x3) + self.calculate_cl_loss(i_x2, i_x3)

            sim_cl_loss = 0.5 * (sim_cl_loss_1 + sim_cl_loss_2)
        else:
            sim_cl_loss = torch.zeros(1).to().cuda()

        reg_loss = self.reg_weight * reg_loss
        vq_loss = self.vq_loss_weight * vq_loss
        sim_cl_loss = self.sim_cl_weight * sim_cl_loss
        cl_loss = self.cl_weight * cl_loss

        loss = mf_loss + reg_loss + vq_loss + sim_cl_loss + cl_loss + embedding_contrast_loss + full_sup_cl_loss + sup_cl_loss

        losses = {'rec_loss': mf_loss, 'reg_loss': reg_loss, 'vq_loss': vq_loss, 'sim_cl_loss': sim_cl_loss, 'cl_loss': cl_loss, 'embedding_contrast_loss': embedding_contrast_loss, 'full_sup_cl_loss': full_sup_cl_loss, 'sup_cl_loss': sup_cl_loss}


        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward()
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds



class ResidualVectorQuantizer(nn.Module):

    def __init__(self, codebook_num, codebook_size, codebook_dim, dist ='cos', tau=0.2, vq_ema=False):
        super().__init__()
        self.codebook_num = codebook_num
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.dist = dist
        self.tau = tau

        self.vq_ema = vq_ema

        if self.vq_ema:
            self.vq_layers = torch.nn.ModuleList([
                EMAVectorQuantizer(codebook_size=self.codebook_size, codebook_dim=self.codebook_dim, dist=self.dist,
                                tau=self.tau)
                for _ in range(self.codebook_num)
            ])
        else:
            self.vq_layers = torch.nn.ModuleList([
                VectorQuantizer(codebook_size=self.codebook_size, codebook_dim=self.codebook_dim, dist=self.dist,
                                tau=self.tau)
                for _ in range(self.codebook_num)
            ])


    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.cat(all_codebook, dim=0)

    @torch.no_grad()
    def get_codes(self, x):
        all_codes = []
        residual = x
        for i in range(len(self.vq_layers)):
            x_res, _, codes = self.vq_layers[i](residual)
            residual = residual - x_res
            all_codes.append(codes)

        all_codes = torch.stack(all_codes, dim=-1)

        return all_codes


    def forward(self, x):

        all_com_losses = []
        all_codes = []


        x_q = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res, com_loss, codes = quantizer(residual)
            residual = residual - x_res

            x_q = x_q + x_res

            all_com_losses.append(com_loss)
            all_codes.append(codes)

        mean_com_loss = torch.stack(all_com_losses).mean()
        all_codes = torch.stack(all_codes, dim=-1)

        return x_q, mean_com_loss, all_codes


class ProductVectorQuantizer(nn.Module):

    def __init__(self, codebook_num, codebook_size, codebook_dim, dist ='cos', tau=0.2, vq_ema=False):
        super().__init__()
        self.codebook_num = codebook_num
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.dist = dist
        self.tau = tau
        self.vq_ema = vq_ema

        if self.vq_ema:
            self.vq_layers = torch.nn.ModuleList([
                EMAVectorQuantizer(codebook_size=self.codebook_size, codebook_dim=self.codebook_dim, dist=self.dist,
                                tau=self.tau)
                for _ in range(self.codebook_num)
            ])
        else:
            self.vq_layers = torch.nn.ModuleList([
                VectorQuantizer(codebook_size=self.codebook_size, codebook_dim=self.codebook_dim, dist=self.dist,
                                tau=self.tau)
                for _ in range(self.codebook_num)
            ])

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.cat(all_codebook, dim=0)

    @torch.no_grad()
    def get_codes(self, x):
        all_codes = []
        x_chunk = torch.chunk(x, self.codebook_num, dim=-1)
        for idx, quantizer in enumerate(self.vq_layers):
            _, _, codes = quantizer(x_chunk[idx])
            all_codes.append(codes)

        all_codes = torch.stack(all_codes, dim=-1)

        return all_codes

    def forward(self, x):

        all_com_losses = []
        all_codes = []
        all_x_q = []

        x_chunk = torch.chunk(x, self.codebook_num, dim=-1)
        for idx, quantizer in enumerate(self.vq_layers):
            x_q, com_loss, codes = quantizer(x_chunk[idx])

            all_x_q.append(x_q)
            all_com_losses.append(com_loss)
            all_codes.append(codes)

        all_x_q = torch.cat(all_x_q, dim=-1)
        mean_com_loss = torch.stack(all_com_losses).mean()
        all_codes = torch.stack(all_codes, dim=-1)

        return all_x_q, mean_com_loss, all_codes




def kmeans(
        samples,
        num_clusters,
):

    B, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device
    x = samples.cpu().detach().numpy()

    cluster = KMeans(n_clusters=num_clusters).fit(x)

    centers = cluster.cluster_centers_
    tensor_centers = torch.from_numpy(centers).to(device)

    return tensor_centers


def replace_nan(x):

    return torch.where(torch.isnan(x), torch.zeros_like(x), x)

@torch.no_grad()
def sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    Q = torch.exp(-distances / epsilon).t()

    B = Q.shape[1]
    K = Q.shape[0]  # how many centroids per block (usually set to 256)

    # make the matrix sums to 1
    sum_Q = torch.clamp(torch.sum(Q), min=1e-5)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.t()

class VectorQuantizer(nn.Module):

    def __init__(self, codebook_size, codebook_dim, dist ='cos', tau=0.2, sk_epsilon=0.005, sk_iters=3):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.dist = dist
        self.tau = tau
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        self.embedding = nn.Embedding(self.codebook_size, self.codebook_dim)
        self.initted = False
        self.embedding.weight.data.zero_()


    def get_codebook(self):
        return self.embedding.weight

    def get_quantize_emb(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):

        centers = kmeans(
            data,
            self.codebook_size,
        )

        self.embedding.weight.data.copy_(centers)
        self.initted = True


    @staticmethod
    def center_distance_for_constraint(distances):

        distances = replace_nan(distances)
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        if torch.isnan(amplitude):
            amplitude = 1e-5

        amplitude = torch.clamp(amplitude, min=1e-5)

        if amplitude <= 0:
            amplitude = 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x):
        # Flatten input
        latent = x.view(-1, self.codebook_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        # Calculate the distances between latent and Embedded weights
        if self.dist.lower() == 'l2':
            d = torch.sum(latent**2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
                2 * torch.matmul(latent, self.embedding.weight.t())
        elif self.dist.lower() == 'cos':
            cos_sim = torch.matmul(F.normalize(latent, dim=-1), F.normalize(self.embedding.weight, dim=-1).t())
            d = (1 - cos_sim) / self.tau
        else:
            raise NotImplementedError

        if torch.isnan(d).any() or torch.isinf(d).any():
            print(f"Distance have nan/inf values.")


        if self.sk_epsilon > 0 and self.training:

            d = self.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)
        else:
            indices = torch.argmin(d, dim=-1)

        x_q = self.embedding(indices).view(x.shape)



        # compute loss for embedding
        if self.dist.lower() == 'l2':
            codebook_loss = F.mse_loss(x_q, x.detach())
            # commitment_loss = F.mse_loss(x_q.detach(), x)
            # loss = codebook_loss + self.beta * commitment_loss
            loss = codebook_loss
        elif self.dist.lower() == 'cos':
            loss = F.cross_entropy(cos_sim / self.tau, indices.detach())
        else:
            raise NotImplementedError


        x_q = x + (x_q - x).detach()


        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices





class EMAVectorQuantizer(nn.Module):

    def __init__(self, codebook_size, codebook_dim, dist='cos', tau=0.2,
                 sk_epsilon=0.005, sk_iters=3, decay=0.99,  eps=1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        self.decay = decay
        self.eps = eps
        self.dist = dist
        self.tau = tau

        embedding = torch.randn(self.codebook_size, self.codebook_dim)

        self.register_buffer('embedding', embedding)
        self.register_buffer('embedding_avg', embedding.clone())
        self.register_buffer('cluster_size', torch.ones(codebook_size))

        self.initted = False


    def get_codebook(self):
        return self.embedding

    def get_quantize_emb(self, indices, shape=None):
        # get quantized latent vectors
        z_q = F.embedding(indices, self.embedding)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def _tile(self, x):
        n, d = x.shape
        if n < self.codebook_size:
            n_repeats = (self.codebook_size + n - 1) // n
            std = 0.01 / np.sqrt(d)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x


    def init_emb(self, data):

        centers = kmeans(
            data,
            self.codebook_size,
        )

        self.embedding.data.copy_(centers)
        self.embedding_avg.data.copy_(centers)
        self.initted = True


    def moving_average(self, moving_avg, new, decay):
        moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))

    def laplace_smoothing(self, x, n_categories, eps=1e-5):
        return (x + eps) / (x.sum() + n_categories * eps)

    @staticmethod
    def center_distance_for_constraint(distances):

        distances = replace_nan(distances)
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        if torch.isnan(amplitude):
            amplitude = 1e-5

        amplitude = torch.clamp(amplitude, min=1e-5)

        if amplitude <= 0:
            amplitude = 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances


    def forward(self, x):
        # Flatten input
        latent = x.view(-1, self.codebook_dim)


        if not self.initted and self.training:
            self.init_emb(latent)

        # Calculate the distances between latent and Embedded weights
        if self.dist.lower() == 'l2':
            d = torch.sum(latent**2, dim=1, keepdim=True) + \
                torch.sum(self.embedding**2, dim=1, keepdim=True).t()- \
                2 * torch.matmul(latent, self.embedding.t())
        elif self.dist.lower() == 'cos':
            cos_sim = torch.matmul(F.normalize(latent, dim=-1), F.normalize(self.embedding, dim=-1).t())
            d = (1 - cos_sim) / self.tau
        else:
            raise NotImplementedError


        if torch.isnan(d).any() or torch.isinf(d).any():
            print(f"Distance have nan/inf values.")


        if self.sk_epsilon > 0 and self.training:
            d_norm = self.center_distance_for_constraint(d)
            d_norm = d_norm.double()
            Q = sinkhorn_algorithm(d_norm, self.sk_epsilon, self.sk_iters)

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)
        else:
            indices = torch.argmin(d, dim=-1)

        x_q = F.embedding(indices, self.embedding).view(x.shape)


        loss = torch.zeros(1).to(x.device)

        if self.training:
            embedding_onehot = F.one_hot(indices, self.codebook_size).type(latent.dtype)
            embedding_sum = embedding_onehot.t() @ latent
            self.moving_average(self.cluster_size, embedding_onehot.sum(0), self.decay)
            self.moving_average(self.embedding_avg, embedding_sum, self.decay)
            n = self.cluster_size.sum()
            weights = self.laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * n
            embedding_normalized = self.embedding_avg / weights.unsqueeze(1)
            self.embedding.data.copy_(embedding_normalized)

        # preserve gradients
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices



class LightGCNConv(MessagePassing):
    def __init__(self, dim):
        super(LightGCNConv, self).__init__(aggr='add')
        self.dim = dim

    def forward(self, x, edge_index, edge_weight):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        # return matmul(adj_t, x, reduce=self.aggr)
        return torch.matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.dim)


class BipartiteGCNConv(MessagePassing):
    def __init__(self, dim):
        super(BipartiteGCNConv, self).__init__(aggr='add')
        self.dim = dim

    def forward(self, x, edge_index, edge_weight, size):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.dim)


class BiGNNConv(MessagePassing):
    r"""Propagate a layer of Bi-interaction GNN

    .. math::
        output = (L+I)EW_1 + LE \otimes EW_2
    """

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.in_channels, self.out_channels = in_channels, out_channels
        self.lin1 = torch.nn.Linear(in_features=in_channels, out_features=out_channels)
        self.lin2 = torch.nn.Linear(in_features=in_channels, out_features=out_channels)

    def forward(self, x, edge_index, edge_weight):
        x_prop = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        x_trans = self.lin1(x_prop + x)
        x_inter = self.lin2(torch.mul(x_prop, x))
        return x_trans + x_inter

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        # return matmul(adj_t, x, reduce=self.aggr)
        return torch.matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class SRGNNConv(MessagePassing):
    def __init__(self, dim):
        # mean aggregation to incorporate weight naturally
        super(SRGNNConv, self).__init__(aggr='mean')

        self.lin = torch.nn.Linear(dim, dim)

    def forward(self, x, edge_index):
        x = self.lin(x)
        return self.propagate(edge_index, x=x)


class SRGNNCell(nn.Module):
    def __init__(self, dim):
        super(SRGNNCell, self).__init__()

        self.dim = dim
        self.incomming_conv = SRGNNConv(dim)
        self.outcomming_conv = SRGNNConv(dim)

        self.lin_ih = nn.Linear(2 * dim, 3 * dim)
        self.lin_hh = nn.Linear(dim, 3 * dim)

        self._reset_parameters()

    def forward(self, hidden, edge_index):
        input_in = self.incomming_conv(hidden, edge_index)
        reversed_edge_index = torch.flip(edge_index, dims=[0])
        input_out = self.outcomming_conv(hidden, reversed_edge_index)
        inputs = torch.cat([input_in, input_out], dim=-1)

        gi = self.lin_ih(inputs)
        gh = self.lin_hh(hidden)
        i_r, i_i, i_n = gi.chunk(3, -1)
        h_r, h_i, h_n = gh.chunk(3, -1)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = (1 - input_gate) * hidden + input_gate * new_gate
        return hy

    def _reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)