import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.loss_utils import reg_params, cal_sup_cl_loss, cal_infonce_loss, make_torch_adj, load_one_mat, cal_bpr_loss
from models.base_model import BaseModel
import os

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class GCNLayer(nn.Module):
    def __init__(self, latdim):
        super(GCNLayer, self).__init__()
        self.W = nn.Parameter(init(t.empty(latdim, latdim)))

    def forward(self, adj, embeds):
        return t.spmm(adj, embeds)

class GCCF_pro(BaseModel):
    def __init__(self, data_handler):
        super(GCCF_pro, self).__init__(data_handler)

        self.adj = data_handler.torch_adj
        
        # hyper-parameter
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = float(self.hyper_config.get('reg_weight', 0.0))
        
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.gcnLayers = nn.Sequential(*[GCNLayer(self.embedding_size) for i in range(self.layer_num)])
        self.is_training = True

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
        self.usrprf_embeds = t.tensor(configs['usrprf_embeds']).float().cuda()
        self.itmprf_embeds = t.tensor(configs['itmprf_embeds']).float().cuda()
        self.register_buffer('semantic_user_embeds', self.usrprf_embeds)
        self.register_buffer('semantic_item_embeds', self.itmprf_embeds)
        self.semantic_embedding_size = self.usrprf_embeds.shape[1]

        self.semantic_mlps = nn.Sequential(
                nn.Linear(self.semantic_embedding_size, (self.embedding_size + self.semantic_embedding_size) // 2),
                nn.LeakyReLU(),
                nn.Linear((self.embedding_size + self.semantic_embedding_size) // 2, self.embedding_size * 3)
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
                precomputed_path = configs['sem_graph_path']

                if os.path.exists(precomputed_path):
                    trn_mat = load_one_mat(precomputed_path)
                    self.semantic_adj = make_torch_adj(trn_mat)
                    print(f"✓ Successfully loaded the precomputed semantic graph. Shape: {self.semantic_adj.size()}")
                    print(f"  Number of non-zero elements: {self.semantic_adj._nnz()}")
                    print(f"  Sparsity: {1.0 - (self.semantic_adj._nnz() / (self.semantic_adj.size(0) * self.semantic_adj.size(1))):.4f}")
                    print(f"  Current device: {self.semantic_adj.device}")
                    print(f"  Target device: {configs['device']}")

                    if self.semantic_adj.device != configs['device']:
                        if self._debug_count < 3:
                            print(f"⚠️ Device mismatch detected. Moving semantic graph from {self.semantic_adj.device} to {configs['device']}")
                        self.semantic_adj = self.semantic_adj.to(configs['device'])

            except Exception as e:
                print(f"⚠️ Failed to load the precomputed semantic graph: {e}")
                print("  Continue using the dynamically constructed semantic graph")
    
    def forward(self, adj=None):
        if adj is None:
            adj = self.adj
        if not self.is_training:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:], None
        embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        embeds_list = [embeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embeds_list[-1])
            embeds_list.append(embeds)
        embeds = t.concat(embeds_list, dim=-1)
        self.final_embeds = embeds
        return embeds[:self.user_num], embeds[self.user_num:], embeds_list[-1]

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
    
    def cal_loss(self, batch_data):
        self.is_training = True
        orig_user_embeds, orig_item_embeds, _ = self.forward(self.adj)
        ancs, poss, negs = batch_data
        sem_user_embeds, sem_item_embeds = self.forward1(self.adj, 1.0)
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

        embedding_contrast_loss = cal_infonce_loss(behavior_anc_embeds, sem_anc_embeds, sem_user_embeds, self.contrast_temperature) + \
                    cal_infonce_loss(behavior_pos_embeds, sem_pos_embeds, sem_pos_embeds, self.contrast_temperature) + \
                    cal_infonce_loss(behavior_neg_embeds, sem_neg_embeds, sem_neg_embeds, self.contrast_temperature)
        
        embedding_contrast_loss /= behavior_anc_embeds.shape[0]
        embedding_contrast_loss *= self.emb_cl_weight

        bpr_loss = cal_bpr_loss(behavior_anc_embeds, behavior_pos_embeds, behavior_neg_embeds) / behavior_anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)
        loss = bpr_loss + reg_loss + embedding_contrast_loss + full_sup_cl_loss + sup_cl_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'embedding_contrast_loss': embedding_contrast_loss, 'full_sup_cl_loss': full_sup_cl_loss, 'sup_cl_loss': sup_cl_loss}
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds, _ = self.forward(self.adj)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds