
import numpy as np
import scipy.sparse as sp
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from models.loss_utils import reg_params
from models.loss_utils import reg_params, cal_sup_cl_loss, cal_infonce_loss, make_torch_adj, load_one_mat
import os


init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform
from models.base_model import BaseModel

import os, torch
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"      # 放在程序最开头
torch.backends.cudnn.enabled = False          # 临时关掉 cudnn（降低异步）

# from recbole.model.init import xavier_uniform_initialization
# from recbole.model.loss import BPRLoss, EmbLoss
# from recbole.utils import InputType
from sklearn.preprocessing import normalize

from config.configurator import configs
from tqdm import tqdm
import random 
from collections import defaultdict

init = nn.init.xavier_uniform_

class NESCL_pro(BaseModel):
    def __init__(self, data_handler):
        super(NESCL_pro, self).__init__(data_handler)

        self.layer_num = configs['model']['layer_num']
        self.reg_weight = float(self.hyper_config.get('reg_weight', 0.0))

        self.adj = data_handler.torch_adj

        self.ssl_reg = self.hyper_config['ssl_reg']
        self.ssl_ratio = self.hyper_config['ssl_ratio']
        self.ssl_temp = self.hyper_config['ssl_temp']
        self.ssl_mode = self.hyper_config['ssl_mode']
        self.aug_type = self.hyper_config['aug_type']

        self.sub_graph_pool = self.hyper_config['sub_graph_pool']
        self.ssl_strategy = self.hyper_config['ssl_strategy']
        self.positive_cl_type = self.hyper_config['positive_cl_type']
        self.k = self.hyper_config['k']
        self.lightgcn_flag = self.hyper_config['lightgcn_flag']
        self.random_strategy = self.hyper_config['random_strategy']
        self.augmentation = self.hyper_config['augmentation']
        self.early_stop = self.hyper_config['early_stop']
        self.add_initial_embedding = self.hyper_config['add_initial_embedding']
        self.pairwise_loss = self.hyper_config['pairwise_loss']
        self.interacted_neighbors = self.hyper_config['interacted_neighbors']
        self.similar_user_neighbors = self.hyper_config['similar_user_neighbors']
        self.similar_item_neighbors = self.hyper_config['similar_item_neighbors']
        self.different_view = self.hyper_config['different_view']

        self.different_view_weight = self.hyper_config['different_view_weight']
        self.interacted_neighbors_weight = self.hyper_config['interacted_neighbors_weight']
        self.sample_item_weight = self.hyper_config['sample_item_weight']
        self.sample_user_weight = self.hyper_config['sample_user_weight']

        self.sample_item_weight_flag = self.hyper_config['sample_item_weight_flag']
        self.sample_user_weight_flag = self.hyper_config['sample_user_weight_flag']

        self.supcon_flag = self.hyper_config['supcon_flag']
        self.prob_sampling = self.hyper_config['prob_sampling']
        self.device = self.hyper_config['device']
        self.neg_num = self.hyper_config['neg_num']



        self.n_users = self.user_num
        self.n_items = self.item_num
        self.n_nodes = self.n_users + self.n_items
        # self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.embedding_size)
        # self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.embedding_size)

        self.user_embedding = nn.Parameter(init(t.empty(self.n_users, self.embedding_size)))
        self.item_embedding = nn.Parameter(init(t.empty(self.n_items, self.embedding_size)))
        
        self.iteration = 0

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None
        

        # parameters initialization
        # self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

        self.interaction_matrix = data_handler._create_sparse_interaction_matrix()

        # user_np, item_np = self.data.interaction_mat.nonzero()
        user_np, item_np = self.interaction_matrix.nonzero()
        self.user_np = user_np.astype(np.int32)
        self.item_np = item_np.astype(np.int32)
        
        self.norm_adj_matrix = data_handler.torch_adj

        if self.lightgcn_flag == True:
            assert self.ssl_strategy == 0
            assert self.augmentation == False

        #import pdb; pdb.set_trace()

        if self.ssl_strategy != 0:
            print('Looking for similar users and items for all users and items!')
            self.user_similar_neighbors_mat, self.user_similar_neighbors_weights_mat, \
            self.item_similar_neighbors_mat, self.item_similar_neighbors_weights_mat = self.get_similar_users_items()
        
        # following variables are used to speedup sampling
        if self.augmentation:
            # self.sub_graph_pool = sub_graph_pool
            self.sub_mat_dict = self.prepare_sub_graphs()
            self.exist_sample_set = set()


        self.semantic_topk = int(self.hyper_config.get('semantic_topk', 10))
        self.sup_cl_weight = float(self.hyper_config.get('sup_cl_weight', 0.1))
        self.cv_sup_cl_weight = float(self.hyper_config.get('cv_sup_cl_weight', 0.1))
        self.emb_cl_weight = float(self.hyper_config.get('emb_cl_weight', 0.0))
        self.contrast_temperature = float(self.hyper_config.get('contrast_temperature', 0.1))
        self.sup_contrast_temperature = float(self.hyper_config.get('sup_contrast_temperature', 0.25))

        self.debug_log = bool(self.hyper_config.get('debug_log', False))
        self.max_debug_steps = int(self.hyper_config.get('max_debug_steps', 2))
        self._debug_count = 0
        target_device = configs['device']  # 直接使用配置中的设备
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

        print(f"✓ 语义嵌入和设备嵌入都在设备: {target_device}")

        # 尝试加载预计算的语义图拉普拉斯矩阵
        self._try_load_semantic_graph_mat()
        
        
        # 确保语义图在正确的设备上（使用配置中的目标设备）
        target_device = configs['device']
        if self.semantic_adj.device != target_device:
            self.semantic_adj = self.semantic_adj.to(target_device)
            print(f"✓ 语义图已移动到目标设备: {target_device}")

        self.interaction_user_indices = None
        self.interaction_item_indices = None

        self._init_weight()

    def _init_weight(self):

        for m in self.semantic_mlps:
            if isinstance(m, nn.Linear):
                init(m.weight)


    def _try_load_semantic_graph_mat(self):
        """尝试加载预计算的语义交互图，使用邻接矩阵"""
        try:
            # 获取数据集名称

            precomputed_path = configs['sem_graph_path']
            
            if os.path.exists(precomputed_path):
                # 使用DataHandlerGeneralCF的方法加载和归一化语义图
                trn_mat = load_one_mat(precomputed_path)
                self.semantic_adj = make_torch_adj(trn_mat)
                print(f"✓ 成功加载预计算的语义图，维度: {self.semantic_adj.size()}")
                print(f"  非零元素数量: {self.semantic_adj._nnz()}")
                print(f"  稀疏度: {1.0 - (self.semantic_adj._nnz() / (self.semantic_adj.size(0) * self.semantic_adj.size(1))):.4f}")
                print(f"  设备: {self.semantic_adj.device}")
                print(f"  目标设备: {configs['device']}")

                # 确保语义图在正确的设备上
                if self.semantic_adj.device != configs['device']:
                    if self._debug_count < 3:  # 只在前几次打印警告
                        print(f"⚠️ 设备不匹配，将语义图从 {self.semantic_adj.device} 移动到 {configs['device']}")
                    self.semantic_adj = self.semantic_adj.to(configs['device'])

        except Exception as e:
            print(f"⚠️ 加载预计算语义图失败: {e}")
            print("  继续使用动态构建的语义图")

    def make_all_one_adj(self):
        idxs = self.adj._indices()
        vals = t.ones_like(self.adj._values())
        shape = self.adj.shape
        return t.sparse.FloatTensor(idxs, vals, shape).cuda()

    def BPRLoss(self, pos_scores, neg_scores):
        # pos_scores, neg_scores: 1D 实数向量
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))
    def EmbLoss(self):
        return reg_params(self) * self.reg_weight

    def get_similar_users_items(self):
        # load parameters info

        # self.shrink = config['shrink'] if 'shrink' in config else 0.0
        self.shrink = 0.0

        # interaction_matrix = self.data.interaction_mat
        interaction_matrix = self.interaction_matrix
        # print(f"interaction_matrix: ", interaction_matrix)
        shape = interaction_matrix.shape
        assert self.n_users == shape[0] and self.n_items == shape[1]
        print(f'ComputeSimilarity Users')
        user_similar_neighbors_mat, user_similar_neighbors_weights_mat = ComputeSimilarity(self, interaction_matrix, topk=self.k,
                                      shrink=self.shrink).compute_similarity('user')
        print(f'ComputeSimilarity Items')
        item_similar_neighbors_mat, item_similar_neighbors_weights_mat = ComputeSimilarity(self, interaction_matrix, topk=self.k,
                                      shrink=self.shrink).compute_similarity('item')

        return user_similar_neighbors_mat, user_similar_neighbors_weights_mat, item_similar_neighbors_mat, item_similar_neighbors_weights_mat
    
    def prepare_first_neighbors_dict(self):
        first_neighbors_dict = defaultdict(set)
        
        for idx, user in enumerate(self.user_np):
            item = self.item_np[idx] + self.n_users
            first_neighbors_dict[user].add(item)
            first_neighbors_dict[item].add(user)
        return first_neighbors_dict

    def prepare_sub_graphs(self):
        aug_type = self.aug_type

        sub_mat_dict = {}
        for idx in tqdm(range(self.sub_graph_pool)):
            # sub_mat_dict[idx] = self.get_norm_adj_mat(True, aug_type).to(self.device) if self.gpu_available else self.get_norm_adj_mat(True, aug_type)
            # sub_mat_dict[idx] = self.get_norm_adj_mat(True, aug_type).to(self.device)
            sub_mat_dict[idx] = self.get_norm_adj_mat(True, aug_type).cuda()
        return sub_mat_dict

    def get_norm_adj_mat(self, is_subgraph=False, aug_type=0):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        if is_subgraph and aug_type in [0, 1, 2] and self.ssl_ratio > 0:
            # data augmentation type --- 0: Node Dropout; 1: Edge Dropout; 2: Random Walk
            if aug_type == 0:
                # print(f"n_users: {self.n_users}, ssl_ratio: {self.ssl_ratio}, size: {int(self.n_users * self.ssl_ratio)}")
                drop_user_idx = self.randint_choice(high=self.n_users, size=int(self.n_users * self.ssl_ratio), replace=False)
                drop_item_idx = self.randint_choice(high=self.n_items, size=int(self.n_items * self.ssl_ratio), replace=False)
                indicator_user = np.ones(self.n_users, dtype=np.float32)
                indicator_item = np.ones(self.n_items, dtype=np.float32)
                indicator_user[drop_user_idx] = 0.
                indicator_item[drop_item_idx] = 0.
                diag_indicator_user = sp.diags(indicator_user)
                diag_indicator_item = sp.diags(indicator_item)

                R = sp.csr_matrix(
                    (np.ones_like(self.user_np, dtype=np.float32), (self.user_np, self.item_np)), 
                    shape=(self.n_users, self.n_items))
                R_prime = diag_indicator_user.dot(R).dot(diag_indicator_item)
                (user_np_keep, item_np_keep) = R_prime.nonzero()
                ratings_keep = R_prime.data
                tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep+self.n_users)), shape=(self.n_nodes, self.n_nodes))
            elif aug_type in [1, 2]:
                '''
                np.random.shuffle(self.nums)
                keep_idx = np.array(range(len(self.user_np)))[self.nums.astype(int)==1].tolist()
                '''
                keep_idx = self.randint_choice(high=len(self.user_np), size=int(len(self.user_np) * (1 - self.ssl_ratio)), replace=False)

                user_np = np.array(self.user_np)[keep_idx]
                item_np = np.array(self.item_np)[keep_idx]
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.n_users)), shape=(self.n_nodes, self.n_nodes))
                #'''
            else:
                raise ValueError("Invalid aug_type!")
        else:
            user_np = np.array(self.user_np)
            item_np = np.array(self.item_np)
            ratings = np.ones_like(user_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.n_users)), shape=(self.n_nodes, self.n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        # print('use the pre adjcency matrix')

        adj_matrix = adj_matrix.tocoo()

        index = torch.LongTensor([adj_matrix.row, adj_matrix.col])
        data = torch.FloatTensor(adj_matrix.data)
        SparseL = torch.sparse.FloatTensor(index, data, torch.Size(adj_matrix.shape))

        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding
        item_embeddings = self.item_embedding
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def sample_subgraph_idx(self):
        idx_1 = np.random.randint(0, self.sub_graph_pool)
        idx_2 = np.random.randint(0, self.sub_graph_pool)

        while (idx_1, idx_2) in self.exist_sample_set:
            idx_1 = np.random.randint(0, self.sub_graph_pool)
            idx_2 = np.random.randint(0, self.sub_graph_pool)

        self.exist_sample_set.add((idx_1, idx_2))
        return self.sub_mat_dict[idx_1], self.sub_mat_dict[idx_2]

    def forward(self):
        aug_type = self.aug_type

        #'''
        # start to augment the input data
        if self.augmentation:
            if aug_type in [0, 1]:
                sub_mat = {}
                sub_mat_1, sub_mat_2 = self.sample_subgraph_idx()
                for k in range(self.layer_num):
                    sub_mat['sub_mat_1_layer_%d' % k] = sub_mat_1
                    sub_mat['sub_mat_2_layer_%d' % k] = sub_mat_2
            else:
                sub_mat = {}
                for k in range(self.layer_num):
                    sub_mat['sub_mat_1_layer_%d' % k], sub_mat['sub_mat_2_layer_%d' % k] = self.sample_subgraph_idx()
        #'''
        
        all_embeddings = self.get_ego_embeddings()
        all_embeddings_sub1 = all_embeddings
        all_embeddings_sub2 = all_embeddings

        if self.add_initial_embedding:
            embeddings_list = [all_embeddings]
            embeddings_list_sub1 = [all_embeddings_sub1]
            embeddings_list_sub2 = [all_embeddings_sub2]
        elif not self.add_initial_embedding:
            embeddings_list = [] 
            embeddings_list_sub1 = []
            embeddings_list_sub2 = []

        for layer_idx in range(self.layer_num):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)

            #'''
            if self.augmentation:
                all_embeddings_sub1 = torch.sparse.mm(sub_mat['sub_mat_1_layer_%d' % layer_idx], all_embeddings_sub1)
                embeddings_list_sub1.append(all_embeddings_sub1)

                all_embeddings_sub2 = torch.sparse.mm(sub_mat['sub_mat_2_layer_%d' % layer_idx], all_embeddings_sub2)
                embeddings_list_sub2.append(all_embeddings_sub2)
            #'''

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])

        #'''
        if self.augmentation:
            lightgcn_all_embeddings_sub1 = torch.stack(embeddings_list_sub1, dim=1)
            lightgcn_all_embeddings_sub1 = torch.mean(lightgcn_all_embeddings_sub1, dim=1)
            user_all_embeddings_sub1, item_all_embeddings_sub1 = torch.split(lightgcn_all_embeddings_sub1, [self.n_users, self.n_items])

            lightgcn_all_embeddings_sub2 = torch.stack(embeddings_list_sub2, dim=1)
            lightgcn_all_embeddings_sub2 = torch.mean(lightgcn_all_embeddings_sub2, dim=1)
            user_all_embeddings_sub2, item_all_embeddings_sub2 = torch.split(lightgcn_all_embeddings_sub2, [self.n_users, self.n_items])
        #'''

        #import pdb; pdb.set_trace()

        # if torch.isnan(torch.sum(self.user_embedding.weight)):
        #     import pdb; pdb.set_trace()

        return user_all_embeddings, item_all_embeddings, user_all_embeddings_sub1, item_all_embeddings_sub1, \
            user_all_embeddings_sub2, item_all_embeddings_sub2

    def forward1(self, adj=None, keep_rate=1.0):
        if adj is None:
            adj = self.adj
            semantic_adj = self.semantic_adj

        seman_embeds = t.cat([self.semantic_user_embeds, self.semantic_item_embeds], dim=0)
        sem_embeds_list = [seman_embeds]   # 语义交互图

        adj = self.adj
        semantic_adj = self.semantic_adj

        # 双图并行学习
        for layer_idx in range(self.layer_num):
            # 语义交互图传播
            sem_embeds = t.spmm(semantic_adj, sem_embeds_list[-1])
            sem_embeds_list.append(sem_embeds)

        # 最终表示
        final_sem_embeds = sum(sem_embeds_list)
        # 分离用户和物品嵌入
        sem_user_embeds = final_sem_embeds[:self.user_num]
        sem_item_embeds = final_sem_embeds[self.user_num:]
        # 返回：行为交互图的用户/物品嵌入 + 语义交互图的用户/物品嵌入 + 行为交互嵌入
        return sem_user_embeds, sem_item_embeds

    def cal_loss(self, batch_data):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user_idx, pos_idx, neg_idx = batch_data

        user = user_idx
        pos_item = pos_idx
        neg_item = neg_idx

        user = torch.tensor(user, dtype=torch.long).cuda()
        pos_item = torch.tensor(pos_item, dtype=torch.long).cuda()
        neg_item = torch.tensor(neg_item, dtype=torch.long).cuda()

        orig_user_embeds, orig_item_embeds, user_all_embeddings_sub1, item_all_embeddings_sub1, \
            user_all_embeddings_sub2, item_all_embeddings_sub2 = self.forward()

        #with torch.no_grad():
        #    count, _ = self.update_cluster(interaction, user_all_embeddings, item_all_embeddings)

        u_embeddings = orig_user_embeds[user]
        pos_embeddings = orig_item_embeds[pos_item]

        u_embeddings = u_embeddings.repeat(self.neg_num, 1)
        pos_embeddings = pos_embeddings.repeat(self.neg_num, 1)

        neg_embeddings = orig_item_embeds[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.BPRLoss(pos_scores, neg_scores)

        # calculate regularization Loss
        # u_ego_embeddings = self.user_embedding(user)
        # pos_ego_embeddings = self.item_embedding(pos_item)
        # neg_ego_embeddings = self.item_embedding(neg_item)
        reg_loss = self.EmbLoss()

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
        # sem_node_emb, node_emb
        # sem_item_embeds, orig_item_embeds
        sup_cl_user_loss = cal_sup_cl_loss(behavior_anc_embeds, sem_pos_embeds, sem_node_emb, self.sup_contrast_temperature)
        sup_cl_item_loss = cal_sup_cl_loss(sem_anc_embeds, behavior_pos_embeds, node_emb, self.sup_contrast_temperature)
        full_sup_cl_loss = (sup_cl_user_loss + sup_cl_item_loss) * self.cv_sup_cl_weight
        embedding_contrast_loss = cal_infonce_loss(behavior_anc_embeds, sem_anc_embeds, sem_user_embeds, self.contrast_temperature) + \
                    cal_infonce_loss(behavior_pos_embeds, sem_pos_embeds, sem_pos_embeds, self.contrast_temperature) + \
                    cal_infonce_loss(behavior_neg_embeds, sem_neg_embeds, sem_neg_embeds, self.contrast_temperature)
        
        embedding_contrast_loss /= behavior_anc_embeds.shape[0]
        embedding_contrast_loss *= self.emb_cl_weight

        if self.lightgcn_flag == False:
            #'''
            # calculate SSL Loss
            if self.ssl_mode in [0, 1, 2]:
                ssl_loss = self.calculate_ssl_loss_1(user_idx, pos_idx, neg_idx)
            elif self.ssl_mode in [3]:
                ssl_loss = self.calculate_ssl_loss_2(user_idx, pos_idx, orig_user_embeds, orig_item_embeds, \
                user_all_embeddings_sub1, item_all_embeddings_sub1, user_all_embeddings_sub2, item_all_embeddings_sub2)
            else:
                raise ValueError("Invalid ssl_mode!")
            #'''

            #import pdb; pdb.set_trace()
            if self.pairwise_loss:
                loss = mf_loss + self.reg_weight * reg_loss + ssl_loss + embedding_contrast_loss + full_sup_cl_loss + sup_cl_loss
            else:
                loss = self.reg_weight + ssl_loss + embedding_contrast_loss + full_sup_cl_loss + sup_cl_loss
        else:
            loss = mf_loss + self.reg_weight * reg_loss + embedding_contrast_loss + full_sup_cl_loss + sup_cl_loss
            # ssl_loss = torch.Tensor([0]).to(self.device)
            ssl_loss = torch.Tensor([0]).cuda()

        losses = {'rec_loss': mf_loss, 'reg_loss': self.reg_weight * reg_loss, 'cl_loss': ssl_loss, 'embedding_contrast_loss': embedding_contrast_loss, 'full_sup_cl_loss': full_sup_cl_loss, 'sup_cl_loss': sup_cl_loss}
        return loss, losses


    def full_predict(self, batch_data):
        user_embeds, item_embeds, _, _, _, _ = self.forward()
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds


    def neighbor_sample(self, input_list, weight_list):
        if len(input_list) == 0:
            # 返回自身作为默认邻居
            return -1, 0.0  # 或其他全局随机节点
        if len(input_list) == 1:
            return input_list[0], weight_list[0]
        else:
            if self.prob_sampling:
                prob = np.asarray(weight_list).astype('float64')
                prob = prob / sum(prob)
                #import pdb; pdb.set_trace()
                idx = np.random.choice(range(0, len(input_list)), size=1, replace=True, p=prob)
                idx = idx.item()
                #import pdb; pdb.set_trace()
            else:
                idx = np.random.randint(0, len(input_list))
            #random.shuffle(input_list)
            return input_list[idx], weight_list[idx]

    def construct_batch_dict(self, interaction):
        batch_user_dict, batch_item_dict = defaultdict(list), defaultdict(list)
        batch_user_list = interaction[self.USER_ID].cpu().numpy().tolist()
        batch_item_list = interaction[self.ITEM_ID].cpu().numpy().tolist()

        for idx, user in enumerate(batch_user_list):
            item = batch_item_list[idx]
            batch_user_dict[user].append(item)
            batch_item_dict[item].append(user)

        return batch_user_dict, batch_item_dict

    # calculate the ssl loss with the ssl_mode ['merge']
    def calculate_ssl_loss_2(self, user_idx, pos_idx, user_all_embeddings, item_all_embeddings, \
    user_all_embeddings_sub1, item_all_embeddings_sub1, user_all_embeddings_sub2, item_all_embeddings_sub2):

        # interaction is the input data
        # user = interaction[self.USER_ID]
        # pos_item = interaction[self.ITEM_ID]

        user = user_idx
        pos_item = pos_idx
        # user = torch.tensor(user, dtype=torch.long, device=self.device)
        # pos_item = torch.tensor(pos_item, dtype=torch.long, device=self.device)

        user = torch.tensor(user, dtype=torch.long).cuda()
        pos_item = torch.tensor(pos_item, dtype=torch.long).cuda()
        
        #batch_user_dict, batch_item_dict = self.construct_batch_dict(interaction)
  
        # Original Algorithm: following is original ssl loss calculation function
        if self.ssl_strategy == 0:
            # following code is used for original ssl loss computation
            #batch_users = torch.unique(user)
            batch_users = user
            user_emb1 = user_all_embeddings_sub1[batch_users]
            user_emb2 = user_all_embeddings_sub2[batch_users]

            #batch_items = torch.unique(pos_item)
            batch_items = pos_item
            item_emb1 = item_all_embeddings_sub1[batch_items]
            item_emb2 = item_all_embeddings_sub2[batch_items]

            emb_merge1 = torch.cat([user_emb1, item_emb1], dim=0)
            emb_merge2 = torch.cat([user_emb2, item_emb2], dim=0)

            # cosine similarity
            normalize_emb_merge1 = torch.nn.functional.normalize(emb_merge1, p=2, dim=1)
            normalize_emb_merge2 = torch.nn.functional.normalize(emb_merge2, p=2, dim=1)

            pos_score = torch.sum(torch.multiply(normalize_emb_merge1, normalize_emb_merge2), dim=1)
            ttl_score = torch.matmul(normalize_emb_merge1, normalize_emb_merge2.transpose(0, 1))

            #'''
            pos_score = torch.exp(pos_score / self.ssl_temp)
            ttl_score = torch.sum(torch.exp(ttl_score / self.ssl_temp), dim=1)
            ssl_loss = -torch.mean(torch.log(pos_score / ttl_score))
            ssl_loss = self.ssl_reg * ssl_loss
            #'''


        # SupCL two kinds of loss functions 
        elif self.ssl_strategy == 9:
            if self.augmentation:
                node_all_embeddings_sub1 = torch.cat([user_all_embeddings_sub1, item_all_embeddings_sub1], dim=0)
                node_all_embeddings_sub2 = torch.cat([user_all_embeddings_sub2, item_all_embeddings_sub2], dim=0)
            else:
                node_all_embeddings_sub1 = torch.cat([user_all_embeddings, item_all_embeddings], dim=0)
                node_all_embeddings_sub2 = torch.cat([user_all_embeddings, item_all_embeddings], dim=0)

            batch_user_weight = []
            batch_item_weight = []

            # batch_users_3 is used to index the user embedding from view-2
            batch_users_3 = []
            # batch_items_3 is used to index the user embedding from view-2
            batch_items_3 = []

            # batch_users_4 is used to index the user embedding from view-1
            batch_users_4 = []
            # batch_items_4 is used to index the user embedding from view-1
            batch_items_4 = []

            batch_nodes_list = []

            with torch.no_grad():
                batch_users_list = user.cpu().numpy().tolist()
                # update item ids to map the original item id to the constructed graph
                batch_items_list = (pos_item+self.n_users).cpu().numpy().tolist()

                # batch_nodes_list stores both the batch_users_list and the batch_item_list
                batch_nodes_list.extend(batch_users_list)
                batch_nodes_list.extend(batch_items_list)

                for idx, user in enumerate(batch_users_list):
                    if self.different_view:
                        batch_user_weight.append(self.different_view_weight)

                        # user itself is also treated as the positive sample, \mathbf{h}_i''
                        batch_users_3.append(user)
                        # \mathbf{h}_i'
                        batch_users_4.append(user)

                    # add user-item positive pair
                    if self.interacted_neighbors:
                        item = batch_items_list[idx]
                        batch_user_weight.append(self.interacted_neighbors_weight)
                        # interacted item is treated as the positive sample, \mathbf{h}_k''
                        batch_users_3.append(item)
                        # \mathbf{h}_i'
                        batch_users_4.append(user)

                    # add user and her k-nearest neighbors positive pair
                    if self.similar_user_neighbors:
                        if self.random_strategy:
                            # the prob_sampling used in self.neighbor_sample controls whether sampling the collaborative neighbors with their probabilities 
                            sample_user, sample_weight = self.neighbor_sample(self.user_similar_neighbors_mat[user], self.user_similar_neighbors_weights_mat[user])
                            if sample_user == -1:
                                # 无邻居节点，使用默认值
                                sample_user = user
                            # sample_user_weight_flag controls whether add weights to the collaborative neighbors 
                            if self.sample_user_weight_flag:
                                batch_user_weight.append(sample_weight * self.sample_user_weight)
                            else:
                                batch_user_weight.append(self.sample_user_weight)

                            batch_users_3.append(sample_user)
                            batch_users_4.append(user)

                            #if self.add_more_neighbors:
                            #    sample_item = self.neighbor_sample(self.item_similar_neighbors_mat[item-self.n_users])+self.n_users

                            #    batch_users_3.append(item)
                            #    batch_users_4.append(user)
                        else:
                            for idx, sample_user in enumerate(self.user_similar_neighbors_mat[user]):
                                batch_users_3.append(sample_user)
                                batch_users_4.append(user)

                                if self.sample_user_weight_flag:
                                    sample_weight = self.user_similar_neighbors_weights_mat[user][idx]
                                    batch_user_weight.append(sample_weight * self.sample_user_weight)

                for idx, item in enumerate(batch_items_list):
                    if self.different_view:
                        
                        batch_item_weight.append(self.different_view_weight)

                        batch_items_3.append(item)
                        batch_items_4.append(item)

                    # add item-user positive pair
                    if self.interacted_neighbors:
                        user = batch_users_list[idx]

                        batch_item_weight.append(self.interacted_neighbors_weight)

                        batch_items_3.append(user)
                        batch_items_4.append(item)

                    # add item and its k-nearest neighbors positive pair
                    if self.similar_item_neighbors:
                        if self.random_strategy:
                            sample_item, sample_weight = self.neighbor_sample(self.item_similar_neighbors_mat[item-self.n_users], self.item_similar_neighbors_weights_mat[item-self.n_users])
                            if sample_item == -1:
                                # 回退成“当前正例 item”的局部 id
                                sample_item = item - self.n_users
                            # 统一在最后加一次 offset，映射到“全图 id”
                            sample_item = sample_item + self.n_users

                            if self.sample_item_weight_flag:
                                batch_item_weight.append(sample_weight * self.sample_item_weight)
                            else:
                                batch_item_weight.append(self.sample_item_weight)

                            batch_items_3.append(sample_item)
                            batch_items_4.append(item)

                            #if self.add_more_neighbors:
                            #    sample_user = self.neighbor_sample(self.user_similar_neighbors_mat[user])

                            #    batch_items_3.append(sample_user)
                            #    batch_items_4.append(user)
                        else:
                            for idx, sample_item in enumerate(self.item_similar_neighbors_mat[item-self.n_users]):
                                sample_item += self.n_users

                                batch_items_3.append(sample_item)
                                batch_items_4.append(item)

                                if self.sample_item_weight_flag:
                                    sample_weight = self.item_similar_neighbors_weights_mat[item-self.n_users][idx]
                                    batch_item_weight.append(sample_weight * self.sample_item_weight)

                # batch_users_3 = torch.tensor(batch_users_3).long().to(self.device)
                # batch_items_3 = torch.tensor(batch_items_3).long().to(self.device)

                # batch_users_4 = torch.tensor(batch_users_4).long().to(self.device)
                # batch_items_4 = torch.tensor(batch_items_4).long().to(self.device)

                batch_users_3 = torch.tensor(batch_users_3).long().cuda()
                batch_items_3 = torch.tensor(batch_items_3).long().cuda()

                batch_users_4 = torch.tensor(batch_users_4).long().cuda()
                batch_items_4 = torch.tensor(batch_items_4).long().cuda()

                #if self.set_batch_nodes:
                #    batch_nodes_list = torch.tensor(list(set(batch_nodes_list))).long().to(self.device)
                #else:
                # batch_nodes_list = torch.tensor(list(batch_nodes_list)).long().to(self.device)
                batch_nodes_list = torch.tensor(list(batch_nodes_list)).long().cuda()


            if self.supcon_flag:
                # batch_users_3, batch_items_3 are consisf of different positive samples, get representations from view-2
                user_emb3 = node_all_embeddings_sub2[batch_users_3]
                item_emb3 = node_all_embeddings_sub2[batch_items_3]

                # batch_users_4, batch_items_4 are consisf of the anchor nodes themseleves, get representations from view-1
                user_emb4 = node_all_embeddings_sub1[batch_users_4]
                item_emb4 = node_all_embeddings_sub1[batch_items_4]

                # get representations from view-2
                batch_node_emb = node_all_embeddings_sub2[batch_nodes_list]

                emb_merge3 = torch.cat([user_emb3, item_emb3], dim=0)
                emb_merge4 = torch.cat([user_emb4, item_emb4], dim=0)

                # cosine similarity
                normalize_emb_merge3 = torch.nn.functional.normalize(emb_merge3, p=2, dim=1)
                normalize_emb_merge4 = torch.nn.functional.normalize(emb_merge4, p=2, dim=1)
                normalize_batch_node_emb = torch.nn.functional.normalize(batch_node_emb, p=2, dim=1)

                # the anchor nodes' representations from view-1 mutliply differeent kinds of positive samples from view-2 
                pos_score = torch.sum(torch.multiply(normalize_emb_merge4, normalize_emb_merge3), dim=1)

                # the anchor nodes' representations from view-1 matmul the negative samples from view-2
                ttl_score = torch.matmul(normalize_emb_merge4, normalize_batch_node_emb.transpose(0, 1))
            
            else:
                # batch_users_3, batch_items_3 are consisf of different positive samples, get representations from view-1
                user_emb3 = node_all_embeddings_sub1[batch_users_3]
                item_emb3 = node_all_embeddings_sub1[batch_items_3]

                # batch_users_4, batch_items_4 are consisf of the anchor nodes themseleves, get representations from view-2
                user_emb4 = node_all_embeddings_sub2[batch_users_4]
                item_emb4 = node_all_embeddings_sub2[batch_items_4]

                # get representations from view-2
                batch_node_emb = node_all_embeddings_sub2[batch_nodes_list]
                
                emb_merge3 = torch.cat([user_emb3, item_emb3], dim=0)
                emb_merge4 = torch.cat([user_emb4, item_emb4], dim=0)

                # cosine similarity
                normalize_emb_merge3 = torch.nn.functional.normalize(emb_merge3, p=2, dim=1)
                normalize_emb_merge4 = torch.nn.functional.normalize(emb_merge4, p=2, dim=1)
                normalize_batch_node_emb = torch.nn.functional.normalize(batch_node_emb, p=2, dim=1)

                n_nodes = node_all_embeddings_sub2.size(0)  # = n_users + n_items
                self._assert_idx(batch_users_3,    n_nodes, "batch_users_3")
                self._assert_idx(batch_items_3,    n_nodes, "batch_items_3")
                self._assert_idx(batch_users_4,    n_nodes, "batch_users_4")
                self._assert_idx(batch_items_4,    n_nodes, "batch_items_4")
                self._assert_idx(batch_nodes_list, n_nodes, "batch_nodes_list")

                # differeent kinds of positive samples from view-1 mutliply the anchor nodes' representations from view-2
                pos_score = torch.sum(torch.multiply(normalize_emb_merge3, normalize_emb_merge4), dim=1)
                #pos_score = pos_score.reshape(-1, self.k_values + 1)

                # different kinds of positive samples from view-1 matmul the negative samples from view-2
                # print(f"normalize_emb_merge3.shape: ", normalize_emb_merge3.shape)
                # print(f"normalize_batch_node_emb.shape: ", normalize_batch_node_emb.shape)
                ttl_score = torch.matmul(normalize_emb_merge3, normalize_batch_node_emb.transpose(0, 1))
            
            #import pdb; pdb.set_trace()


            #batch_node_weight = torch.Tensor(batch_user_weight+batch_item_weight).float().cuda()

            #pos_score = pos_score * batch_node_weight

            pos_score = torch.exp(pos_score / self.ssl_temp)
            ttl_score = torch.sum(torch.exp(ttl_score / self.ssl_temp), dim=1)

            #import pdb; pdb.set_trace()

            # it corresponds to the supcl-in version, log every element
            if self.positive_cl_type == 1:
                ssl_loss = -torch.mean(torch.log(pos_score / ttl_score))
            # it corresponds to the supcl-out version, log the sum of the same node
            elif self.positive_cl_type == 2:
                if self.random_strategy:
                    pos_score = pos_score.reshape(-1, 3)
                    ttl_score = ttl_score.reshape(-1, 3)
                else:
                    pos_score = pos_score.reshape(-1, 2 + self.k)
                    ttl_score = ttl_score.reshape(-1, 2 + self.k)
                ssl_loss = -torch.mean(torch.log(torch.mean(pos_score / ttl_score, dim=1)))
            
            ssl_loss = self.ssl_reg * ssl_loss

        return ssl_loss

    def _assert_idx(self, idx, N, label):
        if idx is None: return
        if not torch.is_tensor(idx):
            idx = torch.as_tensor(idx, device='cuda')
        if idx.dtype != torch.long:
            idx = idx.long()
        mi = int(idx.min().item()); ma = int(idx.max().item())
        assert mi >= 0, f"[{label}] negative index min={mi}"
        assert ma < N, f"[{label}] out of bounds: max={ma} >= {N}"


    def randint_choice(self, high, size=1, replace=True, p=None, exclusion=None):
        """
        Sample random integers from [0, high).

        Args:
            high (int): Upper limit of the range (exclusive).
            size (int): Number of integers to sample.
            replace (bool): Whether to allow duplicate sampling.
            p (list, optional): Probability distribution (not implemented here).
            exclusion (iterable, optional): Numbers to exclude from sampling.

        Returns:
            int or list: A single integer if size is 1, else a list of integers.
        """
        # Input validation
        if size <= 0:
            raise ValueError("'size' must be a positive integer.")
        if not isinstance(replace, bool):
            raise TypeError("'replace' must be a boolean.")
        if p is not None:
            raise NotImplementedError("Probability sampling is not implemented.")
        if exclusion is not None and high <= len(exclusion):
            raise ValueError("The number of 'exclusion' is greater than or equal to 'high'.")
        
        # Initialize exclusion set
        exclusion_set = set(exclusion) if exclusion else set()
        available_size = high - len(exclusion_set)
        if not replace and available_size < size:
            raise ValueError("Not enough integers available for sampling without replacement.")

        result = []
        sampled_set = set() if not replace else None

        while len(result) < size:
            candidate = random.randint(0, high - 1)
            if candidate in exclusion_set:
                continue
            if not replace and candidate in sampled_set:
                continue
            result.append(candidate)
            if not replace:
                sampled_set.add(candidate)

        return result[0] if size == 1 else result
    

class ComputeSimilarity:

    def __init__(self, model, dataMatrix, topk=10, shrink=0, normalize=True):
        r"""Computes the cosine similarity of dataMatrix

        If it is computed on :math:`URM=|users| \times |items|`, pass the URM.

        If it is computed on :math:`ICM=|items| \times |features|`, pass the ICM transposed.

        Args:
            dataMatrix (scipy.sparse.csr_matrix): The sparse data matrix.
            topk (int) : The k value in KNN.
            shrink (int) :  hyper-parameter in calculate cosine distance.
            normalize (bool):   If True divide the dot product by the product of the norms.
        """

        super(ComputeSimilarity, self).__init__()

        self.shrink = shrink
        self.normalize = normalize

        self.n_rows, self.n_columns = dataMatrix.shape
        self.TopK = min(topk, self.n_columns)

        self.dataMatrix = dataMatrix.copy()

        self.model = model

    def compute_similarity(self, method, block_size=100):
        r"""Compute the similarity for the given dataset

        Args:
            method (str) : Caculate the similarity of users if method is 'user', otherwise, calculate the similarity of items.
            block_size (int): divide matrix to :math:`n\_rows \div block\_size` to calculate cosine_distance if method is 'user',
                 otherwise, divide matrix to :math:`n\_columns \div block\_size`. 

        Returns:

            list: The similar nodes, if method is 'user', the shape is [number of users, neigh_num], 
            else, the shape is [number of items, neigh_num].
            scipy.sparse.csr_matrix: sparse matrix W, if method is 'user', the shape is [self.n_rows, self.n_rows],
            else, the shape is [self.n_columns, self.n_columns].
        """

        user_similar_neighbors_mat, item_similar_neighbors_mat = [], []
        user_similar_neighbors_weights_mat, item_similar_neighbors_weights_mat = [], []

        values = []
        rows = []
        cols = []
        neigh = []

        self.dataMatrix = self.dataMatrix.astype(np.float32)

        # Compute sum of squared values to be used in normalization
        if method == 'user':
            sumOfSquared = np.array(self.dataMatrix.power(2).sum(axis=1)).ravel()
            end_local = self.n_rows
        elif method == 'item':
            sumOfSquared = np.array(self.dataMatrix.power(2).sum(axis=0)).ravel()
            end_local = self.n_columns
        else:
            raise NotImplementedError("Make sure 'method' in ['user', 'item']!")
        sumOfSquared = np.sqrt(sumOfSquared)

        start_block = 0

        # Compute all similarities using vectorization
        while start_block < end_local:

            end_block = min(start_block + block_size, end_local)
            this_block_size = end_block - start_block

            # All data points for a given user or item
            if method == 'user':
                self.dataMatrix = self.dataMatrix.tocsr(copy=False)
                data = self.dataMatrix[start_block:end_block, :]
            else:
                self.dataMatrix = self.dataMatrix.tocsc(copy=False)
                data = self.dataMatrix[:, start_block:end_block]
            data = data.toarray().squeeze()

            if data.ndim == 1:
                data = np.expand_dims(data, axis=1)
            
            # Compute similarities

            if method == 'user':
                this_block_weights = self.dataMatrix.dot(data.T)
            else:
                this_block_weights = self.dataMatrix.T.dot(data)

            for index_in_block in range(this_block_size):

                if this_block_size == 1:
                    this_line_weights = this_block_weights.squeeze()
                else:
                    this_line_weights = this_block_weights[:, index_in_block]

                Index = index_in_block + start_block
                this_line_weights[Index] = 0.0

                # Apply normalization and shrinkage, ensure denominator != 0
                if self.normalize:
                    denominator = sumOfSquared[Index] * sumOfSquared + self.shrink + 1e-6
                    this_line_weights = np.multiply(this_line_weights, 1 / denominator)

                elif self.shrink != 0:
                    this_line_weights = this_line_weights / self.shrink

                # Sort indices and select TopK
                # Sorting is done in three steps. Faster then plain np.argsort for higher number of users or items
                # - Partition the data to extract the set of relevant users or items
                # - Sort only the relevant users or items
                # - Get the original index
                relevant_partition = (-this_line_weights).argpartition(self.TopK - 1)[0:self.TopK]

                relevant_partition_sorting = np.argsort(-this_line_weights[relevant_partition])
                top_k_idx = relevant_partition[relevant_partition_sorting]
                neigh.append(top_k_idx)

                # Incrementally build sparse matrix, do not add zeros
                notZerosMask = this_line_weights[top_k_idx] != 0.0
                tmp_values = this_line_weights[top_k_idx][notZerosMask]
                
                '''
                if self.model.renormalize_similarity:
                    if len(tmp_values) != 0:
                        tmp_values = np.array(tmp_values).reshape(1, -1)
                        tmp_values = normalize(tmp_values, norm='l1', axis=1)
                        tmp_values = tmp_values[0].tolist()
                        #import pdb; pdb.set_trace()
                #'''

                if method == 'user':
                    user_similar_neighbors_mat.append(top_k_idx[notZerosMask])
                    user_similar_neighbors_weights_mat.append(tmp_values)
                else:
                    item_similar_neighbors_mat.append(top_k_idx[notZerosMask])
                    item_similar_neighbors_weights_mat.append(tmp_values)

            start_block += block_size

        if method == 'user':
            return user_similar_neighbors_mat, user_similar_neighbors_weights_mat
        elif method == 'item':
            return item_similar_neighbors_mat, item_similar_neighbors_weights_mat