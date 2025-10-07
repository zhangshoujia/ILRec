import torch as t
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix
from config.configurator import configs
import pickle

def cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds):
    pos_preds = (anc_embeds * pos_embeds).sum(-1)
    neg_preds = (anc_embeds * neg_embeds).sum(-1)
    return t.sum(F.softplus(neg_preds - pos_preds))

def reg_pick_embeds(embeds_list):
    reg_loss = 0
    for embeds in embeds_list:
        reg_loss += embeds.square().sum()
    return reg_loss

def cal_infonce_loss(embeds1, embeds2, all_embeds2, temp=1.0):
    normed_embeds1 = embeds1 / t.sqrt(1e-8 + embeds1.square().sum(-1, keepdim=True))
    normed_embeds2 = embeds2 / t.sqrt(1e-8 + embeds2.square().sum(-1, keepdim=True))
    normed_all_embeds2 = all_embeds2 / t.sqrt(1e-8 + all_embeds2.square().sum(-1, keepdim=True))
    nume_term = -(normed_embeds1 * normed_embeds2 / temp).sum(-1)
    deno_term = t.log(t.sum(t.exp(normed_embeds1 @ normed_all_embeds2.T / temp), dim=-1))
    cl_loss = (nume_term + deno_term).sum()
    return cl_loss



def cal_sup_cl_loss(embeds1, embeds2, node_emb, temp=0.25):
    pos_score = (embeds1 * embeds2).sum(dim=-1)
    pos_score = t.exp(pos_score / temp)
    # node_emb = t.cat((embeds1, embeds2), dim=0)
    # pro_user_emb, pro_item_emb = self.extract_similar_embeddings(user_emb, item_emb, pro_user_embedding, pro_item_embedding, self.threshold)
    # node_emb = torch.cat((unique_user_emb, unique_item_emb, pro_user_embedding, pro_item_embedding), dim=0) 
    ttl_u_score = t.matmul(embeds1, node_emb.transpose(0, 1))
    ttl_i_score = t.matmul(embeds2, node_emb.transpose(0, 1))
    ttl_score = t.exp( (ttl_u_score +  ttl_i_score ) / temp).sum(dim=1)
    sup_cl_loss = -t.log(pos_score / ttl_score).mean()
    return sup_cl_loss


    


def reg_params(model):
    reg_loss = 0
    for W in model.parameters():
        reg_loss += W.norm(2).square()
    return reg_loss

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss

def ssl_con_loss(x, y, temp=1.0):
    x = F.normalize(x)
    y = F.normalize(y)
    mole = t.exp(t.sum(x * y, dim=1) / temp)
    deno = t.sum(t.exp(x @ y.T / temp), dim=1)
    return -t.log(mole / (deno + 1e-8) + 1e-8).mean()

def alignment(x, y, alpha=2):
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniformity(x):
    x = F.normalize(x, dim=-1)
    return t.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()


def normalize_adj(mat):
    """Laplacian normalization for mat in coo_matrix

    Args:
        mat (scipy.sparse.coo_matrix): the un-normalized adjacent matrix

    Returns:
        scipy.sparse.coo_matrix: normalized adjacent matrix
    """
    degree = np.array(mat.sum(axis=-1))
    d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
    return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()

def make_torch_adj(mat):
    """Transform uni-directional adjacent matrix in coo_matrix into bi-directional adjacent matrix in torch.sparse.FloatTensor

    Args:
        mat (coo_matrix): the uni-directional adjacent matrix

    Returns:
        torch.sparse.FloatTensor: the bi-directional matrix
    """

    a = csr_matrix((configs['data']['user_num'], configs['data']['user_num']))
    b = csr_matrix((configs['data']['item_num'], configs['data']['item_num']))

    mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
    mat = (mat != 0) * 1.0
    mat = normalize_adj(mat)

    # make torch tensor
    idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
    vals = t.from_numpy(mat.data.astype(np.float32))
    shape = t.Size(mat.shape)
    return t.sparse.FloatTensor(idxs, vals, shape).to(configs['device'])

def load_one_mat(file):
    """Load one single adjacent matrix from file

    Args:
        file (string): path of the file to load

    Returns:
        scipy.sparse.coo_matrix: the loaded adjacent matrix
    """
    with open(file, 'rb') as fs:
        mat = (pickle.load(fs) != 0).astype(np.float32)
    if type(mat) != coo_matrix:
        mat = coo_matrix(mat)
    return mat