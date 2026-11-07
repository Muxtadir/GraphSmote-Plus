import argparse
import json
import os
import scipy.sparse as sp
import numpy as np
import torch
import ipdb
from scipy.io import loadmat
import networkx as nx
import multiprocessing as mp
import torch.nn.functional as F
from functools import partial
import random
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    balanced_accuracy_score,
    average_precision_score,
)
from copy import deepcopy
from scipy.spatial.distance import pdist,squareform

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--nhid', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--size', type=int, default=100)

    parser.add_argument('--epochs', type=int, default=2010,
                help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--batch_nums', type=int, default=6000, help='number of batches per epoch')
    parser.add_argument('--batch_size', type=int, default=40, help='number of batches per epoch')


    parser.add_argument('--imbalance', action='store_true', default=False)
    parser.add_argument('--setting', type=str, default='no', 
        choices=['no','upsampling', 'smote','reweight','embed_up', 'recon','newG_cls','recon_newG',
                 'sta_newG_cls','sta_recon_newG'])
    #upsampling: oversample in the raw input; smote: ; reweight: reweight minority classes; 
    # embed_up: 
    # recon: pretrain; newG_cls: pretrained decoder; recon_newG: also finetune the decoder
    # sta_newG_cls / sta_recon_newG: GraphSMOTE-STA (Zhao et al. GraphSMOTE + anchor-local soft attachment)

    parser.add_argument('--opt_new_G', action='store_true', default=False) # whether optimize the decoded graph based on classification result.
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--up_scale', type=float, default=1)
    parser.add_argument('--im_ratio', type=float, default=0.5)
    parser.add_argument('--rec_weight', type=float, default=0.000001)
    parser.add_argument('--model', type=str, default='sage', 
        choices=['sage','gcn','GAT'])

    # ---- GraphSMOTE-STA flags (additive; default preserves original GraphSMOTE) ----
    # Attribution: GraphSMOTE = Zhao, Zhang, Wang (WSDM 2021). STA is a Plus extension.
    # sta_attach default None → resolved in resolve_sta_flags so explicit or_template
    # ablations on sta_* settings are possible.
    parser.add_argument('--sta_attach', type=str, default=None,
        choices=['or_template', 'anchor_soft', 'anchor_hard', 'none'],
        help='Synthetic attachment mode. Default: or_template for GraphSMOTE, anchor_soft for sta_*.')
    parser.add_argument('--sta_candidate', type=str, default='anchor',
        choices=['anchor', 'anchor_nn', 'or_template', 'all'],
        help='Candidate neighbor set for STA masks.')
    parser.add_argument('--sta_edge_mode', type=str, default=None,
        choices=['soft', 'hard'],
        help='Soft decoder probs vs hard threshold inside STA candidate mask.')
    parser.add_argument('--sta_threshold', type=float, default=0.5,
        help='Hard threshold for STA edges when sta_edge_mode=hard.')
    parser.add_argument('--sta_temp', type=float, default=1.0,
        help='Temperature for soft STA edge scores (sigmoid(z/temp)).')
    parser.add_argument('--sta_top_k', type=int, default=0,
        help='Keep top-k decoder scores within candidates; 0 = all.')
    parser.add_argument('--sta_gate_nn', action='store_true', default=False,
        help='Optionally gate in NN-partner neighbors (train-label agreement only).')
    parser.add_argument('--sta_gate_thresh', type=float, default=0.5,
        help='Min train-label agreement to admit NN neighborhood under sta_gate_nn.')
    parser.add_argument('--config_out', type=str, default=None,
        help='Optional path to write exact run config JSON.')

    # ---- GraphSMOTE + TAM (Song et al., ICML 2022) — additive; default OFF ----
    # Controlled hybrid: does not invent TAM; TAM off leaves GraphSMOTE CE unchanged.
    parser.add_argument('--tam', action='store_true', default=False,
        help='Enable Topology-Aware Margin (Song et al., ICML 2022) on classifier CE. Off = unchanged GraphSMOTE path.')
    parser.add_argument('--tam_alpha', type=float, default=2.5,
        help='ACM coefficient α (official TAM default 2.5).')
    parser.add_argument('--tam_beta', type=float, default=0.5,
        help='ADM coefficient β (official TAM default 0.5).')
    parser.add_argument('--tam_phi', type=float, default=1.2,
        help='Class-wise temperature φ / temp_phi (official default 1.2).')
    parser.add_argument('--tam_delta', type=float, default=0.4,
        help='Class-count mix γ / temp_gamma (official hardcoded 0.4).')
    parser.add_argument('--tam_warmup', type=int, default=5,
        help='Warmup epochs before TAM margins (official warmup=5; active when epoch+1 > warmup).')
    parser.add_argument('--tam_scope', type=str, default='train_orig',
        choices=['train_orig', 'train_all'],
        help='Which nodes receive TAM margins. Default train_orig = original labeled train only.')



    return parser


def resolve_sta_flags(args):
    """Apply setting→STA defaults without altering baseline settings when STA is off."""
    if args.sta_attach is None:
        if args.setting in ('sta_newG_cls', 'sta_recon_newG'):
            args.sta_attach = 'anchor_soft'
        else:
            args.sta_attach = 'or_template'
    if args.sta_edge_mode is None:
        if args.sta_attach == 'anchor_hard':
            args.sta_edge_mode = 'hard'
        else:
            args.sta_edge_mode = 'soft'
    return args


def sta_enabled(args):
    # Explicit STA settings, or GraphSMOTE settings with STA attach override (ablations).
    return args.setting in ('sta_newG_cls', 'sta_recon_newG') or args.sta_attach in ('anchor_soft', 'anchor_hard')


def log_sta_hyperparams(args):
    cfg = {
        'setting': args.setting,
        'sta_attach': args.sta_attach,
        'sta_candidate': args.sta_candidate,
        'sta_edge_mode': args.sta_edge_mode,
        'sta_threshold': args.sta_threshold,
        'sta_temp': args.sta_temp,
        'sta_top_k': args.sta_top_k,
        'sta_gate_nn': bool(args.sta_gate_nn),
        'sta_gate_thresh': args.sta_gate_thresh,
        'tam': bool(getattr(args, 'tam', False)),
        'tam_alpha': getattr(args, 'tam_alpha', 2.5),
        'tam_beta': getattr(args, 'tam_beta', 0.5),
        'tam_phi': getattr(args, 'tam_phi', 1.2),
        'tam_delta': getattr(args, 'tam_delta', 0.4),
        'tam_warmup': getattr(args, 'tam_warmup', 5),
        'tam_scope': getattr(args, 'tam_scope', 'train_orig'),
        'seed': args.seed,
        'dataset': args.dataset,
        'epochs': args.epochs,
        'lr': args.lr,
        'dropout': args.dropout,
        'up_scale': args.up_scale,
        'im_ratio': args.im_ratio,
        'rec_weight': args.rec_weight,
        'model': args.model,
        'opt_new_G': bool(args.opt_new_G),
        'load': args.load,
        'imbalance': bool(args.imbalance),
        'attribution': (
            'GraphSMOTE=Zhao et al. WSDM 2021; '
            'STA=GraphSMOTE-Plus extension (null); '
            'TAM=Song et al. ICML 2022 (hybrid, not novelty claim)'
        ),
    }
    print('[CONFIG]', json.dumps(cfg, sort_keys=True))
    if args.tam:
        print('[TAM/CONFIG]', json.dumps({
            'tam': True,
            'tam_alpha': args.tam_alpha,
            'tam_beta': args.tam_beta,
            'tam_phi': args.tam_phi,
            'tam_delta': args.tam_delta,
            'tam_warmup': args.tam_warmup,
            'tam_scope': args.tam_scope,
            'source': 'Song et al. ICML 2022 / github.com/Jaeyun-Song/TAM',
        }, sort_keys=True))
    if args.config_out:
        os.makedirs(os.path.dirname(args.config_out) or '.', exist_ok=True)
        with open(args.config_out, 'w') as f:
            json.dump(cfg, f, indent=2, sort_keys=True)
    return cfg

def split_arti(labels, c_train_num):
    #labels: n-dim Longtensor, each element in [0,...,m-1].
    #cora: m=7
    num_classes = len(set(labels.tolist()))
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)
    c_num_mat[:,1] = 25
    c_num_mat[:,2] = 55

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        print('{:d}-th class sample number: {:d}'.format(i,len(c_idx)))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        train_idx = train_idx + c_idx[:c_train_num[i]]
        c_num_mat[i,0] = c_train_num[i]

        val_idx = val_idx + c_idx[c_train_num[i]:c_train_num[i]+25]
        test_idx = test_idx + c_idx[c_train_num[i]+25:c_train_num[i]+80]

    random.shuffle(train_idx)

    #ipdb.set_trace()

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)
    #c_num_mat = torch.LongTensor(c_num_mat)


    return train_idx, val_idx, test_idx, c_num_mat

def split_genuine(labels):
    #labels: n-dim Longtensor, each element in [0,...,m-1].
    #cora: m=7
    num_classes = len(set(labels.tolist()))
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        c_num = len(c_idx)
        print('{:d}-th class sample number: {:d}'.format(i,len(c_idx)))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        if c_num <4:
            if c_num < 3:
                print("too small class type")
                ipdb.set_trace()
            c_num_mat[i,0] = 1
            c_num_mat[i,1] = 1
            c_num_mat[i,2] = 1
        else:
            c_num_mat[i,0] = int(c_num/4)
            c_num_mat[i,1] = int(c_num/4)
            c_num_mat[i,2] = int(c_num/2)


        train_idx = train_idx + c_idx[:c_num_mat[i,0]]

        val_idx = val_idx + c_idx[c_num_mat[i,0]:c_num_mat[i,0]+c_num_mat[i,1]]
        test_idx = test_idx + c_idx[c_num_mat[i,0]+c_num_mat[i,1]:c_num_mat[i,0]+c_num_mat[i,1]+c_num_mat[i,2]]

    random.shuffle(train_idx)

    #ipdb.set_trace()

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)
    #c_num_mat = torch.LongTensor(c_num_mat)


    return train_idx, val_idx, test_idx, c_num_mat


def print_edges_num(dense_adj, labels):
    c_num = labels.max().item()+1
    dense_adj = np.array(dense_adj)
    labels = np.array(labels)

    for i in range(c_num):
        for j in range(c_num):
            #ipdb.set_trace()
            row_ind = labels == i
            col_ind = labels == j

            edge_num = dense_adj[row_ind].transpose()[col_ind].sum()
            print("edges between class {:d} and class {:d}: {:f}".format(i,j,edge_num))


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def print_class_acc(output, labels, class_num_list, pre='valid', im_class_num=3):
    pre_num = 0
    #print class-wise performance
    '''
    for i in range(labels.max()+1):
        
        cur_tpr = accuracy(output[pre_num:pre_num+class_num_list[i]], labels[pre_num:pre_num+class_num_list[i]])
        print(str(pre)+" class {:d} True Positive Rate: {:.3f}".format(i,cur_tpr.item()))

        index_negative = labels != i
        labels_negative = labels.new(labels.shape).fill_(i)
        
        cur_fpr = accuracy(output[index_negative,:], labels_negative[index_negative])
        print(str(pre)+" class {:d} False Positive Rate: {:.3f}".format(i,cur_fpr.item()))

        pre_num = pre_num + class_num_list[i]
    '''

    #ipdb.set_trace()
    y_true = labels.detach().cpu().numpy()
    y_pred = torch.argmax(output, dim=-1).detach().cpu().numpy()
    probs = F.softmax(output, dim=-1).detach().cpu().numpy()

    if labels.max() > 1:
        auc_score = roc_auc_score(y_true, probs, average='macro', multi_class='ovr')
    else:
        auc_score = roc_auc_score(y_true, probs[:,1], average='macro')

    macro_F = f1_score(y_true, y_pred, average='macro')
    print(str(pre)+' current auc-roc score: {:f}, current macro_F score: {:f}'.format(auc_score,macro_F))

    # Extended metrics (logging only; does not affect training).
    # Minority = last im_class_num classes (GraphSMOTE artificial-imbalance convention).
    c_largest = int(labels.max().item())
    minority_classes = list(range(c_largest - im_class_num + 1, c_largest + 1))
    min_mask = np.isin(y_true, minority_classes)
    metrics = {
        'auc_roc': float(auc_score),
        'macro_f1': float(macro_F),
        'accuracy': float((y_pred == y_true).mean()) if len(y_true) else float('nan'),
        'balanced_accuracy': float('nan'),
        'minority_f1': float('nan'),
        'minority_recall': float('nan'),
        'pr_auc': float('nan'),
    }
    try:
        metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        pass
    if min_mask.any():
        # Binary view: minority vs rest for PR-AUC; per-minority macro for F1/recall.
        try:
            metrics['minority_f1'] = float(
                f1_score(y_true, y_pred, labels=minority_classes, average='macro', zero_division=0)
            )
            metrics['minority_recall'] = float(
                recall_score(y_true, y_pred, labels=minority_classes, average='macro', zero_division=0)
            )
        except Exception:
            pass
        try:
            # Macro PR-AUC over classes present in y_true (OvR average precision).
            present = np.unique(y_true)
            aps = []
            for c in present:
                y_bin = (y_true == c).astype(int)
                aps.append(average_precision_score(y_bin, probs[:, c]))
            metrics['pr_auc'] = float(np.mean(aps)) if aps else float('nan')
        except Exception:
            pass
    print(
        str(pre)
        + ' extended metrics: bal_acc={:.6f}, minority_f1={:.6f}, minority_recall={:.6f}, pr_auc={:.6f}'.format(
            metrics['balanced_accuracy'],
            metrics['minority_f1'],
            metrics['minority_recall'],
            metrics['pr_auc'],
        )
    )
    return metrics

def src_upsample(adj,features,labels,idx_train, portion=1.0, im_class_num=3):
    c_largest = labels.max().item()
    adj_back = adj.to_dense()
    chosen = None

    #ipdb.set_trace()
    avg_number = int(idx_train.shape[0]/(c_largest+1))

    for i in range(im_class_num):
        new_chosen = idx_train[(labels==(c_largest-i))[idx_train]]
        if portion == 0:#refers to even distribution
            c_portion = int(avg_number/new_chosen.shape[0])

            for j in range(c_portion):
                if chosen is None:
                    chosen = new_chosen
                else:
                    chosen = torch.cat((chosen, new_chosen), 0)

        else:
            c_portion = int(portion)
            portion_rest = portion-c_portion
            for j in range(c_portion):
                num = int(new_chosen.shape[0])
                new_chosen = new_chosen[:num]

                if chosen is None:
                    chosen = new_chosen
                else:
                    chosen = torch.cat((chosen, new_chosen), 0)
            
            num = int(new_chosen.shape[0]*portion_rest)
            new_chosen = new_chosen[:num]

            if chosen is None:
                chosen = new_chosen
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
            

    add_num = chosen.shape[0]
    new_adj = adj_back.new(torch.Size((adj_back.shape[0]+add_num, adj_back.shape[0]+add_num)))
    new_adj[:adj_back.shape[0], :adj_back.shape[0]] = adj_back[:,:]
    new_adj[adj_back.shape[0]:, :adj_back.shape[0]] = adj_back[chosen,:]
    new_adj[:adj_back.shape[0], adj_back.shape[0]:] = adj_back[:,chosen]
    new_adj[adj_back.shape[0]:, adj_back.shape[0]:] = adj_back[chosen,:][:,chosen]

    #ipdb.set_trace()
    features_append = deepcopy(features[chosen,:])
    labels_append = deepcopy(labels[chosen])
    idx_new = np.arange(adj_back.shape[0], adj_back.shape[0]+add_num)
    idx_train_append = idx_train.new(idx_new)

    features = torch.cat((features,features_append), 0)
    labels = torch.cat((labels,labels_append), 0)
    idx_train = torch.cat((idx_train,idx_train_append), 0)
    adj = new_adj.to_sparse()

    return adj, features, labels, idx_train



def src_smote(adj,features,labels,idx_train, portion=1.0, im_class_num=3):
    c_largest = labels.max().item()
    adj_back = adj.to_dense()
    chosen = None
    new_features = None

    #ipdb.set_trace()
    avg_number = int(idx_train.shape[0]/(c_largest+1))

    for i in range(im_class_num):
        new_chosen = idx_train[(labels==(c_largest-i))[idx_train]]
        if portion == 0:#refers to even distribution
            c_portion = int(avg_number/new_chosen.shape[0])

            portion_rest = (avg_number/new_chosen.shape[0]) - c_portion

        else:
            c_portion = int(portion)
            portion_rest = portion-c_portion
            
        for j in range(c_portion):
            num = int(new_chosen.shape[0])
            new_chosen = new_chosen[:num]

            chosen_embed = features[new_chosen,:]
            distance = squareform(pdist(chosen_embed.cpu().detach()))
            np.fill_diagonal(distance,distance.max()+100)

            idx_neighbor = distance.argmin(axis=-1)
            
            interp_place = random.random()
            embed = chosen_embed + (chosen_embed[idx_neighbor,:]-chosen_embed)*interp_place

            if chosen is None:
                chosen = new_chosen
                new_features = embed
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
                new_features = torch.cat((new_features, embed),0)
            
        num = int(new_chosen.shape[0]*portion_rest)
        new_chosen = new_chosen[:num]

        chosen_embed = features[new_chosen,:]
        distance = squareform(pdist(chosen_embed.cpu().detach()))
        np.fill_diagonal(distance,distance.max()+100)

        idx_neighbor = distance.argmin(axis=-1)
            
        interp_place = random.random()
        embed = chosen_embed + (chosen_embed[idx_neighbor,:]-chosen_embed)*interp_place

        if chosen is None:
            chosen = new_chosen
            new_features = embed
        else:
            chosen = torch.cat((chosen, new_chosen), 0)
            new_features = torch.cat((new_features, embed),0)
            

    add_num = chosen.shape[0]
    new_adj = adj_back.new(torch.Size((adj_back.shape[0]+add_num, adj_back.shape[0]+add_num)))
    new_adj[:adj_back.shape[0], :adj_back.shape[0]] = adj_back[:,:]
    new_adj[adj_back.shape[0]:, :adj_back.shape[0]] = adj_back[chosen,:]
    new_adj[:adj_back.shape[0], adj_back.shape[0]:] = adj_back[:,chosen]
    new_adj[adj_back.shape[0]:, adj_back.shape[0]:] = adj_back[chosen,:][:,chosen]

    #ipdb.set_trace()
    features_append = deepcopy(new_features)
    labels_append = deepcopy(labels[chosen])
    idx_new = np.arange(adj_back.shape[0], adj_back.shape[0]+add_num)
    idx_train_append = idx_train.new(idx_new)

    features = torch.cat((features,features_append), 0)
    labels = torch.cat((labels,labels_append), 0)
    idx_train = torch.cat((idx_train,idx_train_append), 0)
    adj = new_adj.to_sparse()

    return adj, features, labels, idx_train

def _binary_neighbor_row(adj_row):
    """Binary 1-hop support from a (possibly weighted/normalized) adjacency row."""
    return (adj_row > 0).to(adj_row.dtype)


def _train_label_agreement(adj, labels, idx_train, node_idx, target_class):
    """Fraction of *labeled-train* neighbors matching target_class. No val/test labels."""
    row = adj[node_idx]
    neigh = (row > 0).nonzero(as_tuple=False).view(-1)
    if neigh.numel() == 0:
        return 0.0
    train_set = set(idx_train.detach().cpu().tolist())
    agree = 0
    total = 0
    labels_cpu = labels.detach().cpu()
    for n in neigh.detach().cpu().tolist():
        if n in train_set:
            total += 1
            if int(labels_cpu[n].item()) == int(target_class):
                agree += 1
    if total == 0:
        return 0.0
    return agree / float(total)


def _build_sta_candidate_rows(adj, chosen, idx_neighbor, labels, idx_train,
                              sta_candidate='anchor', sta_gate_nn=False, sta_gate_thresh=0.5):
    """
    Build per-synthetic candidate masks over original nodes (shape |S| x N).
    Leakage rule: anchors/NN from train minority only; gate uses train labels only.
    """
    n_orig = adj.shape[0]
    rows = []
    # idx_neighbor: local argmin over `chosen` (numpy ints) → NN partner node id = chosen[local_nn]
    for local_i in range(chosen.shape[0]):
        va = chosen[local_i]
        vn = chosen[int(idx_neighbor[local_i])]
        if sta_candidate == 'all':
            mask = adj.new(torch.Size((n_orig,))).fill_(1.0)
        else:
            mask = adj.new(torch.Size((n_orig,))).fill_(0.0)
            # anchor neighbors (default STA)
            if sta_candidate in ('anchor', 'anchor_nn', 'or_template'):
                mask = mask + _binary_neighbor_row(adj[va])
            # NN-partner neighbors (ablation / legacy OR)
            if sta_candidate in ('anchor_nn', 'or_template'):
                mask = mask + _binary_neighbor_row(adj[vn])
            # Optional gated NN under pure anchor mode (train-label agreement only)
            if sta_candidate == 'anchor' and sta_gate_nn:
                target_c = int(labels[va].item())
                agree = _train_label_agreement(adj, labels, idx_train, vn, target_c)
                if agree >= sta_gate_thresh:
                    mask = mask + _binary_neighbor_row(adj[vn])
            mask = torch.clamp(mask, min=0.0, max=1.0)
        rows.append(mask.unsqueeze(0))
    return torch.cat(rows, dim=0)

def recon_upsample(embed, labels, idx_train, adj=None, portion=1.0, im_class_num=3,
                   sta_attach='or_template', sta_candidate='anchor',
                   sta_gate_nn=False, sta_gate_thresh=0.5, sta_top_k=0):
    """
    GraphSMOTE embedding-space SMOTE (+ optional adjacency template).

    When sta_attach == 'or_template' (default): identical to Zhao et al. GraphSMOTE
    seed∨NN OR-template. STA modes build anchor-local (or ablation) candidate masks.
    """
    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0]/(c_largest+1))
    #ipdb.set_trace()
    adj_new = None
    use_sta_mask = (adj is not None) and (sta_attach in ('anchor_soft', 'anchor_hard'))

    # Sanity: embedding dims
    assert embed.dim() == 2, 'embed must be [N, D], got {}'.format(tuple(embed.shape))
    if adj is not None:
        assert adj.dim() == 2 and adj.shape[0] == adj.shape[1], 'adj must be square'
        assert adj.shape[0] == embed.shape[0], 'adj/embed size mismatch {} vs {}'.format(
            adj.shape[0], embed.shape[0])
    if torch.isnan(embed).any():
        raise ValueError('NaN detected in embed before recon_upsample')

    for i in range(im_class_num):
        chosen = idx_train[(labels==(c_largest-i))[idx_train]]
        num = int(chosen.shape[0]*portion)
        if portion == 0:
            c_portion = int(avg_number/chosen.shape[0])
            num = chosen.shape[0]
        else:
            c_portion = 1

        for j in range(c_portion):
            chosen = chosen[:num]

            chosen_embed = embed[chosen,:]
            distance = squareform(pdist(chosen_embed.cpu().detach()))
            np.fill_diagonal(distance,distance.max()+100)

            idx_neighbor = distance.argmin(axis=-1)
            
            interp_place = random.random()
            new_embed = embed[chosen,:] + (chosen_embed[idx_neighbor,:]-embed[chosen,:])*interp_place

            if torch.isnan(new_embed).any():
                raise ValueError('NaN in synthetic embeddings (class {})'.format(c_largest-i))

            new_labels = labels.new(torch.Size((chosen.shape[0],1))).reshape(-1).fill_(c_largest-i)
            idx_new = np.arange(embed.shape[0], embed.shape[0]+chosen.shape[0])
            idx_train_append = idx_train.new(idx_new)

            embed = torch.cat((embed,new_embed), 0)
            labels = torch.cat((labels,new_labels), 0)
            idx_train = torch.cat((idx_train,idx_train_append), 0)

            if adj is not None:
                if use_sta_mask:
                    # STA: candidate mask only (decoder scores applied later in main).
                    # Map local NN indices → node ids via chosen[idx_neighbor].
                    temp = _build_sta_candidate_rows(
                        adj, chosen, idx_neighbor, labels, idx_train,
                        sta_candidate=sta_candidate,
                        sta_gate_nn=sta_gate_nn,
                        sta_gate_thresh=sta_gate_thresh,
                    )
                    # Optional top-k is applied later when decoder scores are known;
                    # store binary candidates here. top-k handled in apply_sta_attachment.
                else:
                    # Original GraphSMOTE OR-template (Zhao et al., WSDM 2021).
                    temp = adj.new(torch.clamp_(adj[chosen,:] + adj[idx_neighbor,:], min=0.0, max = 1.0))
                if adj_new is None:
                    adj_new = temp
                else:
                    adj_new = torch.cat((adj_new, temp), 0)

    if adj is not None:
        add_num = adj_new.shape[0]
        new_adj = adj.new(torch.Size((adj.shape[0]+add_num, adj.shape[0]+add_num))).fill_(0.0)
        new_adj[:adj.shape[0], :adj.shape[0]] = adj[:,:]
        new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:,:]
        new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:,:]

        return embed, labels, idx_train, new_adj.detach()

    else:
        return embed, labels, idx_train


def apply_sta_attachment(generated_G, adj_up, ori_num, adj_true_dense,
                         sta_edge_mode='soft', sta_threshold=0.5, sta_temp=1.0,
                         sta_top_k=0, opt_new_G=False):
    """
    Combine decoder scores with STA candidate mask in adj_up.
    Returns classifier adjacency (synthetic block masked/scored; real block = true A).
    """
    scores = generated_G
    if sta_temp is not None and abs(sta_temp - 1.0) > 1e-8:
        # generated_G is already sigmoid(z); recover soft re-temperature approximately
        # via power transform on probabilities: p^(1/temp) then renormalize per-row
        # among candidates only (done after mask). Safer: clamp and power.
        eps = 1e-6
        scores = torch.clamp(generated_G, min=eps, max=1.0 - eps)
        # logit / temp then sigmoid
        logits = torch.log(scores) - torch.log1p(-scores)
        scores = torch.sigmoid(logits / sta_temp)

    if not opt_new_G:
        adj_new = scores.detach().clone()
    else:
        adj_new = scores

    # Candidate mask from recon_upsample (binary on synthetic rows).
    mask = adj_up.clone()
    # Restrict synthetic→original part for top-k
    if sta_top_k is not None and sta_top_k > 0:
        syn = adj_new[ori_num:, :ori_num]
        syn_mask = mask[ori_num:, :ori_num]
        # zero outside candidates first for ranking
        ranked = syn * syn_mask
        for r in range(ranked.shape[0]):
            row = ranked[r]
            nnz = (syn_mask[r] > 0).sum().item()
            k = min(sta_top_k, int(nnz)) if nnz > 0 else 0
            if k <= 0:
                syn_mask[r] = 0
                continue
            topv, topi = torch.topk(row, k)
            new_row_mask = torch.zeros_like(syn_mask[r])
            new_row_mask[topi] = 1.0
            syn_mask[r] = new_row_mask
        mask[ori_num:, :ori_num] = syn_mask
        mask[:ori_num, ori_num:] = syn_mask.transpose(0, 1)

    if sta_edge_mode == 'hard':
        hard = adj_new.clone()
        hard[hard < sta_threshold] = 0.0
        hard[hard >= sta_threshold] = 1.0
        adj_new = hard

    # Edge accuracy diagnostic on original block (same spirit as GraphSMOTE logs)
    with torch.no_grad():
        hard_ref = adj_new[:ori_num, :ori_num].clone()
        hard_ref[hard_ref < 0.5] = 0.0
        hard_ref[hard_ref >= 0.5] = 1.0
        edge_ac = hard_ref.eq(adj_true_dense).double().sum() / (ori_num ** 2)

    exist_edge_prob = adj_new[:ori_num, :ori_num].mean()
    generated_edge_prob = adj_new[ori_num:, :ori_num].mean()
    print("edge acc: {:.4f}, exist_edge_prob: {:.4f}, generated_edge_prob: {:.4f}".format(
        edge_ac.item(), exist_edge_prob.item(), generated_edge_prob.item()))

    adj_new = torch.mul(mask, adj_new)

    exist_edge_prob = adj_new[:ori_num, :ori_num].mean()
    generated_edge_prob = adj_new[ori_num:, :ori_num].mean()
    print("after filtering, edge acc: {:.4f}, exist_edge_prob: {:.4f}, generated_edge_prob: {:.4f}".format(
        edge_ac.item(), exist_edge_prob.item(), generated_edge_prob.item()))

    # Restore true edges on real–real block (GraphSMOTE convention).
    adj_new[:ori_num, :][:, :ori_num] = adj_true_dense

    if not opt_new_G:
        adj_new = adj_new.detach()

    if torch.isnan(adj_new).any():
        raise ValueError('NaN in STA-augmented adjacency')

    return adj_new

def adj_mse_loss(adj_rec, adj_tgt, adj_mask = None):
    edge_num = adj_tgt.nonzero().shape[0]
    total_num = adj_tgt.shape[0]**2

    neg_weight = edge_num / (total_num-edge_num)

    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt==0] = neg_weight

    loss = torch.sum(weight_matrix * (adj_rec - adj_tgt) ** 2)

    return loss




