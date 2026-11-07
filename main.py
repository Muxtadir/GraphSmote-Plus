import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import models
import utils
import data_load
import tam as tam_lib
import random
import ipdb
import copy

#from torch.utils.tensorboard import SummaryWriter

# Training setting
parser = utils.get_parser()

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args = utils.resolve_sta_flags(args)

# Seeding (uncommented for multi-seed reproduction / STA fairness vs baseline CSV).
# Same four lines as upstream GraphSMOTE; does not add cudnn.deterministic flags.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

utils.log_sta_hyperparams(args)
# Load data
if args.dataset == 'cora':
    adj, features, labels = data_load.load_data()
    class_sample_num = 20
    im_class_num = 3
elif args.dataset == 'BlogCatalog':
    adj, features, labels = data_load.load_data_Blog()
    im_class_num = 14 #set it to be the number less than 100
    class_sample_num = 20 #not used
elif args.dataset == 'twitter':
    adj, features, labels = data_load.load_sub_data_twitter()
    im_class_num = 1
    class_sample_num = 20 #not used
else:
    print("no this dataset: {args.dataset}")


#for artificial imbalanced setting: only the last im_class_num classes are imbalanced
c_train_num = []
for i in range(labels.max().item() + 1):
    if args.imbalance and i > labels.max().item()-im_class_num: #only imbalance the last classes
        c_train_num.append(int(class_sample_num*args.im_ratio))

    else:
        c_train_num.append(class_sample_num)

#get train, validatio, test data split
if args.dataset == 'BlogCatalog':
    idx_train, idx_val, idx_test, class_num_mat = utils.split_genuine(labels)
elif args.dataset == 'cora':
    idx_train, idx_val, idx_test, class_num_mat = utils.split_arti(labels, c_train_num)
elif args.dataset == 'twitter':
    idx_train, idx_val, idx_test, class_num_mat = utils.split_genuine(labels)

#method_1: oversampling in input domain
if args.setting == 'upsampling':
    adj,features,labels,idx_train = utils.src_upsample(adj,features,labels,idx_train,portion=args.up_scale, im_class_num=im_class_num)
if args.setting == 'smote':
    adj,features,labels,idx_train = utils.src_smote(adj,features,labels,idx_train,portion=args.up_scale, im_class_num=im_class_num)


# Model and optimizer
#if oversampling in the embedding space is required, model need to be changed
if args.setting != 'embed_up':
    if args.model == 'sage':
        encoder = models.Sage_En(nfeat=features.shape[1],
                nhid=args.nhid,
                nembed=args.nhid,
                dropout=args.dropout)
        classifier = models.Sage_Classifier(nembed=args.nhid, 
                nhid=args.nhid, 
                nclass=labels.max().item() + 1, 
                dropout=args.dropout)
    elif args.model == 'gcn':
        encoder = models.GCN_En(nfeat=features.shape[1],
                nhid=args.nhid,
                nembed=args.nhid,
                dropout=args.dropout)
        classifier = models.GCN_Classifier(nembed=args.nhid, 
                nhid=args.nhid, 
                nclass=labels.max().item() + 1, 
                dropout=args.dropout)
    elif args.model == 'GAT':
        encoder = models.GAT_En(nfeat=features.shape[1],
                nhid=args.nhid,
                nembed=args.nhid,
                dropout=args.dropout)
        classifier = models.GAT_Classifier(nembed=args.nhid, 
                nhid=args.nhid, 
                nclass=labels.max().item() + 1, 
                dropout=args.dropout)
else:
    if args.model == 'sage':
        encoder = models.Sage_En2(nfeat=features.shape[1],
                nhid=args.nhid,
                nembed=args.nhid,
                dropout=args.dropout)
        classifier = models.Classifier(nembed=args.nhid, 
                nhid=args.nhid, 
                nclass=labels.max().item() + 1, 
                dropout=args.dropout)
    elif args.model == 'gcn':
        encoder = models.GCN_En2(nfeat=features.shape[1],
                nhid=args.nhid,
                nembed=args.nhid,
                dropout=args.dropout)
        classifier = models.Classifier(nembed=args.nhid, 
                nhid=args.nhid, 
                nclass=labels.max().item() + 1, 
                dropout=args.dropout)
    elif args.model == 'GAT':
        encoder = models.GAT_En2(nfeat=features.shape[1],
                nhid=args.nhid,
                nembed=args.nhid,
                dropout=args.dropout)
        classifier = models.Classifier(nembed=args.nhid, 
                nhid=args.nhid, 
                nclass=labels.max().item() + 1, 
                dropout=args.dropout)



decoder = models.Decoder(nembed=args.nhid,
        dropout=args.dropout)


optimizer_en = optim.Adam(encoder.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
optimizer_cls = optim.Adam(classifier.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
optimizer_de = optim.Adam(decoder.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)



if args.cuda:
    encoder = encoder.cuda()
    classifier = classifier.cuda()
    decoder = decoder.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    encoder.train()
    classifier.train()
    decoder.train()
    optimizer_en.zero_grad()
    optimizer_cls.zero_grad()
    optimizer_de.zero_grad()

    embed = encoder(features, adj)

    # GraphSMOTE edge-generation settings + GraphSMOTE-STA counterparts (additive).
    gs_edge_settings = ('recon_newG', 'recon', 'newG_cls', 'sta_recon_newG', 'sta_newG_cls')
    if args.setting in gs_edge_settings:
        ori_num = labels.shape[0]
        adj_dense = adj.detach().to_dense()
        # STA only when explicitly enabled; baseline path keeps sta_attach=or_template.
        embed, labels_new, idx_train_new, adj_up = utils.recon_upsample(
            embed, labels, idx_train, adj=adj_dense, portion=args.up_scale,
            im_class_num=im_class_num,
            sta_attach=args.sta_attach,
            sta_candidate=args.sta_candidate,
            sta_gate_nn=args.sta_gate_nn,
            sta_gate_thresh=args.sta_gate_thresh,
            sta_top_k=args.sta_top_k,
        )
        generated_G = decoder(embed)

        loss_rec = utils.adj_mse_loss(generated_G[:ori_num, :][:, :ori_num], adj_dense)
        
        #ipdb.set_trace()

        if utils.sta_enabled(args) and args.sta_attach in ('anchor_soft', 'anchor_hard'):
            # GraphSMOTE-STA: soft/hard scores inside anchor-local (or ablation) mask.
            adj_new = utils.apply_sta_attachment(
                generated_G, adj_up, ori_num, adj_dense,
                sta_edge_mode=args.sta_edge_mode,
                sta_threshold=args.sta_threshold,
                sta_temp=args.sta_temp,
                sta_top_k=args.sta_top_k,
                opt_new_G=args.opt_new_G,
            )
        else:
            # Original GraphSMOTE attachment (Zhao et al., WSDM 2021) — unchanged.
            if not args.opt_new_G:
                adj_new = copy.deepcopy(generated_G.detach())
                threshold = 0.5
                adj_new[adj_new<threshold] = 0.0
                adj_new[adj_new>=threshold] = 1.0

                #ipdb.set_trace()
                edge_ac = adj_new[:ori_num, :ori_num].eq(adj.to_dense()).double().sum()/(ori_num**2)
            else:
                adj_new = generated_G
                edge_ac = F.l1_loss(adj_new[:ori_num, :ori_num], adj.to_dense(), reduction='mean')


            #calculate generation information
            exist_edge_prob = adj_new[:ori_num, :ori_num].mean() #edge prob for existing nodes
            generated_edge_prob = adj_new[ori_num:, :ori_num].mean() #edge prob for generated nodes
            print("edge acc: {:.4f}, exist_edge_prob: {:.4f}, generated_edge_prob: {:.4f}".format(edge_ac.item(), exist_edge_prob.item(), generated_edge_prob.item()))


            adj_new = torch.mul(adj_up, adj_new)

            exist_edge_prob = adj_new[:ori_num, :ori_num].mean() #edge prob for existing nodes
            generated_edge_prob = adj_new[ori_num:, :ori_num].mean() #edge prob for generated nodes
            print("after filtering, edge acc: {:.4f}, exist_edge_prob: {:.4f}, generated_edge_prob: {:.4f}".format(edge_ac.item(), exist_edge_prob.item(), generated_edge_prob.item()))


            adj_new[:ori_num, :][:, :ori_num] = adj.detach().to_dense()
            #adj_new = adj_new.to_sparse()
            #ipdb.set_trace()

            if not args.opt_new_G:
                adj_new = adj_new.detach()

        # Match GraphSMOTE quirk: newG_cls / sta_newG_cls CE uses original idx_train only.
        if args.setting in ('newG_cls', 'sta_newG_cls'):
            idx_train_new = idx_train

    elif args.setting == 'embed_up':
        #perform SMOTE in embedding space
        embed, labels_new, idx_train_new = utils.recon_upsample(embed, labels, idx_train,portion=args.up_scale, im_class_num=im_class_num)
        adj_new = adj
    else:
        labels_new = labels
        idx_train_new = idx_train
        adj_new = adj

    #ipdb.set_trace()
    output = classifier(embed, adj_new)

    # Classifier CE logits: optional TAM (Song et al., ICML 2022). Off ⇒ identical to GraphSMOTE.
    # Never apply during decoder pretrain (`recon`). Val/test logging still uses raw `output`.
    logits_ce = output
    if args.tam and args.setting != 'recon':
        class_num_list = tam_lib.train_class_counts(labels, idx_train)
        # Original-graph topology for NLD (not the augmented classifier graph).
        adj_for_tam = adj.detach()
        if adj_for_tam.is_sparse:
            adj_for_tam = adj_for_tam.to_dense()
        logits_ce = tam_lib.adjust_logits_tam(
            output,
            adj_for_tam,
            labels_new if labels_new.shape[0] == output.shape[0] else labels,
            idx_train,
            class_num_list=class_num_list,
            epoch=epoch,
            tam_alpha=args.tam_alpha,
            tam_beta=args.tam_beta,
            tam_phi=args.tam_phi,
            tam_delta=args.tam_delta,
            tam_warmup=args.tam_warmup,
            tam_scope=args.tam_scope,
            idx_train_new=idx_train_new,
            n_orig=labels.shape[0],
        )

    if args.setting == 'reweight':
        weight = features.new((labels.max().item()+1)).fill_(1)
        weight[-im_class_num:] = 1+args.up_scale
        loss_train = F.cross_entropy(logits_ce[idx_train_new], labels_new[idx_train_new], weight=weight)
    else:
        loss_train = F.cross_entropy(logits_ce[idx_train_new], labels_new[idx_train_new])

    acc_train = utils.accuracy(output[idx_train], labels_new[idx_train])
    if args.setting in ('recon_newG', 'sta_recon_newG'):
        loss = loss_train+loss_rec*args.rec_weight
    elif args.setting == 'recon':
        loss = loss_rec + 0*loss_train
    else:
        loss = loss_train
        loss_rec = loss_train

    loss.backward()
    if args.setting in ('newG_cls', 'sta_newG_cls'):
        optimizer_en.zero_grad()
        optimizer_de.zero_grad()
    else:
        optimizer_en.step()

    optimizer_cls.step()

    if args.setting in ('recon_newG', 'sta_recon_newG', 'recon'):
        optimizer_de.step()

    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
    acc_val = utils.accuracy(output[idx_val], labels[idx_val])

    #ipdb.set_trace()
    utils.print_class_acc(output[idx_val], labels[idx_val], class_num_mat[:,1], im_class_num=im_class_num)
    print('Epoch: {:05d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_rec: {:.4f}'.format(loss_rec.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test(epoch = 0):
    encoder.eval()
    classifier.eval()
    decoder.eval()
    embed = encoder(features, adj)
    output = classifier(embed, adj)
    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = utils.accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    utils.print_class_acc(output[idx_test], labels[idx_test], class_num_mat[:,2], pre='test', im_class_num=im_class_num)

    '''
    if epoch==40:
        torch
    '''


def save_model(epoch):
    saved_content = {}

    saved_content['encoder'] = encoder.state_dict()
    saved_content['decoder'] = decoder.state_dict()
    saved_content['classifier'] = classifier.state_dict()

    # Baseline GraphSMOTE keeps original filename pattern. STA adds attach tag to avoid clobber.
    if utils.sta_enabled(args):
        torch.save(saved_content, 'checkpoint/{}/{}_{}_{}_{}_{}.pth'.format(
            args.dataset, args.setting, epoch, args.opt_new_G, args.im_ratio, args.sta_attach))
    else:
        torch.save(saved_content, 'checkpoint/{}/{}_{}_{}_{}.pth'.format(
            args.dataset, args.setting, epoch, args.opt_new_G, args.im_ratio))

    return

def load_model(filename):
    loaded_content = torch.load('checkpoint/{}/{}.pth'.format(args.dataset,filename), map_location=lambda storage, loc: storage)

    encoder.load_state_dict(loaded_content['encoder'])
    decoder.load_state_dict(loaded_content['decoder'])
    classifier.load_state_dict(loaded_content['classifier'])

    print("successfully loaded: "+ filename)

    return

# Train model
if args.load is not None:
    load_model(args.load)

t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)

    if epoch % 10 == 0:
        test(epoch)

    if epoch % 100 == 0:
        save_model(epoch)


print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
