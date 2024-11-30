# encoding:utf-8

import os
import numpy as np
from argparse import ArgumentParser
import datetime
import _pickle as cPickle
from sklearn.metrics.pairwise import cosine_similarity

import torch
from torch import optim
from models import *
from corpus import *
from utils import *
from test import *

parser = ArgumentParser()
arg = parser.add_argument
arg('--gpu', default=False)
arg('--data-dir', default='data/fb15k-237/')
arg('--emb-size', type=int, default=100)
arg('--learning-rate', type=float, default=0.001)
arg('--batch-size', type=int, default=128)
arg('--epoch-size', type=int, default=200)
arg('--model', type=str, default='transe')
arg('--IsTrain', type=bool, default=True)
arg('--display', type=bool, default=False)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.gpu
print("utilize gpu: ", os.environ["CUDA_VISIBLE_DEVICES"])
device = torch.device('cuda') if args.gpu else torch.device('cpu')
# device= torch.device('cpu')

def train_transe():
    if not os.path.exists(args.data_dir):
        print("data dir is empty")
        exit()

    dump_path = args.data_dir + "dump_data.pkl"
    if not os.path.exists(dump_path):
        corpus = Corpus()
        corpus.read_triples(args.data_dir)
        corpus.read_ent_types()
        cPickle.dump(corpus, open(dump_path, "wb"))
    else:
        corpus = cPickle.load(open(dump_path, "rb"))

    initial_lr = args.learning_rate

    # model = TransE(corpus.num_ent(), corpus.num_rel(), device,
    #                args.emb_size).cuda()
    model = TransE(corpus.num_ent(), corpus.num_rel(), device,
                                   args.emb_size)
    print(model)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=initial_lr)

    for epoch_id in range(1, args.epoch_size + 1):
        print("Starting epoch: ", epoch_id)
        loss_sum = 0
        model.train()
        last_time = datetime.datetime.now()

        batch_count = len(corpus.triples['train']) // args.batch_size
        for batch_id in range(1, batch_count + 1):

            ph_idx, pr_idx, pt_idx, nh_idx, nr_idx, nt_idx = corpus.next_batch(format="pos_neg")

            optimizer.zero_grad()

            # ph_idx_t = torch.tensor(ph_idx).cuda()
            # pr_idx_t = torch.tensor(pr_idx).cuda()
            # pt_idx_t = torch.tensor(pt_idx).cuda()
            # nh_idx_t = torch.tensor(nh_idx).cuda()
            # nr_idx_t = torch.tensor(nr_idx).cuda()
            # nt_idx_t = torch.tensor(nt_idx).cuda()
            ph_idx_t = torch.tensor(ph_idx)
            pr_idx_t = torch.tensor(pr_idx)
            pt_idx_t = torch.tensor(pt_idx)
            nh_idx_t = torch.tensor(nh_idx)
            nr_idx_t = torch.tensor(nr_idx)
            nt_idx_t = torch.tensor(nt_idx)

            curr_loss, _, _ = model(ph_idx_t, pr_idx_t, pt_idx_t,
                                    nh_idx_t, nr_idx_t, nt_idx_t)

            loss_sum += curr_loss.cpu().item()
            curr_loss.backward()
            optimizer.step()
        new_time = datetime.datetime.now()
        print("Loss: %0.2f, spend: %s" % (loss_sum, (new_time - last_time).seconds))

        if epoch_id % 50 == 0:
            model_dump_name = os.path.join(args.data_dir, 'transe-model-%d.pkl' % epoch_id)
            torch.save(model.state_dict(), model_dump_name)

    model_dump_name = os.path.join(args.data_dir, 'transe-model-final.pkl')
    torch.save(model.state_dict(), model_dump_name)


def train_rescal():
    if not os.path.exists(args.data_dir):
        print("data dir is empty")
        exit()

    dump_path = args.data_dir + "dump_data.pkl"
    if not os.path.exists(dump_path):
        corpus = Corpus()
        corpus.read_triples(args.data_dir)
        corpus.read_ent_types()
        cPickle.dump(corpus, open(dump_path, "wb"))
    else:
        corpus = cPickle.load(open(dump_path, "rb"))

    initial_lr = args.learning_rate

    # model = RESCAL(corpus.num_ent(), corpus.num_rel(), device,
    #                args.emb_size).cuda()
    model = RESCAL(corpus.num_ent(), corpus.num_rel(), device,
                   args.emb_size)
    print(model)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=initial_lr)

    for epoch_id in range(1, args.epoch_size + 1):
        print("Starting epoch: ", epoch_id)
        loss_sum = 0
        model.train()
        last_time = datetime.datetime.now()

        batch_count = len(corpus.triples['train']) // args.batch_size
        for batch_id in range(1, batch_count + 1):
            h_idx, r_idx, t_idx, labels = corpus.next_batch(format="triple_label", neg_ratio=10)

            optimizer.zero_grad()

            # h_idx_t = torch.tensor(h_idx).cuda()
            # r_idx_t = torch.tensor(r_idx).cuda()
            # t_idx_t = torch.tensor(t_idx).cuda()
            # labels_t = torch.tensor(labels).cuda()
            h_idx_t = torch.tensor(h_idx)
            r_idx_t = torch.tensor(r_idx)
            t_idx_t = torch.tensor(t_idx)
            labels_t = torch.tensor(labels)
            curr_loss, _ = model(h_idx_t, r_idx_t, t_idx_t, labels_t)

            loss_sum += curr_loss.cpu().item()
            curr_loss.backward()
            optimizer.step()
        new_time = datetime.datetime.now()
        print("Loss: %0.2f, spend: %s" % (loss_sum, (new_time - last_time).seconds))

        if epoch_id % 50 == 0:
            model_dump_name = os.path.join(args.data_dir, 'rescal-model-%d.pkl' % epoch_id)
            torch.save(model.state_dict(), model_dump_name)

    model_dump_name = os.path.join(args.data_dir, 'rescal-model-final.pkl')
    torch.save(model.state_dict(), model_dump_name)
    pass


def display():

    if not os.path.exists(args.data_dir):
        print("data dir is empty")
        exit()

    dump_path = args.data_dir + "dump_data.pkl"
    corpus = cPickle.load(open(dump_path, "rb"))
    # 给定若干类型，提取实体表示
    sel_types = ['people.person',
                 'film.film',
                 #'book.author',
                 #'film.director',
                 'book.book_subject',
                 'award.award_winner',
                 'organization.organization']

    type_list, type_entities = corpus.get_type_entities(sel_types, 100)
    type_ent_ids = corpus.get_entity_ids(type_entities)

    # model = TransE(corpus.num_ent(), corpus.num_rel(), device,
    #                args.emb_size).cuda()
    # model_dump_name = os.path.join(args.data_dir, 'transe-model-200.pkl')
    # model.load_state_dict(torch.load(model_dump_name))

    # model = RESCAL(corpus.num_ent(), corpus.num_rel(), device,
    #                args.emb_size).cuda()
    model = RESCAL(corpus.num_ent(), corpus.num_rel(), device,
                   args.emb_size)
    model_dump_name = os.path.join(args.data_dir, 'rescal-model-50.pkl')
    model.load_state_dict(torch.load(model_dump_name))

    type_ent_labels, type_ent_resps = list(), list()
    total_colors = ['red', 'green', 'blue', 'purple', 'yellow']
    type_score_means = list()
    model.eval()
    for ind, ent_ids in enumerate(type_ent_ids[:len(total_colors)]):
        # ent_ids_t = torch.tensor(ent_ids).cuda()
        ent_ids_t = torch.tensor(ent_ids)
        ent_resps = model.get_ent_resps(ent_ids_t).cpu().detach().numpy()

        type_ent_resps.extend(ent_resps.tolist())
        type_ent_labels.extend([ind] * len(ent_resps))

        ent_sims = cosine_similarity(ent_resps)
        type_score_means.append(np.mean(ent_sims))

    # 一：计算均值
    for name, score in zip(sel_types, type_score_means):
        print("%s\t%.2f" % (name, score))
    # 二：画图展示
    # labels_colors = [total_colors[x] for x in type_ent_labels]
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    low_dim_embs = tsne.fit_transform(type_ent_resps)
    plot_with_labels(low_dim_embs, type_ent_labels, total_colors)

    pass

def evaluation():
    if not os.path.exists(args.data_dir):
        print("data dir is empty")
        exit()

    dump_path = args.data_dir + "dump_data.pkl"
    corpus = cPickle.load(open(dump_path, "rb"))
    print('ent:{},rel:{}'.format(corpus.num_ent(), corpus.num_rel()))

    train_list, valid_list, test_list = corpus.get_list()

    # model = TransE(corpus.num_ent(), corpus.num_rel(), device,
    #                args.emb_size).cuda()
    model = TransE(corpus.num_ent(), corpus.num_rel(), device,
                   args.emb_size)
    print(model)
    #entity_emb = model.ent_embeddings.weight.data
    #relation_emb = model.rel_embeddings.weight.data
    #print('trans: ent:{}, rel:{}'.format(entity_emb.size(), relation_emb.size()))
    model_dump_name = os.path.join(args.data_dir, 'transe-model-50.pkl')
    model.load_state_dict(torch.load(model_dump_name))

    entity_emb = model.ent_embeddings.weight.data
    relation_emb = model.rel_embeddings.weight.data
    print('trans: ent:{}, rel:{}'.format(entity_emb.size(), relation_emb.size()))
    norm = model.norm

    test_link_prediction(test_list, train_list, entity_emb, relation_emb, norm)

def save_emb():
    #model = TransE(14541, 237, device,
     #              args.emb_size).cuda()
    # model = RESCAL(14541, 237, device,
    #                args.emb_size).cuda()
    model = RESCAL(14541, 237, device,
                   args.emb_size)
    print(model)
    # entity_emb = model.ent_embeddings.weight.data
    # relation_emb = model.rel_embeddings.weight.data
    # print('trans: ent:{}, rel:{}'.format(entity_emb.size(), relation_emb.size()))
    model_dump_name = os.path.join(args.data_dir, 'rescal-model-50.pkl')
    model.load_state_dict(torch.load(model_dump_name))

    entity_emb = model.ent_embeddings.weight.data
    relation_emb = model.rel_embeddings.weight.data
    print('trans: ent:{}, rel:{}'.format(entity_emb.size(), relation_emb.size()))
    torch.save(entity_emb, os.path.join(args.data_dir, 'rescal_ent50.pkl'))
    torch.save(relation_emb, os.path.join(args.data_dir, 'rescal_rel50.pkl'))


if __name__ == '__main__':
    if args.IsTrain:
        if args.model == 'transe':
            train_transe()
        if args.model == 'rescal':
            train_rescal()
    else:
        evaluation()

    if args.display:
        display()

    pass