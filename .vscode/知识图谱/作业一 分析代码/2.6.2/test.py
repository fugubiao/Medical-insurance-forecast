import torch
import os
import numpy as np


def get_head_batch(golden_triple, entity_total):
    head_batch = np.zeros((entity_total, 3), dtype=np.int32)
    head_batch[:, 0] = np.array(list(range(entity_total)))
    head_batch[:, 1] = np.array([golden_triple[1]] * entity_total)
    head_batch[:, 2] = np.array([golden_triple[2]] * entity_total)
    return head_batch


def get_tail_batch(golden_triple, entity_total):
    tail_batch = np.zeros((entity_total, 3), dtype=np.int32)
    tail_batch[:, 0] = np.array([golden_triple[0]] * entity_total)
    tail_batch[:, 1] = np.array([golden_triple[1]] * entity_total)
    tail_batch[:, 2] = np.array(list(range(entity_total)))
    return tail_batch


def _calc(h, t, r, norm):
    return torch.norm(h + r - t, p=norm, dim=1).cpu().numpy().tolist()


def predict(batch, entity_emb, relation_emb, norm):
    pos_hs = batch[:, 0]
    pos_rs = batch[:, 1]
    pos_ts = batch[:, 2]

    pos_hs = torch.IntTensor(pos_hs).cuda()
    pos_rs = torch.IntTensor(pos_rs).cuda()
    pos_ts = torch.IntTensor(pos_ts).cuda()

    #print('pos: h:{}, r:{}, t:{}'.format(pos_hs, pos_rs, pos_ts))

    p_score = _calc(entity_emb[pos_hs.type(torch.long)],
                    entity_emb[pos_ts.type(torch.long)],
                    relation_emb[pos_rs.type(torch.long)],
                    norm)

    return p_score


def test_head(golden_triple, train_set, entity_emb, relation_emb, norm):
    head_batch = get_head_batch(golden_triple, len(entity_emb))
    value = predict(head_batch, entity_emb, relation_emb, norm)
    golden_value = value[golden_triple[0]]
    # li = np.argsort(value)
    res = 1
    sub = 0
    for pos, val in enumerate(value):
        if val < golden_value:
            res += 1
            if (pos, golden_triple[1], golden_triple[2]) in train_set:
                sub += 1

    return res, res - sub


def test_tail(golden_triple, train_set, entity_emb, relation_emb, norm):
    tail_batch = get_tail_batch(golden_triple, len(entity_emb))
    value = predict(tail_batch, entity_emb, relation_emb, norm)
    golden_value = value[golden_triple[2]]
    # li = np.argsort(value)
    res = 1
    sub = 0
    for pos, val in enumerate(value):
        if val < golden_value:
            res += 1
            if (golden_triple[0], golden_triple[1], pos) in train_set:
                sub += 1

    return res, res - sub


def test_link_prediction(test_list, train_set, entity_emb, relation_emb, norm):
    test_total = len(test_list)

    l_mr = 0
    r_mr = 0
    l_hit10 = 0
    l_hit3 = 0
    l_hit1 = 0
    l_mrr = 0
    r_hit10 = 0
    r_hit3 = 0
    r_hit1 = 0
    r_mrr = 0

    l_mr_filter = 0
    r_mr_filter = 0
    l_hit10_filter = 0
    l_hit3_filter = 0
    l_hit1_filter = 0
    l_mrr_filter = 0
    r_hit10_filter = 0
    r_hit3_filter = 0
    r_hit1_filter = 0
    r_mrr_filter = 0

    for i, golden_triple in enumerate(test_list):
        #print('test ---' + str(i) + '--- triple')
        #print(i, end="\r")
        print(i)
        l_pos, l_filter_pos = test_head(golden_triple, train_set, entity_emb, relation_emb, norm)
        r_pos, r_filter_pos = test_tail(golden_triple, train_set, entity_emb, relation_emb, norm)

        '''
        print(golden_triple, end=': ')
        print('l_pos=' + str(l_pos), end=', ')
        print('l_filter_pos=' + str(l_filter_pos), end=', ')
        print('r_pos=' + str(r_pos), end=', ')
        print('r_filter_pos=' + str(r_filter_pos), end='\n')
        '''

        if l_pos <= 10:
            l_hit10 = l_hit10 + 1
        if l_pos <= 3:
            l_hit3 = l_hit3 + 1
        if l_pos == 1:
            l_hit1 = l_hit1 + 1
        if r_pos <= 10:
            r_hit10 = r_hit10 + 1
        if r_pos <= 3:
            r_hit3 = r_hit3 + 1
        if r_pos == 1:
            r_hit1 = r_hit1 + 1

        if l_filter_pos <= 10:
            l_hit10_filter = l_hit10_filter + 1
        if l_filter_pos <= 3:
            l_hit3_filter = l_hit3_filter + 1
        if l_filter_pos == 1:
            l_hit1_filter = l_hit1_filter + 1
        if r_filter_pos <= 10:
            r_hit10_filter = r_hit10_filter + 1
        if r_filter_pos <= 3:
            r_hit3_filter = r_hit3_filter + 1
        if r_filter_pos == 1:
            r_hit1_filter = r_hit1_filter + 1

        l_mr += l_pos
        r_mr += r_pos

        l_mr_filter += l_filter_pos
        r_mr_filter += r_filter_pos

        l_mrr += 1.0 / (float(l_pos))
        r_mrr += 1.0 / (float(r_pos))

        l_mrr_filter += 1.0 / (float(l_filter_pos))
        r_mrr_filter += 1.0 / (float(r_filter_pos))

    l_mr /= test_total
    r_mr /= test_total

    l_mr_filter /= test_total
    r_mr_filter /= test_total

    l_mrr /= test_total
    r_mrr /= test_total

    l_mrr_filter /= test_total
    r_mrr_filter /= test_total

    l_hit10 /= test_total
    r_hit10 /= test_total
    l_hit3 /= test_total
    r_hit3 /= test_total
    l_hit1 /= test_total
    r_hit1 /= test_total

    l_hit10_filter /= test_total
    r_hit10_filter /= test_total
    l_hit3_filter /= test_total
    r_hit3_filter /= test_total
    l_hit1_filter /= test_total
    r_hit1_filter /= test_total

    print('\t\t\t\tmean_rank\t\tmrr\t\t\thit@10\t\t\thit@3\t\t\thit@1\t\t\t')
    print('head(raw)\t\t\t%.3f\t\t\t%.4f\t\t\t%.4f\t\t\t%.4f\t\t\t%.4f\t\t\t' % (l_mr, l_mrr, l_hit10, l_hit3, l_hit1))
    print('tail(raw)\t\t\t%.3f\t\t\t%.4f\t\t\t%.4f\t\t\t%.4f\t\t\t%.4f\t\t\t' % (r_mr, r_mrr, r_hit10, r_hit3, r_hit1))
    print('average(raw)\t\t\t%.3f\t\t\t%.4f\t\t\t%.4f\t\t\t%.4f\t\t\t%.4f\t\t\t' % ((l_mr + r_mr) / 2,
                                                                                    (l_mrr + r_mrr) / 2,
                                                                                    (l_hit10 + r_hit10) / 2,
                                                                                    (l_hit3 + r_hit3) / 2,
                                                                                    (l_hit1 + r_hit1) / 2,))
    print('head(filter)\t\t\t%.3f\t\t\t%.4f\t\t\t%.4f\t\t\t%.4f\t\t\t%.4f\t\t\t' % (l_mr_filter, l_mrr_filter,
                                                                                    l_hit10_filter, l_hit3_filter,
                                                                                    l_hit1_filter))
    print('tail(filter)\t\t\t%.3f\t\t\t%.4f\t\t\t%.4f\t\t\t%.4f\t\t\t%.4f\t\t\t' % (r_mr_filter, r_mrr_filter,
                                                                                    r_hit10_filter, r_hit3_filter,
                                                                                    r_hit1_filter))
    print('average(filter)\t\t\t%.3f\t\t\t%.4f\t\t\t%.4f\t\t\t%.4f\t\t\t%.4f\t\t\t' % ((l_mr_filter + r_mr_filter) / 2,
                                                                                    (l_mrr_filter + r_mrr_filter) / 2,
                                                                                    (l_hit10_filter + r_hit10_filter) / 2,
                                                                                    (l_hit3_filter + r_hit3_filter) / 2,
                                                                                    (l_hit1_filter + r_hit1_filter) / 2,))


