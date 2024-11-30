# encoding:utf-8
import os, random
from collections import Counter

class Corpus:
    def __init__(self):
        self.ent2id = dict()
        self.rel2id = dict()
        self.triples = {'train':[], 'valid':[], 'test':[]}
        self.start_batchs = {"train": 0, "valid": 0, "test": 0}

    def get_add_ent_id(self, entity):
        if entity in self.ent2id:
            entity_id = self.ent2id[entity]
        else:
            entity_id = len(self.ent2id)
            self.ent2id[entity] = entity_id
        return entity_id

    def get_add_rel_id(self, relation):
        if relation in self.rel2id:
            relation_id = self.rel2id[relation]
        else:
            relation_id = len(self.rel2id)
            self.rel2id[relation] = relation_id
        return relation_id

    def read_triples(self, directory="data/fb15k-237/"):
        for file in ["train", "valid", "test"]:
            file_path = directory + file + ".txt"
            print("read triples from: " + file_path)
            with open(file_path, "r") as f:
                for line in f.readlines():
                    head, rel, tail = line.strip().split("\t")
                    head_id = self.get_add_ent_id(head)
                    rel_id = self.get_add_rel_id(rel)
                    tail_id = self.get_add_ent_id(tail)
                    self.triples[file].append((head_id, rel_id, tail_id))
            random.shuffle(self.triples[file])

    def get_list(self, directory="data/fb15k-237/"):
        train_list = []
        valid_list = []
        test_list = []
        for file in ["train", "valid", "test"]:
            file_path = directory + file + ".txt"
            with open(file_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    li = line.split()
                    if len(li) == 3:
                        if file == 'train':
                            train_list.append((self.get_add_ent_id(li[0]), self.get_add_rel_id(li[1]), self.get_add_ent_id(li[2])))
                        if file == 'valid':
                            valid_list.append((self.get_add_ent_id(li[0]), self.get_add_rel_id(li[1]), self.get_add_ent_id(li[2])))
                        if file == 'test':
                            test_list.append((self.get_add_ent_id(li[0]), self.get_add_rel_id(li[1]), self.get_add_ent_id(li[2])))

        return train_list, valid_list, test_list


    def read_ent_types(self, directory="data/fb15k-237/"):
        file_path = directory + 'mid2types'
        print("read triples from: " + file_path)
        self.type2entids = dict()
        with open(file_path, "r") as f:
            for line in f.readlines():
                terms = [x.strip() for x in line.split('\t')]
                ent = terms[0].strip()
                for typ in terms[1:]:
                    ents = self.type2entids.get(typ, set())
                    ents.add(ent)
                    self.type2entids[typ] = ents
        print("total type: %d" % len(self.type2entids))

    def num_ent(self):
        return len(self.ent2id)

    def num_rel(self):
        return len(self.rel2id)

    def rand_ent_except(self, except_ent):
        rand_ent = random.randint(0, self.num_ent() - 1)
        while rand_ent == except_ent:
            rand_ent = random.randint(0, self.num_ent() - 1)
        return rand_ent

    def generate_neg_triples(self, batch_pos_triples):
        neg_triples = []
        for head, rel, tail in batch_pos_triples:
            head_or_tail = random.randint(0, 1)
            if head_or_tail == 0:  # head
                new_head = self.rand_ent_except(head)
                neg_triples.append((new_head, rel, tail))
            else:  # tail
                new_tail = self.rand_ent_except(tail)
                neg_triples.append((head, rel, new_tail))
        return neg_triples

    def next_pos_batch(self, dataset='train', batch_size = 128):
        if self.start_batchs[dataset] + batch_size > len(self.triples[dataset]):
            self.start_batchs[dataset] = 0
            random.shuffle(self.triples[dataset])
        sb = self.start_batchs[dataset]
        ret_triples = self.triples[dataset][sb:sb + batch_size]
        self.start_batchs[dataset] += batch_size
        return ret_triples

    def split_triples(self, triples):
        h_idx = [x[0] for x in triples]
        r_idx = [x[1] for x in triples]
        t_idx = [x[2] for x in triples]
        return h_idx, r_idx, t_idx

    def split_triples_labels(self, triple_labels):
        h_idx = [x[0][0] for x in triple_labels]
        r_idx = [x[0][1] for x in triple_labels]
        t_idx = [x[0][2] for x in triple_labels]
        labels = [x[1] for x in triple_labels]
        return h_idx, r_idx, t_idx, labels

    def next_batch(self, format="pos_neg", neg_ratio=1):
        if format == "pos_neg":
            bp_triples = self.next_pos_batch()
            bn_triples = self.generate_neg_triples(bp_triples)
            ph_idx, pr_idx, pt_idx = self.split_triples(bp_triples)
            nh_idx, nr_idx, nt_idx = self.split_triples(bn_triples)
            return ph_idx, pr_idx, pt_idx, nh_idx, nr_idx, nt_idx
        elif format == "triple_label":
            bp_triples = self.next_pos_batch()
            bp_triples_and_labels = [(bp_triples[i], 1.0) for i in range(len(bp_triples))]
            bn_triples_and_labels = []

            for _ in range(neg_ratio):
                bn_triples = self.generate_neg_triples(bp_triples)
                bn_triples_and_labels += [(bn_triples[i], 0.0) for i in range(len(bn_triples))]
            all_triples_and_labels = bp_triples_and_labels + bn_triples_and_labels
            random.shuffle(all_triples_and_labels)
            return self.split_triples_labels(all_triples_and_labels)
        else:
            print("Unrecognizeable format in reader.next_batch")
            exit()

    def get_type_entities(self, types, sample_num=100):
        type_list = list()
        type_entities = list()
        for typ in types:
            if typ not in self.type2entids:
                continue
            type_list.append(typ)
            total_ents = self.type2entids[typ]
            if len(total_ents) <= sample_num:
                type_entities.append(total_ents)
            else:
                type_entities.append(random.sample(total_ents, sample_num))
        return type_list, type_entities

    def get_entity_ids(self, entities):
        ent_ids = [[self.ent2id[x] for x in ents if x in self.ent2id]
                   for ents in entities]
        return ent_ids


