# encoding:utf-8
import os
from collections import Counter

if __name__ == '__main__':

    data_path = 'data/fb15k-237'
    entities = set()
    for name in ['train', 'valid', 'test']:
        lines = open(os.path.join(data_path, '%s.txt' % name), "r").readlines()
        for line in lines:
            terms = line.split('\t')
            sub, obj = terms[0].strip(), terms[2].strip()
            entities.add(sub)
            entities.add(obj)
    print(len(entities))

    data_path = '/data/hesz/freebase/mid2types'
    typecounter = Counter()
    with open(data_path, "r") as rf:
        for line in rf:
            terms = [x.strip() for x in line.split('\t')]
            ent = '/m/%s' % terms[0][2:]
            if ent not in entities:
                continue
            typecounter.update(terms[1:])

    print(len(typecounter))
    types = set()
    for key, frq in typecounter.most_common():
        if key.startswith('base.') or frq < 100:
            continue
        if key in set(['common.topic']):
            continue
        types.add(key)
        print("%s\t%d" % (key, frq))
    print(len(types))

    # # 记录实体-类型关系
    # data_path = '/data/hesz/freebase/mid2types'
    # typecounter = Counter()
    # file_writer = open('data/fb15k-237/mid2types', 'w')
    # with open(data_path, "r") as rf:
    #     for line in rf:
    #         terms = [x.strip() for x in line.split('\t')]
    #         ent = '/m/%s' % terms[0][2:]
    #         if ent not in entities:
    #             continue
    #         ent_types = [x for x in terms[2:] if x in types]
    #         if len(ent_types) == 0:
    #             continue
    #         file_writer.write("%s\t%s\n" % (ent, '\t'.join(ent_types)))
    # file_writer.close()

    pass