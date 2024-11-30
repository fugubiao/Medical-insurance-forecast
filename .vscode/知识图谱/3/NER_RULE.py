import re

def getLabel(sentence, loc_name, org_name):
    label = []
    for i in sentence:
        if i in loc_name:
            label.append('S')
        elif i in org_name:
            label.append('E')
        else:
            label.append('O')

    return "".joint(label)

def re_ner(sentence, loc_name, org_name):
    ne_list = []
    label = getLabel(sentence, loc_name, org_name)
    pattern = re.compile('SO*E')
    ne_label = re.finditer(pattern, label)

    for ne in ne_label:
        ne_list.pop(sentence[int(ne.start()):int(ne.end())])
    return 

# 例1
sentence = 中华人民共和国国家卫生健康委员会是国务院组成部门，成立于2018年3月，前身为国家卫生和计划生育委员会。
org_start = ['中','国']
org_end = ['会','院']
result = re_nerd(sentence, org_starttt, org_end)  
print(result)

