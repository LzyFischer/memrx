import json, collections
cv, la = collections.Counter(), []
for line in open('results/memrx_train.jsonl'):
    r = json.loads(line)
    for v in r['views']:
        if r['ll'][v] is None:
            cv[v] += 1; la.append(len(r['gold']))
print(cv)
print('缺失时 gold 长度中位数', sorted(la)[len(la)//2] if la else None)