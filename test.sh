python eval/curate_memrx.py --split train --out results/memrx_train.jsonl
# python eval/curate_memrx.py --split val   --out results/memrx_val.jsonl
# python eval/curate_memrx.py --split test --out results/memrx_test.jsonl

# 2. 训练 + val 报表
python eval/train_memrx.py --train results/memrx_train.jsonl \
    --val results/memrx_test.jsonl --tau-sweep 0.0 1.0 \
    --out results/memrx_report.json