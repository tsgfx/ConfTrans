#!/bin/bash

# 参数：pair_dir 为输入目录，dest_dir 为目标目录
pair_dir=$1
dest_dir=$2

# 创建英文和中文的临时文件
echo "Creating temporary files for English and Chinese data..."
cat "$dest_dir/train_src.cut.txt" > "$pair_dir/train_zh"
cat "$dest_dir/train_trg.cut.txt" > "$pair_dir/train_en"

# 定义BPE规则文件路径
EN_BPE_RULES="$pair_dir/bpe.en.50000"
ZH_BPE_RULES="$pair_dir/bpe.zh.50000"

# 对英文数据训练BPE模型并生成词表
echo "Training BPE model for English data..."
subword-nmt learn-joint-bpe-and-vocab \
    -i $1/train_en \
    -s 50000 \
    -o $EN_BPE_RULES \
    --write-vocabulary $1/en.vocab

# 对中文数据训练BPE模型并生成词表
echo "Training BPE model for Chinese data..."
subword-nmt learn-joint-bpe-and-vocab \
    -i $1/train_zh \
    -s 50000 \
    -o $ZH_BPE_RULES \
    --write-vocabulary $1/zh.vocab


# 对训练集、验证集和测试集分别应用 BPE 分词
for mode in train val test; do
    # 对源语言数据应用 BPE 规则，生成分词后的文件
    subword-nmt apply-bpe -c $EN_BPE_RULES < $2/${mode}_trg.cut.txt > $1/${mode}_trg.bpe

    # 对目标语言数据应用 BPE 规则，生成分词后的文件
    subword-nmt apply-bpe -c $ZH_BPE_RULES < $2/${mode}_src.cut.txt > $1/${mode}_src.bpe

    # 打印日志，提示当前模式（train/val/test）的 BPE 处理已完成
    echo " ${mode} mode BPE processing completed."
done
