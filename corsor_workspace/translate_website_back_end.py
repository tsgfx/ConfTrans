import os
import torch
from threading import Thread
import tornado.web
import tornado.ioloop
from tornado.web import RequestHandler
import re
from sacremoses import MosesDetokenizer, MosesTokenizer
from pathlib import Path
import subprocess
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

# 从你的transformer模型文件中导入必要的类和配置
from transformer_bpe_50000_large import TransformerModel, TransformerEmbedding, Tokenizer

Tensor = torch.Tensor

# 配置路径
MODEL_PATH = "./checkpoints/model_bpe_50000_large_epoch40_not_share/best.ckpt"
DATASET_PATH = "./train_data_size_1000000"

class Translator:
    def __init__(self, model, en_tokenizer, zh_tokenizer, dataset_path=DATASET_PATH):
        """
        初始化翻译器类，包含 BPE 处理、分词、去标记化和模型推理功能。
        """
        self.dataset_path = Path(dataset_path)

        # 英文和中文的 BPE 规则文件
        self.en_bpe_rules = self.dataset_path / "bpe.en.50000"
        self.zh_bpe_rules = self.dataset_path / "bpe.zh.50000"

        # 英文和中文的词汇表文件
        self.en_vocab = self.dataset_path / "en.vocab"
        self.zh_vocab = self.dataset_path / "zh.vocab"

        # 初始化 Moses 分词器和去标记化器
        self.mose_tokenizer = MosesTokenizer(lang="en")
        self.mose_detokenizer = MosesDetokenizer(lang="zh")

        # 设置模型和评估模式
        self.model = model
        self.model.eval()

        # 设置英文和中文的 tokenizer
        self.en_tokenizer = en_tokenizer
        self.zh_tokenizer = zh_tokenizer

        # 正则模式，用于去除 BPE 标记
        self.pattern = re.compile(r'(@@ )|(@@ ?$)')

    def apply_bpe(self, input_file, output_file, bpe_rules, vocab_file):
        """
        使用 subword-nmt 的 apply-bpe 命令对文件进行 BPE 分词。
        """
        try:
            subprocess.run([
                "subword-nmt", "apply-bpe",
                "-c", str(bpe_rules),
                "--vocabulary", str(vocab_file),
                "--vocabulary-threshold", "50",
                "-i", str(input_file),
                "-o", str(output_file)
            ], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to apply BPE: {e}")

    def __call__(self, sentence_list):
        """
        执行翻译任务。
        """
        # 英文句子预处理
        sentence_list = [" ".join(self.mose_tokenizer.tokenize(s.lower())) for s in sentence_list]

        # 将句子列表写入临时文件
        temp_input_file = self.dataset_path / "temp_input.txt"
        temp_output_file = self.dataset_path / "temp_output.bpe"

        # 写入英文句子
        with open(temp_input_file, "w", encoding="utf8") as f:
            f.write("\n".join(sentence_list))

        # 对英文句子进行 BPE 分词
        self.apply_bpe(temp_input_file, temp_output_file, self.en_bpe_rules, self.en_vocab)

        # 读取 BPE 分词后的结果
        with open(temp_output_file, "r", encoding="utf8") as f:
            en_tokens_list = [line.strip().split() for line in f]

        # 使用英文tokenizer处理输入
        encoder_input, attn_mask = self.en_tokenizer.encode(en_tokens_list, add_bos=True, add_eos=True, return_mask=True)

        encoder_input = torch.Tensor(encoder_input).to(dtype=torch.int64)

        # 执行模型推理
        outputs = self.model.infer(encoder_inputs=encoder_input, encoder_inputs_mask=attn_mask)
        preds = outputs.preds.numpy()

        # 返回翻译结果，去除 BPE 标记
        return [self.mose_detokenizer.tokenize(self.pattern.sub("", s).split()) for s in self.zh_tokenizer.decode(preds)]

# 初始化翻译器
def init_translator():
    # 加载配置
    config = {
        "bos_idx": 1,
        "eos_idx": 3,
        "pad_idx": 0,
        "en_vocab_size": None,  # 需要从词表计算
        "zh_vocab_size": None,  # 需要从词表计算
        "max_length": 128,
        "d_model": 512,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "layer_norm_eps": 1e-6,
        "num_heads": 8,
        "num_decoder_layers": 6,
        "num_encoder_layers": 6,
        "label_smoothing": 0.1,
        "beta1": 0.9,
        "beta2": 0.98,
        "eps": 1e-9,
        "warmup_steps": 4000,
    }

    # 初始化分词器
    en_word2idx = {"[PAD]": 0, "[BOS]": 1, "[UNK]": 2, "[EOS]": 3}
    zh_word2idx = {"[PAD]": 0, "[BOS]": 1, "[UNK]": 2, "[EOS]": 3}
    en_idx2word = {v: k for k, v in en_word2idx.items()}
    zh_idx2word = {v: k for k, v in zh_word2idx.items()}

    # 加载词表
    en_index = len(en_idx2word)
    zh_index = len(zh_idx2word)
    threshold = 1

    with open(f"{DATASET_PATH}/en.vocab", "r", encoding="utf8") as file:
        for line in file:
            token, counts = line.strip().split()
            if int(counts) >= threshold:
                en_word2idx[token] = en_index
                en_idx2word[en_index] = token
                en_index += 1

    with open(f"{DATASET_PATH}/zh.vocab", "r", encoding="utf8") as file:
        for line in file:
            token, counts = line.strip().split()
            if int(counts) >= threshold:
                zh_word2idx[token] = zh_index
                zh_idx2word[zh_index] = token
                zh_index += 1

    # 更新配置中的词表大小
    config["en_vocab_size"] = len(en_word2idx)
    config["zh_vocab_size"] = len(zh_word2idx)

    # 初始化tokenizer
    en_tokenizer = Tokenizer(en_word2idx, en_idx2word)
    zh_tokenizer = Tokenizer(zh_word2idx, zh_idx2word)

    # 加载模型
    model = TransformerModel(config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    # 创建翻译器实例
    return Translator(model, en_tokenizer, zh_tokenizer)

# 初始化翻译器
translator = init_translator()

class BaseHandler(RequestHandler):
    """解决JS跨域请求问题"""
    def set_default_headers(self):
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Methods', 'POST, GET')
        self.set_header('Access-Control-Max-Age', 1000)
        self.set_header('Access-Control-Allow-Headers', '*')

class IndexHandler(BaseHandler):
    def get(self):
        # 获取用户输入的英文文本
        infos = self.get_query_argument("infos")
        print("English:", infos)
        
        try:
            # 调用翻译器进行翻译
            result = translator([infos])[0]
            print("Chinese:", result)
        except Exception as e:
            print(e)
            result = "服务器内部错误"
            
        self.write(result)

if __name__ == '__main__':
    # 创建一个应用对象
    app = tornado.web.Application([(r'/api/chatbot', IndexHandler)])
    # 绑定一个监听端口
    app.listen(6006)
    # 启动web程序，开始监听端口的连接
    print("Server started at port 6006")
    tornado.ioloop.IOLoop.current().start()