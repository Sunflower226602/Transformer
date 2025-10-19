import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 论文中的W^Q, W^K, W^V, W^O矩阵
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        # 计算QK^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用mask（解码器自注意力需要）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        output = torch.matmul(attn_weights, v)
        return output, attn_weights
        
    def forward(self, q, k, v, mask=None):
        batch_size, seq_len = q.size(0), q.size(1)
        
        # 线性变换并分头
        q = self.w_q(q).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 缩放点积注意力
        attn_output, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        # 拼接多头结果
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 最终线性变换
        output = self.w_o(attn_output)
        return output, attn_weights

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # 论文中的两层前馈网络
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # ReLU激活函数
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 计算位置编码（论文中的公式）
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # 添加位置编码到输入嵌入
        return x + self.pe[:x.size(0), :]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 层归一化（论文中的Add & Norm）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 多头自注意力子层
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络子层
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # 掩码自注意力（防止看到未来信息）
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 交叉注意力（查询来自解码器，键值来自编码器）
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_layers=6, d_ff=2048, max_seq_length=5000, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # 构建编码器和解码器堆叠
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        self.final_linear = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def generate_mask(self, src, tgt):
        # 源序列填充掩码
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        # 目标序列掩码（防止看到未来信息）
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_len = tgt.size(1)
        nopeak_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        tgt_mask = tgt_mask & nopeak_mask.unsqueeze(0).unsqueeze(0)
        
        return src_mask, tgt_mask
        
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # 编码器前向传播
        src_embedded = self.dropout(self.pos_encoding(self.src_embed(src) * math.sqrt(self.d_model)))
        enc_output = src_embedded
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        
        # 解码器前向传播
        tgt_embedded = self.dropout(self.pos_encoding(self.tgt_embed(tgt) * math.sqrt(self.d_model)))
        dec_output = tgt_embedded
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        
        # 最终输出层
        output = self.final_linear(dec_output)
        return output
    
def train_transformer():
    # 超参数（遵循论文设置）
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    dropout = 0.1
    
    # 初始化模型
    model = Transformer(src_vocab_size=10000, tgt_vocab_size=10000, 
                       d_model=d_model, num_heads=num_heads, 
                       num_layers=num_layers, d_ff=d_ff, dropout=dropout)
    
    # 优化器（论文使用Adam with β1=0.9, β2=0.98, ε=10^-9）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, 
                                betas=(0.9, 0.98), eps=1e-9)
    
    # 学习率调度（论文中的warmup策略）
    def lr_scheduler(step, d_model, warmup_steps=4000):
        return d_model**-0.5 * min(step**-0.5, step * warmup_steps**-1.5)