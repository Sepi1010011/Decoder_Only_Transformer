import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F 
from transformers import BertTokenizer, BertModel


# INPUT EMBEDDING 
# POSITIONAL EMBEDDING 
# TRANSFORMER BLOCK(LAYER NORM - LINEAR - MULIHEAD ATTENTION MASKED - LINEAR - RESIDUAL CONNECTION - LAYER NORM - FEED FORWARD) 
# LAYER NORM 
# LINEAR SOFTMAX 
# OUTPUT PROBABILITES

d_model = 768
num_layers = 4
num_heads = 2
block_number = 4
drop_prob = 0.2
NEG_INFTY = -1e9
num_epochs = 10
ffn_hidden = 2048


PRETRAINED_MODEL = "bert-base-uncased"
TOTAL_SENTENCES = 200000
max_sequence_length = 512
en_vocab_size = 30522



def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

def create_decoder_masks(sentence_batch):
    num_sentences = len(sentence_batch)
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length] , True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    decoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)

    for idx in range(num_sentences):
      sentence_length = len(sentence_batch[idx])
      chars_to_padding_mask = np.arange(sentence_length + 1, max_sequence_length)
      decoder_padding_mask[idx, :, chars_to_padding_mask] = True
      decoder_padding_mask[idx, chars_to_padding_mask, :] = True

    decoder_self_attention_mask =  torch.where(look_ahead_mask + decoder_padding_mask, NEG_INFTY, 0)
    return decoder_self_attention_mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = (torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1))
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE
    

class SentenceEmbedding(nn.Module):
    
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.bert_model = BertModel.from_pretrained(PRETRAINED_MODEL)
        self.tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)
        self.linear = nn.Linear(self.bert_model.config.hidden_size, d_model).to("cuda") # type: ignore
        self.dropout = nn.Dropout(p=0.2).to("cuda")

    def forward(self, x):
        encoder = self.tokenizer(x, padding="max_length", truncation=True, max_length = self.max_sequence_length, return_tensors="pt")
        input_ids = encoder["input_ids"]
        attention_mask = encoder["attention_mask"]
        with torch.no_grad():
            outputs = self.bert_model(input_ids.to("cuda"), attention_mask=attention_mask.to("cuda")) # type:ignore
            word_embedded = outputs.last_hidden_state
        
        pos = self.position_encoder().to("cuda")
        word_embedded = self.linear(word_embedded)
        word_embedded = self.dropout(word_embedded + pos)
        return word_embedded
            

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model , 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask):
        batch_size, sequence_length, d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        return out


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super(LayerNormalization, self).__init__()
        self.parameters_shape=parameters_shape
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta =  nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out

  
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlocks(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads) -> None:
        super(TransformerBlocks, self).__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.multihead = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm1 = LayerNormalization([d_model])
        self.norm2 = LayerNormalization([d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.dropout3 = nn.Dropout(p=drop_prob)
        self.dropout4 = nn.Dropout(p=drop_prob)
        
    def forward(self, x, decoder_mask):
        residual_x = x
        x = self.norm1(x)
        
        x = self.linear1(x)
        x = self.dropout1(x)
        
        x = self.multihead(x, mask=decoder_mask)
        x = self.dropout2(x)
        
        x = self.linear2(x)
        x = self.dropout3(x)
        
        x = self.norm2(x + residual_x)
        residual_x = x
        
        x = self.ffn(x)
        x = self.dropout4(x)
        
        x = x + residual_x
        return x
        

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, drop_prob, ffn_hidden):
        super(DecoderLayer, self).__init__()
        self.block = SequentialDecoder(*[TransformerBlocks(d_model, ffn_hidden, num_heads) for _ in range(block_number)])
        self.dropout1 = nn.Dropout(p=drop_prob)
        
    
    def forward(self, x, self_attention_mask):
        x = self.block(x, self_attention_mask)
        x = self.dropout1(x)
        return x
        
    
class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attention_mask = inputs
        for module in self._modules.values():
            y = module(x, self_attention_mask)
        return y
        

class Decoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length) -> None:
        super(Decoder, self).__init__() 
        self.embedding = SentenceEmbedding(d_model, max_sequence_length)
        self.layer = SequentialDecoder(*[DecoderLayer(d_model, num_heads, drop_prob, ffn_hidden) for _ in range(num_layers)])
    
    def forward(self, x, self_attention_mask):
        x = self.embedding(x)
        x = self.layer(x, self_attention_mask)
        return x 
     
     
class GPTLanguageModel(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers,max_sequence_length, en_vocab_size) -> None:
        super(GPTLanguageModel, self).__init__()
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length)
        self.norm = LayerNormalization(parameters_shape=[d_model])
        self.linear = nn.Linear(d_model, en_vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.device = get_device()
            
    def forward(self, x, self_attention_mask):
        x = self.decoder(x, self_attention_mask)
        out = self.norm(x)
        out = self.linear(out)
        out = self.softmax(out)
        return out
    
