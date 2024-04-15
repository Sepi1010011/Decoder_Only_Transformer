import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F 


# INPUT EMBEDDING 
# POSITIONAL EMBEDDING 
# TRANSFORMER BLOCK(LAYER NORM - LINEAR - MULIHEAD ATTENTION MASKED - LINEAR - RESIDUAL CONNECTION - LAYER NORM - FEED FORWARD) 
# LAYER NORM 
# LINEAR SOFTMAX 
# OUTPUT PROBABILITES

d_model = 128
num_layers = 4
num_heads = 2
batch_size = 16
learning_rate = 3e-4
block_number = 4
drop_prob = 0.2
NEG_INFTY = -1e9
num_epochs = 10
ffn_hidden = 2048


START_TOKEN = '<start>'
END_TOKEN = '<end>'
PADDING_TOKEN = '<pad>'
TOTAL_SENTENCES = 100000
max_sequence_length = 2048
english_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',':', '<', '=', '>', '?', '@','[', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l','m', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN]
en_vocab_size = len(english_vocabulary)
language_to_index = {v:k for k,v in enumerate(english_vocabulary)}
index_to_language = {k:v for k,v in enumerate(english_vocabulary)}


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
        self.position_embedding = nn.Embedding(max_sequence_length, d_model)

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
    "For a given sentence, create an embedding"
    def __init__(self, max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
    
    def batch_tokenize(self, batch, start_token, end_token):

        def tokenize(sentence, start_token, end_token):
            sentence_word_indicies = [self.language_to_index[token] for token in list(sentence)]
            if start_token:
                sentence_word_indicies.insert(0, self.language_to_index[self.START_TOKEN])
            if end_token:
                sentence_word_indicies.append(self.language_to_index[self.END_TOKEN])
            for _ in range(len(sentence_word_indicies), self.max_sequence_length):
                sentence_word_indicies.append(self.language_to_index[self.PADDING_TOKEN])
            return torch.tensor(sentence_word_indicies)

        tokenized = []
        for sentence_num in range(len(batch)):
            tokenized.append(tokenize(batch[sentence_num], start_token, end_token) )
        tokenized = torch.stack(tokenized)
        return tokenized.to(get_device())
    
    def forward(self, x, start_token, end_token): 
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)
        pos = self.position_encoder().to(get_device())
        x = self.dropout(x + pos)
        return x


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
        self.block1 = TransformerBlocks(d_model, ffn_hidden, num_heads)
        self.block2 = TransformerBlocks(d_model, ffn_hidden, num_heads)
        self.block3 = TransformerBlocks(d_model, ffn_hidden, num_heads)
        self.block4 = TransformerBlocks(d_model, ffn_hidden, num_heads)
        
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.dropout3 = nn.Dropout(p=drop_prob)
    
    def forward(self, x, self_attention_mask):
        x = self.block1(x, decoder_mask=self_attention_mask)
        x = self.dropout1(x)
        x = self.block2(x, decoder_mask=self_attention_mask)
        x = self.dropout2(x)
        x = self.block3(x, decoder_mask=self_attention_mask)
        x = self.dropout3(x)
        x = self.block4(x, decoder_mask=self_attention_mask)
        return x
        
    
class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attention_mask = inputs
        for module in self._modules.values():
            y = module(x, self_attention_mask)
        return y
        

class Decoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN) -> None:
        super(Decoder, self).__init__() 
        self.embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layer = SequentialDecoder(*[DecoderLayer(d_model, num_heads, drop_prob, ffn_hidden) for _ in range(num_layers)])
    
    def forward(self, x, self_attention_mask, start_token, end_token):
        x = self.embedding(x, start_token, end_token)
        x = self.layer(x, self_attention_mask)
        return x 
     
     
class GPTLanguageModel(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers,max_sequence_length, en_vocab_size, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN) -> None:
        super(GPTLanguageModel, self).__init__()
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.norm = LayerNormalization(parameters_shape=[d_model])
        self.linear = nn.Linear(d_model, en_vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.device = get_device()
    
    def forward(self, x, self_attention_mask, dec_start_token=False, dec_end_token=False):
        x = self.decoder(x, self_attention_mask, start_token=dec_start_token, end_token=dec_end_token)
        out = self.norm(x)
        out = self.linear(out)
        out = self.softmax(out)
        return out
    
