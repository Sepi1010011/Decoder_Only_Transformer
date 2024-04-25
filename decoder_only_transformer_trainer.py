import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from decoder_only_transformer import create_decoder_masks, GPTLanguageModel, get_device
from transformers import BertTokenizer, BertModel



d_model = 768
num_layers = 4
num_heads = 2
batch_size = 15
learning_rate = 0.00002
block_number = 4
drop_prob = 0.2
NEG_INFTY = -1e9
num_epochs = 10
ffn_hidden = 2048

PRETRAINED_MODEL = "bert-base-uncased"
START_TOKEN = '[START]'
END_TOKEN = '[END]'
PADDING_TOKEN = '[PAD]'
TOTAL_SENTENCES = 200000
max_sequence_length = 512

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)
bert_model = BertModel.from_pretrained(PRETRAINED_MODEL)

english_vocabulary = tokenizer.get_vocab()
token_to_index = {token: index for token, index in english_vocabulary.items()}
index_to_token = {index: token for token, index in english_vocabulary.items()}
en_vocab_size = 30522 

file_input = open("input.txt", 'r')
dataset_input = file_input.readlines()

file_target = open("target.txt", 'r')
dataset_target = file_target.readlines()

dataset_input = [sentence.replace('\n', '') for sentence in dataset_input]
dataset_target = [sentence.replace('\n', '') for sentence in dataset_target]



def choose_random_line(dataset1, dataset2):
    input, target = [], []
    indices = list(range(len(dataset1)))
    random.shuffle(indices)
    selected_inidcies = indices[:TOTAL_SENTENCES]
    
    for index in selected_inidcies:
        input.append(dataset1[index])
        target.append(dataset2[index])
    
    return input, target
    

randomly_input, randomly_target = choose_random_line(dataset_input, dataset_target)


class TextDataset(Dataset):

    def __init__(self, input, target):
        self.input = input
        self.target = target

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx], self.target[idx]


dataset = TextDataset(randomly_input, randomly_target )
train_loader = DataLoader(dataset, batch_size)


transformer = GPTLanguageModel(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, en_vocab_size)

# Define loss function and optimizer
for params in transformer.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)
        
criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=token_to_index[PADDING_TOKEN])
optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate)
device = get_device()
target_linear = nn.Linear(768, 1)

transformer.train()
transformer.to(device)

checkpoint_interval = 1
checkpoint_dir = "Checkpoints"

os.makedirs(checkpoint_dir, exist_ok=True)


# Training loop

for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    iterator = iter(train_loader)
    total_loss = 0
    for batch_num, batch in enumerate(iterator):
        # Forward pass
        transformer.train()
        scaler = GradScaler()
        sentence, target = batch
        
        self_attention_mask = create_decoder_masks(sentence)
        optimizer.zero_grad()
        with autocast():
            prediction = transformer(sentence, self_attention_mask.to(device))
            input = tokenizer(target, padding="max_length", truncation=True, max_length=max_sequence_length, return_tensors="pt",)
            outputs = bert_model(**input) 
            target_embedded = outputs.last_hidden_state
            target_embedded = target_linear(target_embedded).to(device)
            
            loss = criterion((prediction.view(-1, prediction.size(-1))).to(device), (target_embedded.view(-1).long()).to(device)).to(device)
            valid_indicies = torch.where(target_embedded.view(-1) == token_to_index[PADDING_TOKEN], False, True)
            loss = loss.sum() / valid_indicies.sum()
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        if batch_num % 2 == 0:
            print(f"Epoch {epoch} Batch {batch_num} Loss {loss.item():.4f}") 
            print(f"input: {sentence[0]}")
            print(f"target: {target[0]}")
            
            bert_embeddings = (bert_model.get_input_embeddings().weight).to(device)
            decoder_output_flat = prediction.view(-1, prediction.size(-1)).to(device)  

            similarities = torch.matmul(decoder_output_flat, bert_embeddings).to(device)
            similarities = similarities.view(1, max_sequence_length, -1).to(device)

            softmax_scores = F.softmax(similarities, dim=-1).to(device)

            predicted_token_ids = torch.argmax(softmax_scores, dim=-1).to(device)

            predicted_tokens = [tokenizer.convert_ids_to_tokens(ids.tolist()) for ids in predicted_token_ids]

            print(f"Text Generation: {predicted_tokens}")
                                
        # torch.cuda.empty_cache()
        
    if epoch % checkpoint_interval == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss
        }, checkpoint_path)


torch.save(transformer.state_dict(), "Decoder_Only_Transformer_v1.pth")

