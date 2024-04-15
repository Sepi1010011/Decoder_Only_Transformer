import random
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from decoder_only_transformer import create_decoder_masks, GPTLanguageModel, get_device


d_model = 128
num_layers = 4
num_heads = 2
batch_size = 15
learning_rate = 0.0006
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


dataset = TextDataset(dataset_input, dataset_target)
train_loader = DataLoader(dataset, batch_size)


transformer = GPTLanguageModel(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, en_vocab_size, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)

# Define loss function and optimizer
for params in transformer.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)
        
criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate)
device = get_device()

transformer.train()
transformer.to(device)

checkpoint_interval = 2
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
            prediction = transformer(sentence, self_attention_mask.to(device), dec_end_token=True, dec_start_token=False)
            labels = transformer.decoder.embedding.batch_tokenize(target, start_token=False, end_token=True)
            loss = criterion(prediction.view(-1, prediction.size(-1)).to(device), (labels.view(-1)).to(device)).to(device)
            valid_indicies = torch.where(labels.view(-1) == language_to_index[PADDING_TOKEN], False, True)
            loss = loss.sum() / valid_indicies.sum()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        if batch_num % 25 == 0:
            print(f"Epoch {epoch} Batch {batch_num} Loss {loss.item():.4f}") 
            print(f"input: {sentence[0]}")
            print(f"target: {target[0]}")
            text_generated = torch.argmax(prediction[0], axis=1) # type: ignore
            predicted_sentence = ""
            for idx in text_generated:
                if idx == language_to_index[END_TOKEN]:
                    break
                index = idx.item()
                predicted_sentence += index_to_language[index]
            
            print(f"Text Generation: {predicted_sentence}")
            
        del sentence, target, self_attention_mask, prediction, labels, loss
        torch.cuda.empty_cache()
        
    if epoch % checkpoint_interval == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss
        }, checkpoint_path)



torch.save(transformer.state_dict(), "Decoder_Only_Transformer_v1.pth")

