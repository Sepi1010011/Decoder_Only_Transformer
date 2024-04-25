import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from decoder_only_transformer import (GPTLanguageModel,d_model, ffn_hidden, num_heads, 
                                      drop_prob, num_layers, max_sequence_length, 
                                      en_vocab_size, language_to_index, create_decoder_masks, START_TOKEN,
                                      END_TOKEN, index_to_language, max_sequence_length, 
                                      learning_rate, PADDING_TOKEN)

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


model = GPTLanguageModel(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, en_vocab_size, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
device = get_device()
model = model.to(device)
load_model = torch.load("Decoder_Only_Transformer_v1.pth", map_location=device)
model.load_state_dict(load_model)

model.eval()
def text_generation(sentence, temperature=0.4):
    sentence = (sentence,)
    for word_counter in range(max_sequence_length):
        self_attention_mask = create_decoder_masks(sentence)
        predictions = model(sentence,
                                 self_attention_mask.to(device), 
                                 dec_start_token=True,
                                 dec_end_token=True)
        
        next_token_prob_distribution = predictions[0][word_counter] / temperature
        next_token_prob_distribution = torch.softmax(next_token_prob_distribution, dim=-1)
        next_token_index = torch.multinomial(next_token_prob_distribution, 1).item()
        next_token = index_to_language[next_token_index] # type: ignore
        text_generated = (sentence[0] + next_token,)
        if next_token == END_TOKEN:
            break
    return text_generated[0]



# # <<<<<<<<<<<<<<<<<<<<<<<<<<<For Fine Tuning the model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# # Step 1: Load Pre-trained Model
# # Step 2: Modify Model Architecture (if needed)
# # For example, replace the output layer

# # load_model.fc = nn.Linear(load_model.fc.in_features, num_classes)

# # Step 3: Define New Dataset and DataLoader
# # Define your new dataset and DataLoader

# # Step 4: Define Loss Function and Optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(load_model.parameters(), lr=learning_rate)

# # Step 5: Train the Model
# for epoch in range(num_epochs):
#     for inputs, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = load_model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

# # Step 6: Monitor and Evaluate
# # Evaluate the model, fine-tune hyperparameters, etc.

# # Step 7: Save the Fine-tuned Model
# torch.save(load_model, 'fine_tuned_model.pth')

if __name__ == "__main__":
    print("HELLO THIS IS GPT V1 YOU CAN CHAT.....")
    num_op = int(input("Hello: if you want to chat enter 1 else enter 2 for Fine_Tuning: "))
    while True:
        if num_op == 1:
            chat = input("BOT: ")
            print(text_generation(chat))
        else:
            dataset = input("Enter your dataset")
            # fine_tune(dataset)
        