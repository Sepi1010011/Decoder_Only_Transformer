# START_TOKEN = ''
# END_TOKEN = ''
# PADDING_TOKEN = ''
# TOTAL_SENTENCES = 700000
# max_sequence_length = 2048
# english_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',':', '<', '=', '>', '?', '@','[', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l','m', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN]
# index_to_english = {k:v for k,v in enumerate(english_vocabulary)}
# english_to_index = {v:k for k,v in enumerate(english_vocabulary)}


file_wiki = open("AllCombined.txt",'r', encoding='utf-8')
wiki = file_wiki.readlines()
file_wiki.seek(0)

filtered_text = []
filtered_text_input = []
filtered_text_target = []

# for text in wiki:
#     if text == "\n" or len(text) <= 35:
#         continue
    
#     filtered_text.append(text)
 

# dataset = filtered_text [:TOTAL_SENTENCES]


# dataset = [sentence.rstrip('\n').lower() for sentence in filtered_text]


# def is_valid_token(sentence, vocab):
#     for token in list(set(sentence)):
        
#         if token not in vocab:
#             return False
#     return True

# def is_valid_length(sentence, max_sequence_length):
#     token = 0
#     for t in sentence:
#         if t != " ":
#             token =+ 1
#     return token < (max_sequence_length)

# valid_sentence = []

# def validation(dataset:list):
#     valid_list = []
#     for index in range(len(dataset)):
#         sentence = dataset[index]
#         if is_valid_length(sentence, max_sequence_length) and is_valid_token(sentence, english_vocabulary):
#             valid_list.append(index)
#     return valid_list

# valid_sentence = validation(dataset)
# dataset = [dataset[index] for index in valid_sentence]


# for text in dataset:
    
#     text = text.split(" ")
#     target_text = text[1:]
#     for t in text:
#         input = " ".join(text)
#     for t in target_text:
#         target = " ".join(target_text)
        
#     filtered_text_input.append(input)
#     filtered_text_target.append(target)
    
    
# print(filtered_text_input[331900], filtered_text_target[331900], sep="\n")

# with open("input.txt", "w") as file:
#     for text in filtered_text_input:
#         file.write(text)
#         file.write("\n")
#     file.close()

# with open("target.txt", "w") as file:
#     for text in filtered_text_target:
#         file.write(text)
#         file.write("\n")
#     file.close()
    

wiki_input = open("input.txt",'r')
input = wiki_input.readlines()

wiki_target = open("target.txt",'r')
target = wiki_target.readlines()

print(input[230000], target[230000], sep="\n")
