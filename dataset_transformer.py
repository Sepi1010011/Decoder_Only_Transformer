START_TOKEN = '<start>'
END_TOKEN = '<end>'
PADDING_TOKEN = '<pad>'
TOTAL_SENTENCES = 650000
max_sequence_length = 2048
english_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',':', '<', '=', '>', '?', '@','[', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l','m', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN]

file_wiki = open("asset/AllCombined.txt",'r', encoding='utf-8')
wiki = file_wiki.readlines()
file_wiki.seek(0)

filtered_text_input = []
filtered_text_target = []
valid_sentence = []

def preprocessing(file):
    list = []
    for text in file:
        if text == "\n" or len(text) <= 35:
            continue
        
        list.append(text)
    return list
 
def processing(pre_filter):
    dataset = pre_filter [:TOTAL_SENTENCES]
    dataset = [sentence.rstrip('\n').lower() for sentence in pre_filter]
    return dataset


def is_valid_token(sentence, vocab):
    for token in list(set(sentence)):
        
        if token not in vocab:
            return False
    return True

def is_valid_length(sentence, max_sequence_length):
    token = 0
    for t in sentence:
        if t != " ":
            token =+ 1
    return token < (max_sequence_length)

def validation(dataset:list):
    valid_list = []
    for index in range(len(dataset)):
        sentence = dataset[index]
        if is_valid_length(sentence, max_sequence_length) and is_valid_token(sentence, english_vocabulary):
            valid_list.append(index)
    return valid_list

def seperate(dataset):
    for text in dataset:

        text = text.split(" ")
        target_text = text[1:]
        for t in text:
            input = " ".join(text)
        for t in target_text:
            target = " ".join(target_text)
            
        filtered_text_input.append(input)
        filtered_text_target.append(target)
        
def save():
    with open("asset/input.txt", "w") as file:
        for text in filtered_text_input:
            file.write(text)
            file.write("\n")
        file.close()

    with open("asset/target.txt", "w") as file:
        for text in filtered_text_target:
            file.write(text)
            file.write("\n")
        file.close()



def get_test(index):
    wiki_input = open("asset/input.txt",'r')
    input = wiki_input.readlines()

    wiki_target = open("asset/target.txt",'r')
    target = wiki_target.readlines()
    print(input[index], target[index], sep="\n")
    

# for creating dataset called input and target run these codes:

# filtered_text = preprocessing(wiki)
# dataset = processing(filtered_text)

# valid_sentence = validation(dataset)
# dataset = [dataset[index] for index in valid_sentence]
# seperate(dataset)
# save()

# for test that each rows of input and target are the same and differences is just a right shift
# get_test(4545)