import pandas as pd
import re
import torch
import torch.utils.data as data
import nltk
from nltk.corpus import stopwords
import statistics
from pathlib import Path
import csv

PAD_INDEX = 0
UNK_INDEX = 1
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
task_name = 'task_1_'

def convert(source_files_path, output_path):

    for child in Path(source_files_path).iterdir():
        if child.suffix == '.deft':
            write_converted(child, Path.joinpath(output_path, task_name + child.name))
        elif child.is_dir():
            convert(child, output_path)

def write_converted(source_file, output_file):

    sentences = pd.DataFrame(columns=['label', 'sentence'])
    with open(source_file, encoding="utf-8") as source_text:
        has_def = 0
        new_sentence = ''
        all_lines = list(source_text.readlines())
        for index, line in enumerate(all_lines):

            if re.match('^\s+$', line) and len(new_sentence) > 0 and (not re.match(r'^\s*\d+\s*\.$', new_sentence)
                                                                      or all_lines[index-1] == '\n'):
                sentences = sentences.append({'label': has_def, 'sentence': new_sentence}, ignore_index=True)
                new_sentence = ''
                has_def = 0

            if line == '\n':
                continue

            line_parts = line.split('\t')
            new_sentence = new_sentence + ' ' + line_parts[0]
            if line_parts[4][3:] == 'Definition':
                has_def = 1
        if len(new_sentence) > 0:
            sentences = sentences.append({'label': has_def, 'sentence': new_sentence}, ignore_index=True)
    sentences.to_csv(output_file, header=True, index=False, quoting=csv.QUOTE_ALL, sep='\t')

def listToString(s):
    str1 = ""
    for ele in s:
        str1 += ele
        str1 += " "
    return str1

def clean(sent):
    sent = sent.replace(',', ' ').replace('.', ' ').replace('!', ' ').replace('?', ' ').replace('/', ' ').replace(':',' ').replace('&', ' ').split()
    sentence1 = []
    for i in sent:
        tmp = re.sub(r'[^A-Za-z]+', ' ', i.strip()).lower()
        if tmp.strip().replace(' ', '') != "":
            sentence1.append(tmp.strip().replace(' ', ''))
    sentence2 = [word for word in sentence1 if word.lower() not in stop_words]
    return listToString(sentence2)

def cleanToList(sent):
    sent = sent.replace(',',' ').replace('.',' ').replace('!',' ').replace('?',' ').replace('/',' ').replace(':',' ').replace('&',' ').split()
    sentence1 = []
    for i in sent:
        tmp = re.sub(r'[^A-Za-z]+', ' ', i.strip()).lower()
        if tmp.strip().replace(' ','') != "":
            sentence1.append(tmp.strip().replace(' ',''))
    sentence2 = [word for word in sentence1 if word.lower() not in stop_words]
    return listToString(sentence2),sentence2

def average(num):
    sum_num = 0
    for t in num:
        sum_num = sum_num + t

    avg = sum_num / len(num)
    return avg

class Vocab():
    def __init__(self):
        self.word2index = {"PAD": PAD_INDEX, "UNK": UNK_INDEX}
        self.word2count = {}
        self.index2word = {PAD_INDEX: "PAD", UNK_INDEX: "UNK"}
        self.n_words = 2 # Count default tokens
        self.word_num = 0

    def index_words(self, sentence):
        for word in sentence:
            self.word_num += 1
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.word2count[word] = 1
                self.n_words += 1
            else:
                self.word2count[word] += 1

def Lang(vocab, file_name):
    statistic = {"sent_num": 0, "word_num": 0, "vocab_size": 0, "max_len": 0, "avg_len": 0, "len_std": 0, "class_distribution": {}}
    df = pd.read_csv(file_name)
    statistic["sent_num"] = len(df)
    sent_len_list = []
    max_len = 0

    statistic['sent_num'] = len(df)
    for i in df['sentence']:
        sentenceList,sentence = cleanToList(str(i))
        vocab.index_words(sentence)
        sent_len_list.append(len(sentence))
    statistic['word_num'] = sum(sent_len_list)
    statistic['max_len'] = max(sent_len_list)
    statistic['avg_len'] = average(sent_len_list)
    statistic['len_std'] = statistics.stdev(sent_len_list)
    wordDict = vocab.word2index
    statistic['vocab_size'] = wordDict[list(wordDict)[-1]]
    label = df['label'].value_counts()
    statistic['class_distribution'] = {0:int(label.get(0)),1:int(label.get(1))}
    print(statistic)
    return vocab, statistic


class Dataset(data.Dataset):

    def __init__(self, data, vocab):
        self.id, self.X, self.y = data
        self.vocab = vocab
        self.num_total_seqs = len(self.X)
        self.id = torch.LongTensor(self.id)
        if self.y is not None:
            self.y = torch.LongTensor(self.y)

    def __getitem__(self, index):

        ind = self.id[index]
        X = self.tokenize(self.X[index])
        if(self.y is not None):
            y = self.y[index]
            return torch.LongTensor(X), y, ind
        else:
            return torch.LongTensor(X), ind

    def __len__(self):
        return self.num_total_seqs

    def tokenize(self, sentence):
        return [self.vocab.word2index[word] if word in self.vocab.word2index else UNK_INDEX for word in sentence]

def preprocess(filename, max_len=200, test=False):
    df = pd.read_csv(filename)
    id_ = [] # review id
    label = [] # rating
    content = [] # review content

    for i in range(len(df)):
        id_.append(int(df['id'][i]))
        if not test:
            label.append(int(df['label'][i]))
        sentence = clean(str(df['sentence'][i]).strip())
        sentence = sentence.split()
        sent_len = len(sentence)
        if sent_len > max_len:
            content.append(sentence[:max_len])
        else:
            content.append(sentence + ["PAD"] * (max_len - sent_len))

    if test:
        len(id_) == len(content)
        return (id_, content, None)
    else:
        assert len(id_) == len(content) == len(label)
        return (id_, content, label)

def get_dataloaders(batch_size, max_len):
    vocab = Vocab()
    vocab, statistic = Lang(vocab, "train.csv")

    train_data = preprocess("train.csv", max_len)
    dev_data = preprocess("dev.csv", max_len)
    test_data = preprocess("test.csv", max_len, test=True)
    train = Dataset(train_data, vocab)
    dev = Dataset(dev_data, vocab)
    test = Dataset(test_data, vocab)

    data_loader_tr = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    data_loader_dev = torch.utils.data.DataLoader(dataset=dev, batch_size=batch_size, shuffle=False)
    data_loader_test = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)
    return data_loader_tr, data_loader_dev, data_loader_test, statistic["vocab_size"]
