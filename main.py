import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
import sys
import torch
import torch.nn as nn

from sklearn.metrics import classification_report, accuracy_score
import csv
from preprocess import convert
import os
import glob
import pandas as pd
from csv import reader
from model import WordCNN
from preprocess import get_dataloaders

def trainer(train_loader, dev_loader, model, optimizer, criterion, epoch=1000, early_stop=3, scheduler=None):
    early_stop_counter = early_stop
    best_acc = 0
    best_model = deepcopy(model)

    for e in range(epoch):
        loss_log = []
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (X, y, ind) in pbar:
            outputs = model(X)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_log.append(loss.item())
            pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f}".format((e + 1), np.mean(loss_log)))

        model.eval()
        logits = []
        ys = []
        for X, y, ind in dev_loader:
            logit = model(X)
            logits.append(logit.data.cpu().numpy())
            ys.append(y.data.cpu().numpy())

        logits = np.concatenate(logits, axis=0)
        preds = np.argmax(logits, axis=1)
        ys = np.concatenate(ys, axis=0)
        acc = accuracy_score(y_true=ys, y_pred=preds)
        label_names = ['label 0', 'label 1']
        report = classification_report(ys, preds, digits=3, target_names=label_names)

        if acc > best_acc:
            best_acc = acc
            best_model = deepcopy(model)
            early_stop_counter = early_stop
        else:
            early_stop_counter -= 1

        print("current validation report")
        print("\n{}\n".format(report))
        print()
        print("epcoh: {}, current accuracy:{}, best accuracy:{}".format(e + 1, acc, best_acc))

        if early_stop_counter == 0:
            break

        if scheduler is not None:
            scheduler.step()

    return best_model, best_acc

def predict(model, test_loader, save_file="definition.csv"):
    logits = []
    inds = []
    model.eval()
    for X, ind in test_loader:
        logit = model(X)
        logits.append(logit.data.cpu().numpy())
        inds.append(ind.data.cpu().numpy())
    logits = np.concatenate(logits, axis=0)
    inds = np.concatenate(inds, axis=0)
    preds = np.argmax(logits, axis=1)
    result = {'id': list(inds), "label": preds}
    df = pd.DataFrame(result, index=result['id'])
    df.to_csv(save_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.46)
    parser.add_argument("--kernel_num", type=int, default=100)
    parser.add_argument("--kernel_sizes", type=str, default='3,4,5')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--early_stop", type=int, default=3)
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--class_num", type=int, default=3)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    args = parser.parse_args()

    #convert(Path("data/test_files/unlabeled"), Path("test_draft"))
    # #convert(Path("data/deft_files/train"), Path("train_deft"))
    # index = 0
    # for csvFile in Path("test_draft").iterdir():
    #     sentence = pd.read_csv(csvFile, sep="\t")
    #     fileName = "test" + str(index) + ".csv"
    #     sentence.to_csv(Path(fileName))
    #     index = index + 1

    # lines = list()
    # counter = 0
    # with open(Path("train_2.csv"), encoding="utf-8") as csv:
    #     for sentence in csv:
    #         lines.append(sentence)
    #         new_line = sentence.split(',')
    #         if new_line[1] == "0" and counter < 4000:
    #             lines.remove(sentence)
    #             counter = counter + 1
    #
    # with open(Path("train_3.csv"), 'w', encoding="utf-8") as new_csv:
    #     for new in lines:
    #         new_csv.write(new)


    # os.chdir(Path("test_csv"))
    # extension = 'csv'
    # all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    # combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    # combined_csv.to_csv("combined_csv.csv", index=False, encoding='utf-8-sig')

    train_loader, dev_loader, test_loader, vocab_size = get_dataloaders(args.batch_size, args.max_len)
    model = WordCNN(args, vocab_size, embedding_matrix=None)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    model, best_acc = trainer(train_loader, dev_loader, model, optimizer, criterion, early_stop=args.early_stop)

    print('best_dev_acc:{}'.format(best_acc))
    predict(model, test_loader)


if __name__ == "__main__":
    main()
