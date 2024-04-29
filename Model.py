# Author: Robert Guthrie
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import csv
from tqdm import tqdm, trange
import  Net
from args import args
from utils import argmax, prepare_sequence,log_sum_exp
import os
import numpy as np

torch.manual_seed(1)

####init__params
START_TAG = args.START_TAG
STOP_TAG = args.STOP_TAG


###load_data
filename = 'train.csv'
training_data = []
with open(filename) as csvfile:
    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    list1 = []
    list2 = []
    for row in csv_reader:  # 将csv 文件中的数据保存到data中
        if (row[0],row[1]) == ('word', 'expected'):
            continue
        if row[0] == '。':   
            list1.append(row[0])
            list2.append(row[1])
            training_data.append((list1, list2))
            list1, list2 = [], []
        else:
            list1.append(row[0])
            list2.append(row[1])


word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
tag_to_ix = {
        'O': 0,
        'S-GPE': 1,
        'S-PER': 2,
        'B-ORG': 3,
        'E-ORG': 4,
        'S-ORG': 5,
        'M-ORG': 6,
        'S-LOC': 7,
        'E-GPE': 8,
        'B-GPE': 9,
        'B-LOC': 10,
        'E-LOC': 11,
        'M-LOC': 12,
        'M-GPE': 13,
        'B-PER': 14,
        'E-PER': 15,
        'M-PER': 16,
         START_TAG: 17,
         STOP_TAG: 18
    }

##model build or load

model = Net.BiLSTM_CRF(len(word_to_ix), tag_to_ix)
# model.to(device)

# 加载模型参数
epoch_init_ = 0
losses = []  # 用于记录损失值
if os.path.exists(args.model_path):
    
    try:
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint['model'])
            optimizer = optim.Adam(model.parameters(), args.lr)
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch_init_ = checkpoint['epoch']
            losses = np.load(args.loss_path)
            print("模型成功加载")
    except FileNotFoundError:
            print(f"找不到模型文件 {args.model_path}，跳过加载模型。")
             # 定义优化器，
            optimizer = optim.Adam(model.parameters(), args.lr)
            # 如果文件不存在，可以选择使用默认初始化模型或其他备用方法
            # model = YourModelClass()
    except Exception as e:
             # 定义优化器
            optimizer = optim.Adam(model.parameters(), args.lr)
            print(f"加载模型时发生错误：{e}")
            # 处理其他可能的异常情况


else:  optimizer = optim.Adam(model.parameters(), args.lr)


# Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    print(model(precheck_sent))

# Make sure prepare_sequence from earlier in the LSTM section is loaded

with tqdm(total=args.epochs - epoch_init_, desc="Training", unit="step") as pbar:
    for epoch in range(epoch_init_, args.epochs):

        # start = epoch % len(training_data) * args.batch_size
        # train_data_index = range(start,start + args.batch_size)
        # sentence, tags = [training_data[x] for x in range(train_data_index)]
        start = epoch % len(training_data)
        sentence, tags = training_data[start]
        
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)
        # losses.append(loss.item())
        losses = np.concatenate([losses, [loss.item()]])

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

        epoch += 1
        pbar.set_postfix(loss=loss, epoches=epoch)
        pbar.update(1)

        if epoch % 10 == 0 or epoch == args.epochs-1:
              #memorize information 
            checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), \
          'epoch': epoch}
            torch.save(checkpoint , args.model_path)
            np.save(args.loss_path, losses)

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))
# We got it!


