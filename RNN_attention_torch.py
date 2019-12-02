#encoding = utf-8
'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
'''
import numpy as np
import math
import operator
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.autograd import Variable

# torch.cuda.set_device(1)
#parameter
n_hidden = 5  #hidden size
train_batch_size = 512
valid_batch_size = 512
num_layers = 1  #hidden layers number
size_window = 4   #the size of window to word
learn_rate = 0.001  #learn rate

dtype = torch.FloatTensor    #number type
train_path = './penn/train.txt'   #the path of dataset to train
valid_path = './penn/valid.txt'   #the path of dataset to valid
test_path = './penn/test.txt'     #the path of dataset to test

#get word_dict
def getDict(path):
    dict = {}
    text = open(path, encoding='utf-8')
    for line in text:
        line = line.split(' ')
        line.pop(0)
        line.pop(-1)
        for it in line:
            if it == '<unk>':
                continue
            elif it not in dict.keys():
                dict[it] = 1
            else:
                dict[it] += 1
    dict_sort = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)
    dict_ret = {}
    for it in dict_sort:
        dict_ret[it[0]] = it[1]
    return dict_ret

word_list = getDict(train_path).keys()    #vocabulary_dict
word_dict = {w:i for i,w in enumerate(word_list)}
word_dict['<unk>'] = len(word_list)
word_dict['eos'] = len(word_list)+1
n_class = len(word_dict)
onehot = np.eye(n_class)

#get input_batch and target_batch of train
def get_train_batch(path):
    input_batch = []
    target_batch = []
    all_word_list = []
    text = open(path, encoding='utf-8')
    for line in text:
        line = line.split(' ')
        line.pop(0)
        line.pop(-1)
        all_word_list.extend(line)
        all_word_list.append('eos')
    for i in range(0, len(all_word_list) - size_window):
        temp = []
        for j in range(0, size_window):
            try:
                temp.append(word_dict[all_word_list[i + j]])
            except:
                temp.append(word_dict['<unk>'])
        input_batch.append(temp)
        try:
            target_batch.append(word_dict[all_word_list[i + size_window]])
        except:
            target_batch.append(word_dict['<unk>'])
    return input_batch, target_batch
input_batch_original, target_batch_original = get_train_batch(train_path)
valid_input_oringinal, valid_target_original = get_train_batch(valid_path)
print('get data!')

#get onehot input batch
def onehotbatch(input_batch):
    for index in range(0,len(input_batch)):
        input_batch[index] = onehot[input_batch[index]]
    return input_batch

#valid
def valid():
    global  time_start
    record = 0
    with torch.no_grad():  #不求梯度
        for i in range(Int_sentence_number):
            valid_input_batch = torch.FloatTensor(onehotbatch(valid_input_oringinal[i * valid_batch_size:(i + 1) * valid_batch_size]))
            valid_target_batch = torch.LongTensor(valid_target_original[i * valid_batch_size:(i + 1) * valid_batch_size])

            hidden = torch.zeros(num_layers, valid_batch_size, n_hidden)
            valid_output = model(hidden, valid_input_batch)
            record += math.pow(math.e,float(criterion(valid_output, valid_target_batch)))*(valid_batch_size/sentence_number)

        valid_input_batch = torch.FloatTensor(onehotbatch(valid_input_oringinal[Int_sentence_number * valid_batch_size:]))
        valid_target_batch = torch.LongTensor(valid_target_original[Int_sentence_number * valid_batch_size:])

        hidden = torch.zeros(num_layers, len(valid_target_batch), n_hidden)
        valid_output = model(hidden, valid_input_batch)
        record += math.pow(math.e ,float(criterion(valid_output, valid_target_batch)))*(len(valid_target_original[Int_sentence_number*valid_batch_size:])/sentence_number)
        ppl_num.append(record)
        print('time loss:',time.time()-time_start, 'ppl:',record)
        time_start = time.time()
        temp = np.array(ppl_num)
        np.save('ppl_dictall.npy', temp)

class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()
        self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden)
        self.W = nn.Parameter(torch.randn([2*n_hidden, n_class]).type(dtype))
        self.b = nn.Parameter(torch.randn([n_class]).type(dtype))

    def forward(self, hidden, X):
        X = X.transpose(0, 1) # X : [n_step, batch_size, n_class]
        outputs, hidden = self.rnn(X, hidden)
        # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        output = outputs[-1]
        attention = []
        for it in outputs[:-1]:
            attention.append(torch.mul(it, output).sum(dim=1).tolist()) #get attention score
        attention = torch.tensor(attention)
        attention = attention.transpose(0, 1)
        attention = nn.functional.softmax(attention, dim=1).transpose(0, 1)
        #get soft attention
        attention_output = torch.zeros(outputs.size()[1], n_hidden)
        for i in range(outputs.size()[0]-1):
            attention_output += torch.mul(attention[i],outputs[i].transpose(0,1)).transpose(0,1)
        output = torch.cat((attention_output, output), 1)
        #joint ouput output:[batch_size, 2*n_hidden]
        model = torch.mm(output, self.W) + self.b # model : [batch_size, n_class]
        return model

model = TextRNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learn_rate)

ppl_num = []
loss_value = []

#get valib batch info
sentence_number = len(valid_target_original)
Int_sentence_number = int(sentence_number / valid_batch_size)

#get train batch info
sentence_number_train = len(target_batch_original)
Int_sentence_number_tarin = int(sentence_number_train / train_batch_size)

# Training
print('training')
flag = True
step = 0
time_start = time.time()
for epoch in range(50):
    seed = epoch
    np.random.seed(seed)
    np.random.shuffle(input_batch_original)
    np.random.seed(seed)
    np.random.shuffle(target_batch_original)
    print('epoch:', epoch)
    for train_num in range(int(len(input_batch_original) / train_batch_size)):
        input_batch = Variable(torch.FloatTensor(onehotbatch(input_batch_original[train_num * train_batch_size:
                                                          (train_num + 1) * train_batch_size])))
        target_batch = torch.LongTensor(target_batch_original[train_num * train_batch_size:
                                                                       (train_num + 1) * train_batch_size])
        optimizer.zero_grad()
        # hidden : [num_layers * num_directions, batch, hidden_size]
        hidden = Variable(torch.zeros(num_layers, train_batch_size, n_hidden))
        # input_batch : [batch_size, n_step, n_class]
        output = model(hidden, input_batch)
        # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, target_batch)
        loss.backward()
        # torch.nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, model.parameters()), max_norm = 35, norm_type = 2)
        optimizer.step()
        step += 1
        if step % 500 == 0:
            valid()

    input_batch = Variable(torch.FloatTensor(onehotbatch(input_batch_original[Int_sentence_number_tarin * train_batch_size:])))
    target_batch = torch.LongTensor(target_batch_original[Int_sentence_number_tarin * train_batch_size:])
    optimizer.zero_grad()
    # hidden : [num_layers * num_directions, batch, hidden_size]
    hidden = Variable(torch.zeros(num_layers, len(target_batch), n_hidden))
    # input_batch : [batch_size, n_step, n_class]
    output = model(hidden, input_batch)
    # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
    loss = criterion(output, target_batch)
    loss.backward()

    # torch.nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, model.parameters()), max_norm = 35, norm_type = 2)
    optimizer.step()
    step += 1
    if step%500 == 0:
        valid()

#save some data
torch.save(model,'model.ckpt')