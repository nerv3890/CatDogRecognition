import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2

BATCH_SIZE = 64
EPOCH = 100

# Load data
print('Loading the data...')

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    #torchvision.transforms.CenterCrop(84),
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])
])

train_data = torchvision.datasets.ImageFolder(root='data/training_set/', transform=data_transform)
test_data = torchvision.datasets.ImageFolder(root='data/test_set/', transform=data_transform)

#print(train_data.__getitem__(0)[0].shape)
#print(test_data)

train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_dataloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
'''
for i in range(100):
    if i == 0:
        test_x = test_data.__getitem__(i)[0]
    else:
        test_x = torch.cat((test_x, test_data.__getitem__(i)[0]), 0)

#print(test_y.size(), type(test_y))
test_x = torch.reshape(test_x, (100, 3, 200, 200))
#test_x = torch.unsqueeze(test_x, dim=1).type(torch.FloatTensor)
test_y = torch.tensor(test_data.targets[:100], dtype=torch.long)

test_x = test_x.cuda()
test_y = test_y.cuda()
#print(test_y, type(test_y))
'''
# Build the model
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1) # output: (64, 224, 224)
        self.maxpool1 = nn.AdaptiveMaxPool2d((112, 112))
        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1) # output: (128, 112, 112)
        self.maxpool2 = nn.AdaptiveMaxPool2d((56, 56))
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1) # output: (256, 56, 56)
        self.maxpool3 = nn.AdaptiveMaxPool2d((28, 28))
        self.conv4 = nn.Conv2d(256, 512, 3, stride=1, padding=1) # output: (512, 28, 28) 
        self.maxpool4 = nn.AdaptiveMaxPool2d((14, 14))
        self.conv5 = nn.Conv2d(512, 512, 3, stride=1, padding=1) # output: (512, 14, 14) 
        self.maxpool5 = nn.AdaptiveMaxPool2d((7, 7))
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 2)
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(F.relu(x))
        x = self.conv2(x)
        x = self.maxpool2(F.relu(x))
        x = self.conv3(x)
        x = self.maxpool3(F.relu(x))
        x = self.conv4(x)
        x = self.maxpool4(F.relu(x))
        x = self.conv5(x)
        x = self.maxpool5(F.relu(x))
        x = self.avgpool1(x)
        #print('avg_pool: ', x.shape) 
        #x = torch.flatten(x))
        x = x.view(x.size(0), -1) # (32, -1), x.view makes (32, 64, 1, 1) to (32, 64)
        #print('avg_pool_flatten: ', x.shape)
        x = self.fc1(x)
        #print('fc1: ', x.shape)

        return x

print('Building the net...')

net = Net()
net = net.cuda()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

f = open('log.txt', 'w')

print('Training...')
#print(train_dataloader)

for epoch in range(EPOCH):
    for data ,label in train_dataloader:
        data = data.cuda()
        label = label.cuda()
        #print(data)
        output = net(data)
        loss = loss_func(output, label)
        optimizer.zero_grad() # clear gradients for this training step
        loss.backward() # backpropagation, compute gradients
        optimizer.step() # apply gradients
    total = 0    
    correct = 0
    for test_d, test_l in test_dataloader:
        #print('test_d: ', test_d)
        #print('test_l:', test_l)
        test_d = test_d.cuda()
        test_l = test_l.cuda()
        output = net(test_d)
        #print('output: ', output)
        predict = torch.max(output, 1)[1].cuda()
        #print(test_l)
        #print('predict: ', predict)
        total += test_l.size(0)
        correct += (predict == test_l).sum().item()
        #print('correct:', correct)
    accuracy = correct / total
    print('epoch:{} | loss:{:.4f} | accuracy:{:.4f}'.format(epoch+1, loss, accuracy))
    f.write('epoch:{} | loss:{:.4f} | accuracy:{:.4f}'.format(epoch+1, loss, accuracy))
    f.write('\n')
        
    '''
    test_x, test_y = test_dataloader
    test_output = net(test_x)
    #print(test_output)
    predict = torch.max(test_output,1)[1].cuda()
    # print(predict, predict.size(), type(predict))
    # print(test_y, test_y.size(), type(test_y))
    accuracy = (predict == test_y).sum().item() / float(test_y.size(0))
    # torch.item() returns the value of this tensor as a standard Python number. This only works for tensors with one element.
    print('epoch:{} | loss:{:.4f} | accuracy:{:.4f}'.format(epoch+1, loss, accuracy))
    '''
torch.save(net.state_dict(), 'net_params.pkl')
f.close()
