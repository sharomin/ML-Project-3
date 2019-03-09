# import load_data as input
# import torch
# import torchvision as tv 
# import torchvision.transforms as transforms
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim


# #Data Set Prepration
# transform = transforms.Compose([tv.transforms.ToTensor(), 
#                                tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

# trainset = tv.datasets.CIFAR10(
#     '/Users/saeedshoarayenejati/Downloads/COMP 551/mini project-3/comp-551-w2019-project-3-modified-mnist/data', train=True, download=False, transform=transform)
# dataloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False, num_workers=4)
# # images,_ = iter(dataloader).next()
# # print images.max()

# #Models 
# class FisrtCNN(nn.Module):
#     def __init__(self):
#         super(FisrtCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5 )
#         self.fc1 = nn.Linear(16*5*5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16*5*5)
#         x = F.relu(self.fc1(x))  
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x 
# net = FisrtCNN()


# #Optimization and score function 
# loss_function = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# #Training of the model 
# for epoch in range(2):
#     running_loss = 0.0
#     for i, data in enumerate(dataloader, 0):
#         inputs, labels = data
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = loss_function(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         if i%2000== 1999 :
#             print('[%d, %5d]loss: %.3f'%(epoch+1, i+1, running_loss/2000))
#             running_loss=0.0

# print('Training is finished')
