import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torchvision
import load_data as input
import numpy as np

# Hyperprameters
n_epochs = 3
batch_size_train = 64
batch_size_test = 10000
learning_rate = 0.01
momentum = 0.9
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# load data
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/Users/saeedshoarayenejati/Downloads/COMP 551/mini project-3/comp-551-w2019-project-3-modified-mnist/data', 
                                train=True, 
                                download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=False)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/Users/saeedshoarayenejati/Downloads/COMP 551/mini project-3/comp-551-w2019-project-3-modified-mnist/test',
                                train=False,
                                download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)

# # let's see some examples :
# examples = enumerate(train_loader)
# batch_idx, (example_data, example_targets) = next(examples)
# example_data.shape

# fig = plt.figure()
# for i in range(5):
#   plt.subplot(3, 3, i+1)
#   plt.tight_layout()
#   plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#   plt.title("Ground Truth: {}".format(example_targets[i]))
#   plt.xticks([])
#   plt.yticks([])
# fig

# 3.Building the Network


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


# initialize the network and the optimizer
network = Net()
optimizer = optim.SGD(network.parameters(), 
                      lr=learning_rate,
                      momentum=momentum)

# 4.Training the Model
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, 
          batch_idx * len(data), 
          len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
          (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(),   '/Users/saeedshoarayenejati/Downloads/COMP 551/mini project-3/comp-551-w2019-project-3-modified-mnist/mini_results/model.pth')
      torch.save(optimizer.state_dict(), '/Users/saeedshoarayenejati/Downloads/COMP 551/mini project-3/comp-551-w2019-project-3-modified-mnist/mini_results/optimizer.pth')

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, 
                                                                            correct, 
                                                                            len(test_loader.dataset),
                                                                            100. * correct / len(test_loader.dataset)))
#   if epoch == 6:
#     np.savetxt(fname='/Users/saeedshoarayenejati/Downloads/COMP 551/mini project-3/comp-551-w2019-project-3-modified-mnist/final_results.csv',
#                X=pred.numpy(), delimiter=',', fmt='%d')
#     print(pred) 
  return pred


test()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()

# # 5.Evaluating the Model's Performance
# fig = plt.figure()
# plt.plot(train_counter, train_losses, color='blue')
# plt.scatter(test_counter, test_losses, color='red')
# plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
# plt.xlabel('number of training examples seen')
# plt.ylabel('negative log likelihood loss')
# fig


# 6.Continued Training from Checkpoints
continued_network = Net()
continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                                momentum=momentum)
network_state_dict = torch.load(
    '/Users/saeedshoarayenejati/Downloads/COMP 551/mini project-3/comp-551-w2019-project-3-modified-mnist/mini_results/model.pth')
continued_network.load_state_dict(network_state_dict)

optimizer_state_dict = torch.load(
    '/Users/saeedshoarayenejati/Downloads/COMP 551/mini project-3/comp-551-w2019-project-3-modified-mnist/mini_results/optimizer.pth')
continued_optimizer.load_state_dict(optimizer_state_dict)

for i in range(4, 9):
  test_counter.append(i*len(train_loader.dataset))
  train(i)
  test()
  np.savetxt(fname='/Users/saeedshoarayenejati/Downloads/COMP 551/mini project-3/comp-551-w2019-project-3-modified-mnist/final_results.csv',
                            X=test().numpy(), delimiter=',', fmt='%d')
