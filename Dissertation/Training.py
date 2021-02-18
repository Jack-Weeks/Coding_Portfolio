import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Models import vgg11_bn, vgg11, vgg13, vgg13_bn
from DataLoader import npy_loader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as tf


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


# def accuracy(predictions,labels):
#     '''
#     Accuracy of a given set of predictions of size (N x n_classes) and
#     labels of size (N x n_classes)
#     '''
#     return np.sum(np.argmax(predictions,axis=1)==np.argmax(labels,axis=1))*100.0/labels.shape[0]

# transforms = tf.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.ToTensor(),
#])
dataset = datasets.DatasetFolder(
    root='NewNift/Patches/train48/',
    loader=npy_loader,
    extensions='.npy',
    #transform= transforms
)
validset = datasets.DatasetFolder(
    root='NewNift/Patches/valid48/',
    loader=npy_loader,
    extensions='.npy',
    #transform= transforms
)
batch_size = 15
learning_rates = [0.001, 0.0001, 1e-4, 1e-5, 1e-6]



num_epoch = 100



net = vgg13_bn()
print(net)
# dataiter = iter(dataset_loader)
# images, labels = dataiter.next()
# images = images.unsqueeze(1)
#
# writer.add_graph(net, images)
criterion = nn.CrossEntropyLoss()


writer = SummaryWriter(comment='train')
writer2 = SummaryWriter(comment='valid')



optimizer = optim.Adam(net.parameters(), lr=0.0001 , betas=(0.9, 0.999), eps=1e-08, weight_decay= 0, amsgrad=False)
## Perform training
for epoch in range(num_epoch):  # loop over the dataset multiple times
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    validset_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True)
    dataiter = iter(dataset_loader)
    images, labels = dataiter.next()
    images = images.unsqueeze(1)

    writer.add_graph(net, images)
    criterion = nn.CrossEntropyLoss()
    print('Epoch: ', epoch)

    running_loss = 0.0
    total_correct = 0.0
    val_correct = 0.0
    test_loss = 0.0

    for i, data in enumerate(dataset_loader):
        # get the inputs; data is a list of [inputs, labels]
        # print('In the epoch ', epoch, ' running iteration: ', i)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)

        # print(data)

        inputs, labels = data
        # print(labels)
        inputs = inputs.unsqueeze(1)

        inputs = inputs.to(device=device)
        # ###########################################################
        labels = labels.to(device=device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float())

        outputs = outputs.to(device=device, dtype=torch.float32)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        # print(preds)
        # print(labels)

        loss.backward()
        optimizer.step()
        torch.no_grad()

        # print statistics
        running_loss += loss.item()
        total_correct += get_num_correct(outputs, labels)

        # Validation

        # if i % 30 == 29:  # print every 100 mini-batches
        #
        #     # for j, val_data in enumerate(validset_loader):
        #     #
        #     #     val_inputs, val_labels = val_data
        #     #     val_inputs = val_inputs.unsqueeze(1)
        #     #     val_inputs = val_inputs.to(device=device)
        #     #     # ###########################################################
        #     #     val_labels = val_labels.to(device=device)
        #     #     val_outputs = net(val_inputs)
        #     #
        #     #     val_loss = criterion(val_outputs, val_labels)
        #     #     loss_cpu = val_loss.cpu().detach().numpy()
        #     #     test_loss = test_loss + loss_cpu
        #     #     val_correct += get_num_correct(val_outputs, val_labels)
        #
        #     print('[%d, %5d] Training loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 30))


                    # test_loss = 0.0
                    # running_loss = 0.0

                # print(total_correct)
    for j, val_data in enumerate(validset_loader):
        val_inputs, val_labels = val_data
        val_inputs = val_inputs.unsqueeze(1)
        val_inputs = val_inputs.to(device=device)
        # ###########################################################
        val_labels = val_labels.to(device=device)
        val_outputs = net(val_inputs)

        val_loss = criterion(val_outputs, val_labels)
        # loss_cpu = val_loss.cpu().detach().numpy()
        test_loss += val_loss.item()
        # print('Correct Lables Are:', val_labels)
        # print('Per Batch accuracy', get_num_correct(val_outputs, val_labels)/len(val_outputs))
        # print('Predicted Labels are:', val_outputs.argmax(dim=1))
        val_correct += get_num_correct(val_outputs, val_labels)

    writer.add_scalar('Training loss', running_loss/len(dataset), epoch)
    writer.add_scalar('Train_Accuracy', total_correct / len(dataset), epoch)
    writer2.add_scalar('Validation Loss', test_loss/len(validset), epoch)
    writer2.add_scalar('Validation Accuracy', val_correct / len(validset), epoch)
    writer.add_scalar('loss', running_loss/len(dataset), epoch)
    writer2.add_scalar('loss', test_loss/len(validset), epoch)
    writer.add_scalar('Accuracy', total_correct / len(dataset), epoch)
    writer2.add_scalar('Accuracy', val_correct / len(validset), epoch)
    print(val_correct)
    print(len(validset))

writer.close()
writer2.close()
torch.cuda.empty_cache()
print('Finished Training')
