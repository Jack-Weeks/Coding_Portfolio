from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from collections import Counter
from tqdm import tqdm
import PIL
from PIL import Image
import pandas as pd
import datetime
from torch.utils.tensorboard import SummaryWriter
from timeit import default_timer as timer
import seaborn as sns


def train(model,
          criterion,
          optimizer,
          train_loader,
          valid_loader,
          save_file_name,
          scheduler,
          max_epochs_stop=5,
          n_epochs=20,
          print_every=1,
          ):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()

    # Main loop
    for epoch in tqdm(range(n_epochs)):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # Set to training
        model.train()
        start = timer()

        # Training loop
        for ii, (data, target) in enumerate(train_loader):
            # Tensors to gpu
            data, target = data.cuda(), target.cuda()

            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            output = model(data)

            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()
            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)

            # Track training progress
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                end='\r')

        # After training loops ends, start validation
        else:

            model.epochs += 1

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data, target in valid_loader:
                    # Tensors to gpu
                    data, target = data.cuda(), target.cuda()

                    # Forward pass
                    output = model(data)

                    # Validation loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)

                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(
                        correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)

                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)

                # Calculate average accuracy
                train_acc = train_acc / len(train_loader.dataset)
                valid_acc = valid_acc / len(valid_loader.dataset)
                scheduler.step()

                history.append([train_loss, valid_loss, train_acc, valid_acc])

                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    print(
                        f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                    )
                    print(
                        f'\t\tTrainig Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                    )

                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # Save model
                    torch.save(model.state_dict(), save_file_name)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    best_epoch = epoch

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(
                            f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_best_acc:.2f}%'
                        )
                        total_time = timer() - overall_start
                        print(
                            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch + 1):.2f} seconds per epoch.'
                        )

                        # Load the best state dict
                        model.load_state_dict(torch.load(save_file_name))
                        # Attach the optimizer
                        model.optimizer = optimizer

                        # Format history
                        history = pd.DataFrame(
                            history,
                            columns=[
                                'train_loss', 'valid_loss', 'train_acc',
                                'valid_acc'
                            ])
                        return model, history

        writer.add_scalar('Loss', train_loss, epoch)
        writer2.add_scalar('Loss', valid_loss, epoch)
        writer.add_scalar('Accuracy', train_acc, epoch)
        writer2.add_scalar('Accuracy', valid_acc, epoch)

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start

    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
    )
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history



if __name__ == '__main__':

    data_dir = '../cartoon_set/img/'
    input_size = 410
    batch_size = 16
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(224),
            transforms.RandomGrayscale(p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=35),
            transforms.ColorJitter(0.75,0.75),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(224),
            # transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                      ['train', 'valid']}

    # full_dataset = datasets.ImageFolder('C:/Users/weeks/PycharmProjects/IPMI/MA/Merge_Data/train/',
    # transform=data_transforms['train'])

    batch_sizes = [8]
    learning_rates = [0.001]
    momentums = [0.9]
    model_list = ['vgg11_bn']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dict = {}
    num_epochs = 10


    categories = []
    for d in os.listdir(data_dir + '/train'):
        categories.append(d)

    n_classes = len(categories)
    print(f'There are {n_classes} different classes.')

    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for momentum in momentums:
                for model_choice in model_list:

                    writer = SummaryWriter(comment=str('train' + str(batch_size) + str(learning_rate) + str(momentum) + model_choice))
                    writer2 = SummaryWriter(comment='valid' + str(batch_size) + str(learning_rate) + str(momentum) + model_choice)

                    dataloaders_dict = {
                        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True,
                                                       num_workers=5) for x in ['train', 'valid']}

                    if model_choice == 'vgg11':
                        model = models.vgg11(pretrained=True)

                        for param in model.parameters():
                            param.requires_grad = True

                        n_inputs = model.classifier[6].in_features
                        model.classifier[6] = nn.Sequential(
                            nn.Linear(n_inputs, 2048), nn.ReLU(), nn.Dropout(0.65),
                            nn.Linear(2048, 1024), nn.ReLU(),
                            nn.Dropout(0.7),
                            nn.Linear(1024, 512), nn.ReLU(),
                            nn.Linear(512, n_classes))
                        model = model.to(device)

                    elif model_choice == 'vgg11_bn':
                        model = models.vgg11_bn(pretrained=True)

                        for param in model.parameters():
                            param.requires_grad = True

                        n_inputs = model.classifier[6].in_features
                        model.classifier[6] = nn.Sequential(
                            nn.Linear(n_inputs, 2048), nn.ReLU(), nn.Dropout(0.65),
                            nn.Linear(2048, 1024), nn.ReLU(),
                            nn.Dropout(0.7),
                            nn.Linear(1024, 512), nn.ReLU(),
                            nn.Linear(512, n_classes))

                        model = model.to(device)

                    # elif model_choice == 'vgg16':
                    #     model = models.vgg16(pretrained=True)
                    #
                    #     for param in model.parameters():
                    #         param.requires_grad = True
                    #
                    #     n_inputs = model.classifier[6].in_features
                    #     model.classifier[6] = nn.Sequential(
                    #         nn.Linear(n_inputs, 2048), nn.ReLU(), nn.Dropout(0.65),
                    #         nn.Linear(2048, 1024), nn.ReLU(),
                    #         nn.Dropout(0.6),
                    #         nn.Linear(1024, 512), nn.ReLU(),
                    #         nn.Linear(512, n_classes))
                    #
                    #     model = model.to(device)
                    #
                    # elif model_choice == 'vgg19':
                    #     model = models.vgg19(pretrained=True)
                    #
                    #     for param in model.parameters():
                    #         param.requires_grad = True
                    #
                    #     n_inputs = model.classifier[6].in_features
                    #     model.classifier[6] = nn.Sequential(
                    #         nn.Linear(n_inputs, 2048), nn.ReLU(), nn.Dropout(0.65),
                    #         nn.Linear(2048, 1024), nn.ReLU(),
                    #         nn.Dropout(0.6),
                    #         nn.Linear(1024, 512), nn.ReLU(),
                    #         nn.Linear(512, n_classes))
                    #
                    #     model = model.to(device)
                    #
                    #
                    # elif model_choice == 'resnet152':
                    #
                    #     model = models.resnet152(pretrained=True)
                    #
                    #     for param in model.parameters():
                    #         param.requires_grad = True
                    #
                    #     num_inputs = model.fc.in_features
                    #     model.fc = nn.Sequential(nn.Linear(num_inputs, n_classes))
                    #     model = model.to(device)
                    #
                    # elif model_choice == 'resnet101':
                    #
                    #     model = models.resnet101(pretrained=True)
                    #
                    #     for param in model.parameters():
                    #         param.requires_grad = True
                    #
                    #     num_inputs = model.fc.in_features
                    #     model.fc = nn.Sequential(nn.Linear(num_inputs, n_classes))
                    #     model = model.to(device)
                    #
                    # if model_choice == 'resnet50':
                    #
                    #     model = models.resnet50(pretrained=True)
                    #
                    #     for param in model.parameters():
                    #         param.requires_grad = True
                    #
                    #     num_inputs = model.fc.in_features
                    #     model.fc = nn.Sequential(nn.Linear(num_inputs, n_classes))
                    #     model = model.to(device)

                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.01, momentum=momentum)
                    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)

                    NEWPATH = 'Outputs/'
                    save_path = os.path.join(NEWPATH + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.pth')
                    checkpoint_path = NEWPATH

                    model, history = train(model, criterion, optimizer, dataloaders_dict['train'],
                                           dataloaders_dict['valid'],
                                           save_path,
                                           scheduler=exp_lr_scheduler,
                                           max_epochs_stop=25, n_epochs=num_epochs,
                                           )

                    print('')
                    print('Batch Size', batch_size, 'Learning_rate', learning_rate)
                    print('')
                    print('Momentum', momentum, 'model', model_choice)

                    # writer.add_scalar('Training Loss', history['train_loss'])
                    # writer2.add_scalar('Validation Loss', history['valid_loss'])
                    # writer.add_scalar('Training Accuracy', history['train_acc'])
                    # writer2.add_scalar('Validation Accuracy', history['valid_acc'])

                    writer.close()
                    writer2.close()

                    confusion_matrix = torch.zeros(n_classes, n_classes)
                    with torch.no_grad():
                        for i, (inputs, classes) in enumerate(dataloaders_dict['valid']):
                            inputs = inputs.to(device)
                            classes = classes.to(device)
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            for t, p in zip(classes.view(-1), preds.view(-1)):
                                confusion_matrix[t.long(), p.long()] += 1

                    print(confusion_matrix)

                    x = sns.heatmap(confusion_matrix, xticklabels = categories, yticklabels = categories,annot=True)
                    plt.savefig('Outputs/Confusion_matrix' + str(batch_size) + str(learning_rate) + str(
                        momentum) + model_choice + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')
                    plt.show()

                    plt.figure(figsize=(8, 6))
                    for c in ['train_loss', 'valid_loss']:
                        plt.plot(
                            history[c], label=c)
                    plt.legend()
                    plt.xlabel('Epoch')
                    plt.ylabel('Average Loss')
                    plt.title('Training and Validation Losses')

                    plt.savefig('Outputs/loss' + str(batch_size) + str(learning_rate) + str(
                        momentum) + model_choice + '.png')
                    plt.show()

                    plt.figure(figsize=(8, 6))
                    for c in ['train_acc', 'valid_acc']:
                        plt.plot(
                            100 * history[c], label=c)
                    plt.legend()
                    plt.xlabel('Epoch')
                    plt.ylabel('Average Accuracy')
                    plt.title('Training and Validation Accuracy')

                    plt.savefig('Outputs/acc' + str(batch_size) + str(learning_rate) + str(
                        momentum) + model_choice + '.png')
                    plt.show()

                    output_dict[str(batch_size) + str(learning_rate) + str(momentum) + model_choice] = history

                    torch.cuda.empty_cache()
