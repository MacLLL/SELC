import torch
import sys

sys.path.append('../..')
import random
from SELC.models.wideresnet import *
import torch.optim as optim
import os
import numpy as np
from dataloader_cifar import cifar_dataloader
import argparse
import torch.nn.functional as F
import pandas as pd
import math

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--noise_mode', default='IDN', help='feature dependent noise from paper SEAL')
parser.add_argument('--model', default='wideresnet', type=str)
parser.add_argument('--op', default='SGD', type=str, help='optimizer')
parser.add_argument('--alpha', default=0.9, help='alpha in SELC')
parser.add_argument('--lr_s', default='MultiStepLR', type=str, help='learning rate scheduler')
parser.add_argument('--loss', default='SELCLoss', type=str, help='loss function')
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--log_interval', default=100, type=int)
parser.add_argument('--r', default=0.1, type=float, help='noise ratio, select from {0.1,0.2,0.3,0.4}')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='', type=str,
                    help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.seed:
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(args.seed)  # CPU seed
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)  # GPU seed
    random.seed(args.seed)  # python seed for image transformation


class SELCLoss(torch.nn.Module):
    def __init__(self, labels, num_classes, es=10, momentum=0.9):
        super(SELCLoss, self).__init__()
        self.num_classes = num_classes
        self.soft_labels = torch.zeros(len(labels), num_classes, dtype=torch.float).cuda()
        self.soft_labels[torch.arange(len(labels)), labels] = 1
        self.es = es
        self.momentum = momentum
        self.CEloss = torch.nn.CrossEntropyLoss()

    def forward(self, logits, labels, index, epoch):
        pred = F.softmax(logits, dim=1)
        if epoch <= self.es:
            ce = self.CEloss(logits, labels)
            return ce.mean()
        else:
            pred_detach = F.softmax(logits.detach(), dim=1)
            self.soft_labels[index] = self.momentum * self.soft_labels[index] \
                                      + (1 - self.momentum) * pred_detach

            selc_loss = -torch.sum(torch.log(pred) * self.soft_labels[index], dim=1)
            return selc_loss.mean()


estimated_es = 40

loader = cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size,
                          num_workers=5,
                          root_dir=args.data_path,
                          noise_file='%s/%.1f_%s.json' % (args.data_path, args.r, args.noise_mode))

all_trainloader, _, clean_labels = loader.run('train')
test_loader = loader.run('test')
eval_train_loader, _, _ = loader.run('eval_train')

# you need to download the IDN noisy labels in github

data = pd.read_table(
    "/IDN_noisy_labels/cifar10_dependent" + str(args.r) + ".csv",
    sep=",")
noisy_labels = data['label_noisy'].values

if args.model == 'wideresnet':
    model = Wide_ResNet(depth=28, widen_factor=10, dropout_rate=0, num_classes=args.num_class).to(args.gpuid)



if args.op == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)


criterion = SELCLoss(noisy_labels, args.num_class, estimated_es, args.alpha)


def train(args, model, train_loader, optimizer, epoch):
    model.train()
    loss_per_batch = []
    correct = 0
    acc_train_per_batch = []

    for batch_idx, (data, _, index) in enumerate(train_loader):
        target = torch.from_numpy(noisy_labels[index])
        data, target = data.to(args.gpuid), target.to(args.gpuid)
        optimizer.zero_grad()
        output= model(data)

        loss = criterion(output, target, index, epoch)

        loss.backward(retain_graph=True)
        optimizer.step()
        loss_per_batch.append(loss.item())

        # save accuracy:
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx + 1) * args.batch_size))

        if batch_idx % args.log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(),
                           100. * correct / ((batch_idx + 1) * args.batch_size),
                    optimizer.param_groups[0]['lr']))
    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]

    return loss_per_epoch, acc_train_per_epoch


def test_cleaning(test_batch_size, model, device, test_loader):
    model.eval()
    loss_per_batch = []
    acc_val_per_batch = []
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc_val_per_batch.append(100. * correct / ((batch_idx + 1) * test_batch_size))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_val_per_epoch = [np.array(100. * correct / len(test_loader.dataset))]

    return loss_per_epoch, acc_val_per_epoch



exp_path = os.path.join('./',
                        'dataset={0}_models={1}_loss={2}_opt={3}_lr_s={4}_epochs={5}_bs={6}_alpha_{7}'.format(
                            args.dataset,
                            args.model,
                            args.loss,
                            args.op,
                            args.lr_s, args.num_epochs,
                            args.batch_size, args.alpha),
                        args.noise_mode + '_noise_rate=' + str(args.r) + '_es=' + str(estimated_es) + '_seed=' + str(
                            args.seed))
if not os.path.isdir(exp_path):
    os.makedirs(exp_path)

cont = 0
acc_train_per_epoch_model = np.array([])
loss_train_per_epoch_model = np.array([])
acc_val_per_epoch_model = np.array([])
loss_val_per_epoch_model = np.array([])

def learning_rate(lr_init, epoch):
    optim_factor = 0
    if(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1
    return lr_init*math.pow(0.2, optim_factor)

for epoch in range(1, args.num_epochs + 1):

    loss_train_per_epoch, acc_train_per_epoch = train(
        args,
        model,
        all_trainloader,
        optimizer,
        epoch)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4)

    # note that we check the accuracy for each epoch below, but the accuracy in paper is recorded from the last epoch
    loss_per_epoch, acc_val_per_epoch_i = test_cleaning(args.batch_size, model, args.gpuid, test_loader)

    acc_train_per_epoch_model = np.append(acc_train_per_epoch_model, acc_train_per_epoch)
    loss_train_per_epoch_model = np.append(loss_train_per_epoch_model, loss_train_per_epoch)
    acc_val_per_epoch_model = np.append(acc_val_per_epoch_model, acc_val_per_epoch_i)
    loss_val_per_epoch_model = np.append(loss_val_per_epoch_model, loss_per_epoch)

    if epoch == 1:
        best_acc_val = acc_val_per_epoch_i[-1]
        snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%.1f_bestAccVal_%.5f' % (
            epoch, loss_per_epoch[-1], acc_val_per_epoch_i[-1], args.r, best_acc_val)
        torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
        torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
    else:
        if acc_val_per_epoch_i[-1] > best_acc_val:
            best_acc_val = acc_val_per_epoch_i[-1]
            if cont > 0:
                try:
                    os.remove(os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
                    os.remove(os.path.join(exp_path, snapBest + '.pth'))
                    # os.remove(os.path.join(exp_path, lossBest))
                except OSError:
                    pass
            snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%.1f_bestAccVal_%.5f' % (
                epoch, loss_per_epoch[-1], acc_val_per_epoch_i[-1], args.r, best_acc_val)
            torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))

    cont += 1

    if epoch == args.num_epochs:
        torch.save(model.state_dict(), os.path.join(exp_path, 'model_last.pth'))
        torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_last.pth'))

np.save(os.path.join(exp_path, 'acc_train_per_epoch_model.npy'), acc_train_per_epoch_model)
np.save(os.path.join(exp_path, 'loss_train_per_epoch_model.npy'), loss_train_per_epoch_model)
np.save(os.path.join(exp_path, 'acc_val_per_epoch_model.npy'), acc_val_per_epoch_model)
np.save(os.path.join(exp_path, 'loss_val_per_epoch_model.npy'), loss_val_per_epoch_model)
