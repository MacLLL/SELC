import torch
import random
from SELC.models.resnet import *
import torch.optim as optim
import os
import sys
import numpy as np
from dataloader_cifar import cifar_dataloader
import argparse
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode', default='sym', help='sym or asym')
parser.add_argument('--model', default='resnet34', type=str)
parser.add_argument('--op', default='SGD', type=str, help='optimizer')
parser.add_argument('--alpha', default=0.9, help='alpha in SELC')
parser.add_argument('--lr_s', default='MultiStepLR', type=str, help='learning rate scheduler')
parser.add_argument('--loss', default='SELCLoss', type=str, help='loss function')
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--log_interval', default=100, type=int)
parser.add_argument('--r', default=0.4, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='', type=str, help='path to dataset')
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


estimated_es = {'cifar10_sym0.2': 40, 'cifar10_sym0.4': 30, 'cifar10_sym0.6': 30, 'cifar10_sym0.8': 40,
                'cifar10_asym0.4': 40,
                'cifar100_sym0.2': 30, 'cifar100_sym0.4': 20, 'cifar100_sym0.6': 30, 'cifar100_sym0.8': 40,
                'cifar100_asym0.4': 20}

loader = cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size,
                          num_workers=5,
                          root_dir=args.data_path,
                          noise_file='%s/%.1f_%s.json' % (args.data_path, args.r, args.noise_mode))

all_trainloader, noisy_labels, clean_labels = loader.run('train')
test_loader = loader.run('test')
eval_train_loader, _, _ = loader.run('eval_train')

if args.model == 'resnet34':
    model = ResNet34(num_classes=args.num_class).to(args.gpuid)

if args.op == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

if args.lr_s == 'MultiStepLR':
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)

criterion = SELCLoss(noisy_labels, args.num_class, estimated_es[args.dataset + '_' + args.noise_mode + str(args.r)],
                     args.alpha)


def train(args, model, train_loader, optimizer, epoch):
    model.train()
    loss_per_batch = []
    correct = 0
    acc_train_per_batch = []

    for batch_idx, (data, target, index) in enumerate(train_loader):

        data, target = data.to(args.gpuid), target.to(args.gpuid)
        optimizer.zero_grad()
        output, _, _ = model(data)

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
            output, _, _ = model(data)
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
                        args.noise_mode + str(args.r) + '_es=' + str(
                            estimated_es[args.dataset + '_' + args.noise_mode + str(args.r)]) + '_seed=' + str(
                            args.seed))
if not os.path.isdir(exp_path):
    os.makedirs(exp_path)

t = torch.zeros(50000, args.num_class).to(args.gpuid)
cont = 0
acc_train_per_epoch_model = np.array([])
loss_train_per_epoch_model = np.array([])
acc_val_per_epoch_model = np.array([])
loss_val_per_epoch_model = np.array([])

nn_acc_per_epoch_list = np.array([])
for epoch in range(1, args.num_epochs + 1):

    loss_train_per_epoch, acc_train_per_epoch = train(
        args,
        model,
        all_trainloader,
        optimizer,
        epoch)
    scheduler.step()

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

# save corrected labels
_, corrected_labels = torch.max(criterion.soft_labels, dim=1)
correct_num = np.sum(corrected_labels.cpu().numpy() == np.array(clean_labels))
corrected_acc = correct_num / len(clean_labels)
print('Corrected accuracy = {}/{} = {}'.format(correct_num, len(clean_labels), corrected_acc))

np.save(os.path.join(exp_path, 'corrected_labels_%.4f.npy' % (corrected_acc)),
        corrected_labels.cpu().numpy())

np.save(os.path.join(exp_path, 'acc_train_per_epoch_model.npy'), acc_train_per_epoch_model)
np.save(os.path.join(exp_path, 'loss_train_per_epoch_model.npy'), loss_train_per_epoch_model)
np.save(os.path.join(exp_path, 'acc_val_per_epoch_model.npy'), acc_val_per_epoch_model)
np.save(os.path.join(exp_path, 'loss_val_per_epoch_model.npy'), loss_val_per_epoch_model)
