import torch
import random
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import dataloader_clothing1M as dataloader
import argparse
import torch.nn.functional as F
import torchvision.models as models
import torchextractor as tx

parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batch size')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--model', default='resnet50', type=str)
parser.add_argument('--op', default='SGD', type=str, help='optimizer')
parser.add_argument('--es', default=5, help='the epoch starts update target')
parser.add_argument('--alpha', default=0.2, help='alpha in backward')
parser.add_argument('--warmup', default=1, help='warm up epochs')
parser.add_argument('--lr_s', default='MultiStepLR', type=str, help='learning rate scheduler')
parser.add_argument('--loss', default='SELCLoss', type=str, help='loss function')
parser.add_argument('--num_epochs', default=40, type=int)
parser.add_argument('--log_interval', default=100, type=int)
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=234)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=14, type=int)
parser.add_argument('--data_path', default='/u40/luy100/projects/datasets/clothing1m', type=str, help='path to dataset')
parser.add_argument('--num_batches', default=2000, type=int)
parser.add_argument('--dataset', default='clothing1M', type=str)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# fix the seed, for experiment
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
        self.soft_labels = torch.zeros(labels.shape[0], num_classes, dtype=torch.float).to(args.gpuid)
        self.soft_labels[torch.arange(labels.shape[0]), labels] = 1
        self.es = es
        self.momentum = momentum
        self.CELoss = torch.nn.CrossEntropyLoss()

    def forward(self, logits, labels, index, epoch):
        pred = F.softmax(logits, dim=1)
        if epoch <= self.es:
            ce = self.CELoss(logits, labels)
            return ce.mean()
        else:
            pred_detach = F.softmax(logits.detach(), dim=1)
            self.soft_labels[index] = self.momentum * self.soft_labels[index] \
                                      + (1 - self.momentum) * pred_detach

            selc_loss = -torch.sum(torch.log(pred) * self.soft_labels[index], dim=1)
            return selc_loss.mean()


def train(args, model, train_loader, optimizer, epoch):
    model.train()
    loss_per_batch = []
    acc_train_per_batch = []
    correct = 0
    for batch_idx, (data, target, index) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, features = model(data)
        if args.loss == 'SELCLoss':
            loss = criterion(output, target, index, epoch)
        else:
            loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        loss_per_batch.append(loss.item())

        # save accuracy:
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx + 1) * args.batch_size))

        if batch_idx % args.log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {:.0f}%, Learning rate: {:.6f}\n'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(),
                           100. * correct / ((batch_idx + 1) * args.batch_size), optimizer.param_groups[0]['lr']))
            output_log.write(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {:.0f}%, Learning rate: {:.6f}\n'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(),
                           100. * correct / ((batch_idx + 1) * args.batch_size), optimizer.param_groups[0]['lr']))
            output_log.flush()

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
            output, _ = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc_val_per_batch.append(100. * correct / ((batch_idx + 1) * test_batch_size))

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%).\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    output_log.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%).\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    output_log.flush()

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_val_per_epoch = [np.array(100. * correct / len(test_loader.dataset))]

    return loss_per_epoch, acc_val_per_epoch


def test_val(val_batch_size, model, device, val_loader):
    model.eval()
    loss_per_batch = []
    acc_val_per_batch = []
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

            correct += pred.eq(target.view_as(pred)).sum().item()
            acc_val_per_batch.append(100. * correct / ((batch_idx + 1) * val_batch_size))

    test_loss /= len(val_loader.dataset)

    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%).\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

    output_log.write('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%).\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    output_log.flush()

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_val_per_epoch = [np.array(100. * correct / len(val_loader.dataset))]

    return loss_per_epoch, acc_val_per_epoch


loader = dataloader.clothing_dataloader(root=args.data_path, batch_size=args.batch_size,
                                        num_workers=16,
                                        num_batches=args.num_batches)

train_loader, noisy_labels = loader.run('train')
# train_eval_loader,_ = loader.run('eval_train')
test_loader = loader.run('test')
val_loader = loader.run('val')


def create_model():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, args.num_class)
    model = tx.Extractor(model, ['avgpool'])
    model = model.to(args.gpuid)
    return model


model = create_model()

if args.op == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

if args.lr_s == 'MultiStepLR':
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

if args.loss == 'CE':
    criterion = torch.nn.CrossEntropyLoss()
elif args.loss == 'SELCLoss':
    criterion = SELCLoss(noisy_labels, args.num_class, args.es, 1 - args.alpha)

exp_path = os.path.join('./',
                        'data={0}_model={1}_loss={2}_opt={3}_lr_s={4}_epoch={5}_bs={6}_batches={7}_es={8}_alpha={9}'.format(
                            args.dataset,
                            args.model,
                            args.loss,
                            args.op,
                            args.lr_s, args.num_epochs,
                            args.batch_size,
                            args.num_batches, args.es, args.alpha), 'seed=' + str(args.seed))

if not os.path.isdir(exp_path):
    os.makedirs(exp_path)

output_log = open(exp_path + '/log.txt', 'w')

cont = 0
acc_train_per_epoch_model = np.array([])
acc_train_noisy_per_epoch_model = np.array([])
acc_train_clean_per_epoch_model = np.array([])
loss_train_per_epoch_model = np.array([])
acc_val_per_epoch_model = np.array([])
loss_val_per_epoch_model = np.array([])
prob = 0
for epoch in range(1, args.num_epochs + 1):

    loss_train_per_epoch, acc_train_per_epoch = train(args,
                                                      model,
                                                      train_loader,
                                                      optimizer,
                                                      epoch)

    scheduler.step()
    loss_per_epoch, acc_val_per_epoch_i = test_cleaning(args.batch_size, model, args.gpuid, test_loader)
    val_loss, val_acc = test_val(args.batch_size, model, args.gpuid, val_loader)
    acc_train_per_epoch_model = np.append(acc_train_per_epoch_model, acc_train_per_epoch)
    loss_train_per_epoch_model = np.append(loss_train_per_epoch_model, loss_train_per_epoch)
    acc_val_per_epoch_model = np.append(acc_val_per_epoch_model, acc_val_per_epoch_i)
    loss_val_per_epoch_model = np.append(loss_val_per_epoch_model, loss_per_epoch)
    if epoch == 1:
        best_acc_val = acc_val_per_epoch_i[-1]
        snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_bestAccVal_%.5f' % (
            epoch, loss_per_epoch[-1], acc_val_per_epoch_i[-1], best_acc_val)
        torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
        torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
    else:
        if acc_val_per_epoch_i[-1] > best_acc_val:
            best_acc_val = acc_val_per_epoch_i[-1]
            if cont > 0:
                try:
                    os.remove(os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
                    os.remove(os.path.join(exp_path, snapBest + '.pth'))
                except OSError:
                    pass
            snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_bestAccVal_%.5f' % (
                epoch, loss_per_epoch[-1], acc_val_per_epoch_i[-1], best_acc_val)
            torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))

    cont += 1

np.save(os.path.join(exp_path, 'acc_train_per_epoch_model.npy'), acc_train_per_epoch_model)
np.save(os.path.join(exp_path, 'loss_train_per_epoch_model.npy'), loss_train_per_epoch_model)
np.save(os.path.join(exp_path, 'acc_val_per_epoch_model.npy'), acc_val_per_epoch_model)
np.save(os.path.join(exp_path, 'loss_val_per_epoch_model.npy'), loss_val_per_epoch_model)
