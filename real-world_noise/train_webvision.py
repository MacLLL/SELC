import torch
import random
import torch.optim as optim
import numpy as np
import dataloader_webvision as dataloader
import argparse
from SELC.models.InceptionResNetV2 import *
import torch.nn.functional as F
import sys

parser = argparse.ArgumentParser(description='PyTorch webvision Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--es', default=40, help='the epoch starts update target')
parser.add_argument('--alpha', default=0.9, help='alpha')
parser.add_argument('--model', default='InceptionResNetV2', type=str)
parser.add_argument('--op', default='SGD', type=str, help='optimizer')
parser.add_argument('--lr_s', default='MultiStepLR', type=str, help='learning rate scheduler')
parser.add_argument('--loss', default='SELCLoss', type=str, help='loss function')
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--log_interval', default=100, type=int)
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=345)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=50, type=int)
parser.add_argument('--data_path', default='/u40/luy100/projects/datasets/webvision1.0/', type=str,
                    help='path to dataset')  # for webvision
parser.add_argument('--dataset', default='webvision', type=str)
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


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0]


def train(log_interval, batch_size, model, device, train_loader, optimizer, epoch):
    model.train()
    loss_per_batch = []
    acc_train_per_batch = []
    correct = 0
    for batch_idx, (data, target, index) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _, _ = model(data)

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
        acc_train_per_batch.append(100. * correct / ((batch_idx + 1) * batch_size))

        if batch_idx % log_interval == 0:
            sys.stdout.write('\r')
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {:.0f}%, Learning rate: {:.6f}\n'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(),
                           100. * correct / ((batch_idx + 1) * batch_size),
                    optimizer.param_groups[0]['lr']))
            output_log.write(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {:.0f}%, Learning rate: {:.6f}\n'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(),
                           100. * correct / ((batch_idx + 1) * batch_size),
                    optimizer.param_groups[0]['lr']))
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
    correct_topk = 0

    test_total = 0
    test_correct_top1 = 0
    test_correct_top5 = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output, _, _ = model(data)

            pred_top5 = accuracy(output, target, (5,))

            # print(acc_top5)
            pred_top1 = accuracy(output, target, (1,))
            # print(acc_top1)
            test_total += 1
            test_correct_top1 += pred_top1.item()
            test_correct_top5 += pred_top5.item()

            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            k = 5

            pred_topk = torch.topk(output, k)[1]
            # print(pred_topk)
            for i in range(k):
                correct_topk += (pred_topk[:, i].view_as(pred)).eq(target.view_as(pred)).sum().item()

            # print(correct_topk)
            acc_val_per_batch.append(100. * correct / ((batch_idx + 1) * test_batch_size))
            acc_val_per_batch.append(100. * correct_topk / ((batch_idx + 1) * test_batch_size))

    test_loss /= len(test_loader.dataset)
    acc_top1 = float(test_correct_top1) / float(test_total)
    acc_top5 = float(test_correct_top5) / float(test_total)

    print(
        '\nTest set: Average loss: {:.4f}, top 1 Accuracy: {:.2f}% top 5 Accuracy: {:.2f}% \n'.format(
            test_loss, acc_top1, acc_top5))

    output_log.write(
        '\nTest set: Average loss: {:.4f}, top 1 Accuracy: {:.2f}% top 5 Accuracy: {:.2f}% \n'.format(
            test_loss, acc_top1, acc_top5))
    output_log.flush()

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_val_per_epoch = [np.array(100. * correct / len(test_loader.dataset))]
    acc_val_per_epoch_topk = [np.array(100. * correct_topk / len(test_loader.dataset))]

    return loss_per_epoch, acc_val_per_epoch, acc_val_per_epoch_topk

loader = dataloader.webvision_dataloader(batch_size=args.batch_size, num_workers=5, root_dir=args.data_path,
                                         num_class=args.num_class)

all_trainloader, noisy_labels = loader.run('train')
train_eval_loader = loader.run('eval_train')
web_valloader = loader.run('test')
imagenet_valloader = loader.run('imagenet')

model = InceptionResNetV2(num_classes=args.num_class).to(args.gpuid)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

if args.lr_s == 'MultiStepLR':
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)

exp_path = os.path.join('./',
                        'data={0}_model={1}_loss={2}_opt={3}_lr_s={4}_epoch={5}_bs={6}_es={8}_alpha={8}'.format(
                            args.dataset,
                            args.model,
                            args.loss,
                            args.op,
                            args.lr_s, args.num_epochs,
                            args.batch_size,
                            args.es,
                            args.alpha), 'seed=' + str(args.seed))
if not os.path.isdir(exp_path):
    os.makedirs(exp_path)

output_log = open(exp_path + '/log.txt', 'w')

if args.loss == 'CE':
    criterion = torch.nn.CrossEntropyLoss()
elif args.loss == 'SELCLoss':
    criterion = SELCLoss(noisy_labels, args.num_class, args.es, args.alpha)

cont = 0
acc_train_per_epoch_model = np.array([])
loss_train_per_epoch_model = np.array([])
acc_val_per_epoch_model_webvision_top1 = np.array([])
acc_val_per_epoch_model_webvision_topk = np.array([])
loss_val_per_epoch_model_webvision = np.array([])
acc_val_per_epoch_model_imagenet_top1 = np.array([])
acc_val_per_epoch_model_imagenet_topk = np.array([])
loss_val_per_epoch_model_imagenet = np.array([])

for epoch in range(1, args.num_epochs + 1):
    print('\t##### Training DNN with SELC #####')
    loss_train_per_epoch, acc_train_per_epoch = train(args.log_interval,
                                                      args.batch_size,
                                                      model,
                                                      args.gpuid,
                                                      all_trainloader,
                                                      optimizer,
                                                      epoch)

    scheduler.step()
    acc_train_per_epoch_model = np.append(acc_train_per_epoch_model, acc_train_per_epoch)
    loss_train_per_epoch_model = np.append(loss_train_per_epoch_model, loss_train_per_epoch)
    # scheduler.step()

    loss_per_epoch_webvision, acc_val_per_epoch_i_webvision, acc_val_per_epoch_i_topk_webvision = test_cleaning(
        args.batch_size, model, args.gpuid, web_valloader)
    loss_per_epoch_imagenet, acc_val_per_epoch_i_imagenet, acc_val_per_epoch_i_topk_imagenet = test_cleaning(
        args.batch_size, model, args.gpuid, imagenet_valloader)
    # for webvision
    acc_val_per_epoch_model_webvision_top1 = np.append(acc_val_per_epoch_model_webvision_top1,
                                                       acc_val_per_epoch_i_webvision)
    acc_val_per_epoch_model_webvision_topk = np.append(acc_val_per_epoch_model_webvision_topk,
                                                       acc_val_per_epoch_i_topk_webvision)
    loss_val_per_epoch_model_webvision = np.append(loss_val_per_epoch_model_webvision, loss_per_epoch_webvision)
    # for imagenet
    acc_val_per_epoch_model_imagenet_top1 = np.append(acc_val_per_epoch_model_imagenet_top1,
                                                      acc_val_per_epoch_i_imagenet)
    acc_val_per_epoch_model_imagenet_topk = np.append(acc_val_per_epoch_model_imagenet_topk,
                                                      acc_val_per_epoch_i_topk_imagenet)
    loss_val_per_epoch_model_imagenet = np.append(loss_val_per_epoch_model_imagenet, loss_per_epoch_imagenet)
    if epoch == 1:
        best_acc_val = acc_val_per_epoch_i_webvision[-1]
        best_acc_val_top5 = acc_val_per_epoch_i_topk_webvision[-1]
        snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f__bestAccVal_%.5f_bestAccValtop5_%.5f' % (
            epoch, loss_per_epoch_webvision[-1], acc_val_per_epoch_i_webvision[-1], best_acc_val,
            best_acc_val_top5)
        torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
        torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))

    else:
        if acc_val_per_epoch_i_webvision[-1] > best_acc_val:
            best_acc_val = acc_val_per_epoch_i_webvision[-1]
            best_acc_val_top5 = acc_val_per_epoch_i_topk_webvision[-1]
            if cont > 0:
                try:
                    os.remove(os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
                    os.remove(os.path.join(exp_path, snapBest + '.pth'))
                    # os.remove(os.path.join(exp_path, lossBest))
                except OSError:
                    pass
            snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_bestAccVal_%.5f_bestAccValtop5_%.5f' % (
                epoch, loss_per_epoch_webvision[-1], acc_val_per_epoch_i_webvision[-1], best_acc_val,
                best_acc_val_top5)
            torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
            # _, corrected_labels = torch.max(criterion.soft_labels, dim=1)
            # corrected_labels = corrected_labels.detach().cpu().numpy()
            # np.save(os.path.join(exp_path, 'corrected_labels_{0}.npy'.format(epoch)), corrected_labels)
    cont += 1



np.save(os.path.join(exp_path, 'acc_train_per_epoch_model.npy'), acc_train_per_epoch_model)

np.save(os.path.join(exp_path, 'loss_train_per_epoch_model.npy'), loss_train_per_epoch_model)
np.save(os.path.join(exp_path, 'acc_val_per_epoch_model_webvision_top1.npy'), acc_val_per_epoch_model_webvision_top1)
np.save(os.path.join(exp_path, 'acc_val_per_epoch_model_webvision_topk.npy'), acc_val_per_epoch_model_webvision_topk)
np.save(os.path.join(exp_path, 'loss_val_per_epoch_model_webvision.npy'), loss_val_per_epoch_model_webvision)

np.save(os.path.join(exp_path, 'acc_val_per_epoch_model_imagenet_top1.npy'), acc_val_per_epoch_model_imagenet_top1)
np.save(os.path.join(exp_path, 'acc_val_per_epoch_model_imagenet_topk.npy'), acc_val_per_epoch_model_imagenet_topk)
np.save(os.path.join(exp_path, 'loss_val_per_epoch_model_imagenet.npy'), loss_val_per_epoch_model_imagenet)
