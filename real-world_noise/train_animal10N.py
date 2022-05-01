import sys
sys.path.append('..')
import os
import torch.nn as nn
import torch.nn.parallel
import random
import argparse
import numpy as np
from SELC.models.vgg import vgg19_bn
from data_prepare_animal10N import Animal10
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
from SELC.utils.utils import check_folder
from termcolor import cprint


# random seed related
def _init_fn(worker_id):
    np.random.seed(77 + worker_id)


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


def main(arg_seed, arg_timestamp):
    random_seed = arg_seed
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True  # need to set to True as well

    # print('Random Seed {}\n'.format(arg_seed))

    # -- training parameters
    num_epoch = args.epoch
    milestone = [50, 75]
    batch_size = args.batch
    num_workers = 2

    weight_decay = 1e-3
    gamma = 0.2

    lr = args.lr
    start_epoch = 0

    # -- specify dataset
    # data augmentation
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    trainset = Animal10(split='train', data_path=args.data_path, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                              worker_init_fn=_init_fn, drop_last=True)

    testset = Animal10(split='test', data_path=args.data_path, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size * 4, shuffle=False, num_workers=num_workers)

    num_class = 10

    print('train data size:', len(trainset))
    print('test data size:', len(testset))

    # -- create log file
    if arg_timestamp:
        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        file_name = 'Ours(' + time_stamp + ').txt'
    else:
        file_name = 'Ours.txt'

    log_dir = check_folder('logs')
    file_name = os.path.join(log_dir, file_name)
    saver = open(file_name, "w")

    saver.write(args.__repr__() + "\n\n")
    saver.flush()

    # -- set network, optimizer, scheduler, etc
    net = vgg19_bn(num_classes=num_class, pretrained=False)
    net = nn.DataParallel(net)

    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestone, gamma=gamma)
    # criterion = torch.nn.CrossEntropyLoss()
    noisy_labels = trainset.return_corrupted_label()
    criterion = SELCLoss(noisy_labels, num_class, args.es, args.alpha)

    # -- misc
    iterations = 0
    f_record = torch.zeros([args.rollWindow, len(trainset), num_class])

    for epoch in range(start_epoch, num_epoch):
        train_correct = 0
        train_loss = 0
        train_total = 0

        net.train()

        for i, (images, labels, indices) in enumerate(trainloader):
            if images.size(0) == 1:  # when batch size equals 1, skip, due to batch normalization
                continue

            images, labels = images.to(device), labels.to(device)

            outputs, _ = net(images)
            # loss = criterion(outputs, labels)

            loss = criterion(outputs, labels, indices, epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_total += images.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()

            f_record[epoch % args.rollWindow, indices] = F.softmax(outputs.detach().cpu(), dim=1)

            iterations += 1
            if iterations % 100 == 0:
                cur_train_acc = train_correct / train_total * 100.
                cur_train_loss = train_loss / train_total
                cprint('epoch: {}\titerations: {}\tcurrent train accuracy: {:.4f}\ttrain loss:{:.4f}'.format(
                    epoch, iterations, cur_train_acc, cur_train_loss), 'yellow')

                if iterations % 5000 == 0:
                    saver.write('epoch: {}\titerations: {}\ttrain accuracy: {}\ttrain loss: {}\n'.format(
                        epoch, iterations, cur_train_acc, cur_train_loss))
                    saver.flush()

        train_acc = train_correct / train_total * 100.

        cprint('epoch: {}'.format(epoch), 'yellow')
        cprint('train accuracy: {:.4f}\ntrain loss: {:.4f}'.format(train_acc, train_loss), 'yellow')
        saver.write('epoch: {}\ntrain accuracy: {}\ntrain loss: {}\n'.format(epoch, train_acc, train_loss))
        saver.flush()

        exp_lr_scheduler.step()

        # testing
        net.eval()
        test_total = 0
        test_correct = 0
        with torch.no_grad():
            for i, (images, labels, _) in enumerate(testloader):
                images, labels = images.to(device), labels.to(device)

                outputs, _ = net(images)

                test_total += images.size(0)
                _, predicted = outputs.max(1)
                test_correct += predicted.eq(labels).sum().item()

            test_acc = test_correct / test_total * 100.

        cprint('>> current test accuracy: {:.4f}'.format(test_acc), 'cyan')

        saver.write('>> current test accuracy: {}\n'.format(test_acc))
        saver.flush()

    saver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default='1', help='delimited list input of GPUs', type=str)
    parser.add_argument('--timelog', action='store_false', help='whether to add time stamp to log file name')
    parser.add_argument("--es", default=40, type=int)
    parser.add_argument("--rollWindow", default=10, help="rolling window to calculate the confidence", type=int)
    parser.add_argument("--eval_freq", default=2000, help="evaluation frequency (every a few iterations)", type=int)
    parser.add_argument("--batch", default=128, help="batch size", type=int)
    parser.add_argument("--epoch", default=100, help="total number of epochs", type=int)
    parser.add_argument("--seed", default=77, help="random seed", type=int)
    parser.add_argument("--lr", default=0.1, help="learning rate", type=float)
    parser.add_argument("--alpha", default=0.9, help="alpha", type=float)
    parser.add_argument('--data_path', default='', type=str, help='path to dataset')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    seed = args.seed

    main(seed, args.timelog)
