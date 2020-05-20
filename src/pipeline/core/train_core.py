import time
import torch
import logging
from .. import config
from .utils import *
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

import numpy as np

def log_config():
    logging.info('\nStarting training with params:')
    logging.info(config.CONFIG_LOG_MESSAGE.format(config.rotation, config.STEPS, config.init_lr))
    logging.info('\n-------')


# def train(network, epoch, train_loader, optimizer, loss, train_losses=[], train_counter = []):
#   network.train()
#   for batch_idx, (data, target) in enumerate(train_loader):
#     data = torch.Tensor(data).to(get_device())
#     optimizer.zero_grad()
#     output, out_image = network(data, target)
#     loss_nn = loss(target, output, data, out_image)
#     loss_nn.backward()
#     optimizer.step()
#     if batch_idx % config.log_interval == 0:
#       train_losses.append(loss_nn.item())
#       train_counter.append(
#         (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

def train(network, epoch, train_loader, optimizer, train_losses=[], train_counter = []):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = torch.Tensor(data).to(get_device())
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target.to(get_device()))
        loss.backward()
        optimizer.step()
        if batch_idx % config.log_interval == 0:
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            logging.info('Loss: {}'.format(loss.item()))
            print('Loss: {}'.format(loss.item()))

def good_train(network, dataset, model_name, train_loader, test_loader, val_loader, n_epochs, optimizer, device, train_losses=[], train_counter = []):
    best_acc= 0
    start = time.time()
    network.train()
    for epoch in range(1, n_epochs + 1):
        print("Epoch: {}".format(epoch))
        iterator = tqdm(train_loader)
        for batch_idx, (data, target) in enumerate(iterator):
            data = torch.Tensor(data).to(device)
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target.to(torch.int64).to(device))
            loss.backward()
            optimizer.step()
            if batch_idx % config.log_interval == 0:
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            # Log average value of loss per epoch
            status = "loss: {:.4f}".format(np.mean(np.array(train_losses)))
            iterator.set_description(status)
        elapsed = (time.time() - start)
        print("Epoch time: {:.2f}".format(elapsed))

        current_acc, _ = test_accuracy(val_loader, network, device)
        if current_acc > best_acc:
            best_acc = current_acc
            print("Best validation accuracy: {}".format(best_acc))
            torch.save(network, './save_model_{}/best_{}.pth'.format(dataset, model_name))
        else:
            print("Current validation accuracy: {}".format(current_acc))


def test_accuracy1(dataloader, model):
    start = time.time()
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm(dataloader)):
            X = X
            y = y
            out = model(X)
            _, argmax = torch.max(out, 1)
            y_pred += [argmax]
            y_true += [y]
        y_pred = torch.cat(y_pred).cpu()
        y_true = torch.cat(y_true).cpu()

    elapsed = (time.time() - start)
    print("Epoch time: {:.2f}".format(elapsed))

    return accuracy_score(y_true, y_pred)*100, y_pred

def test_accuracy(dataloader, model, device):
    start = time.time()
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm(dataloader)):
            X = X.to(device)
            y = y.to(device)
            out = model(X)
            _, argmax = torch.max(out, 1)
            y_pred += [argmax]
            y_true += [y]
        y_pred = torch.cat(y_pred).cpu()
        y_true = torch.cat(y_true).cpu()

    elapsed = (time.time() - start)
    print("Epoch time: {:.2f}".format(elapsed))

    return accuracy_score(y_true, y_pred)*100, y_pred

def test(network, test_loader, loss, test_losses=[], test_counter=[]):
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    acc, y_pred = test_accuracy(test_loader, network)
    if batch_idx % config.log_interval == 0:
      loss_nn = loss(target, output, data, out_image)
      test_losses.append(loss_nn.item())
      test_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
    
    print("acc {0:.2f}%".format(acc))
