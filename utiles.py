import sys
sys.path.append('.')
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam, SGD
from losses import loss_inverse_initial
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import copy

def cal_similarity(test_data, train_data):
    similarity = []
    for i in range(train_data.shape[0]):
        test_data_repeat = test_data.repeat(train_data[i][:, -1, :].shape[0], 1)
        sim = torch.sum(F.cosine_similarity(test_data_repeat, train_data[i][:, -1, :]))
        similarity.append(sim)
    return np.argmax(np.array(similarity))

def x_hat_initialization (test_i, model, x_hat, x, y_true, f_z_bar, f_z_all, BN, threshold=0.45, lr=1e-3, epochs=100):
    input_optimizer = Adam([x_hat], lr=lr)

    initial_x, initial_x_f1 = [], []

    for epoch in range(epochs):
        input_optimizer.zero_grad()

        y_hat = model(x_hat)

        loss, pdf_loss, pdf_loss_all = loss_inverse_initial(y_true, y_hat, x_hat, f_z_bar, f_z_all, BN)
        x_pred = x_hat.clone().cpu().detach().numpy()

        x_pred[x_pred > threshold] = 1
        x_pred[x_pred != 1] = 0
        precision = precision_score(x[0], x_pred[0], average='macro')
        recall = recall_score(x[0], x_pred[0], average='macro')
        f1 = f1_score(x[0], x_pred[0], average='macro')
        accuracy = accuracy_score(x[0], x_pred[0])

        print("Test #{} Epoch: {:2d}".format(test_i + 1, epoch + 1),
              "\tTotal Loss: {:.5f}".format(loss.item()),
              "\tPrec: {:.5f}".format(precision),
              "\tRec: {:.5f}".format(recall),
              "\tF1: {:.5f}".format(f1),
              "\tACC {:.5f}".format(accuracy))

        initial_x.append(copy.deepcopy(x_hat))
        initial_x_f1.append(copy.deepcopy(f1))

        loss.backward()
        input_optimizer.step()

        with torch.no_grad():
            x_hat.clamp_(0, 1)

    return initial_x, initial_x_f1