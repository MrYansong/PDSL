# coding: utf-8
import torch
import torch.nn as nn
import numpy as np
import copy
import random
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

from modules import CVAEModel, DiffusionODE, Encoder, Decoder, MLPTransform
from Train_Infer import TrainingModel, InferenceModel
from utiles import cal_similarity, x_hat_initialization
from losses import loss_seed_x, loss_all
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from torch.optim import Adam
import pickle
import os

def run(args):
    print('Is GPU available? {}\n'.format(torch.cuda.is_available()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print('use device: ', device)
    print('dataset: ', args.dataset)

    with open('../data/used_data/' + args.dataset + '.pkl', 'rb') as f:
        graph = pickle.load(f)

    adj, inverse_pairs, prob_matrix = graph['adj'].toarray(), graph['inverse_pairs'], graph['weighted_adj'].toarray()
    np.fill_diagonal(adj, 1)
    np.fill_diagonal(prob_matrix, 1)
    tensor_adj = torch.tensor(adj).float().to(device)
    adj_idx = torch.nonzero(tensor_adj).T.to(device)

    num_training = int(len(inverse_pairs) * args.train_ratio)
    train_set = inverse_pairs[:num_training]
    test_set = inverse_pairs[num_training:]

    nodes_num = inverse_pairs.shape[3]
    Time_dim = inverse_pairs.shape[2] - 2

    encoder = Encoder( input_dim=nodes_num + Time_dim * nodes_num, hidden_dim=args.encoder_hidden, latent_dim=args.encoder_latent)
    decoder = Decoder( input_dim=args.encoder_latent + Time_dim * nodes_num, latent_dim=args.encoder_hidden, hidden_dim=args.encoder_latent, output_dim=nodes_num)
    cvae_model = CVAEModel(Encoder=encoder, Decoder=decoder)
    propagate = DiffusionODE(edge_index=adj_idx, node_size=nodes_num, input_dim=1, out_dim=args.ODE_outdim, t_size=args.ODE_t_size)

    MLP = MLPTransform(input_dim=args.ODE_outdim, hiddenunits=args.MLP_hidden, num_classes=1, prob_matrix=adj)

    model = TrainingModel(cvae_model, propagate, MLP).to(device)

    model = model.to(device)

    eval_vae_model, eval_forward_model = train(args, train_set, model)

    opt_thres = 0.55

    print('--------------------------------------------------------------------------------')
    print('opt_thres: ', opt_thres)
    precision, recall, f1, acc= eval(args, train_set, test_set, eval_vae_model, eval_forward_model, opt_thres)
    return precision, recall, f1, acc



def train(args, train_set, model):
    device = next(model.parameters()).device
    model.train()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    sample_number = train_set[:].shape[0] * train_set[:].shape[1]
    for epoch in range(args.max_train_epochs):
        re_overall = 0
        kld_overall = 0
        forward_overall = 0
        total_overall = 0
        precision_all = 0
        recall_all = 0
        f1_all = 0

        for batch_idx, data_pair in enumerate(train_set):
            x_true = data_pair[:, 0, :].float().to(device)
            x_condition = data_pair[:, 1:-1, :].reshape(x_true.shape[0], -1).float().to(device)
            y = data_pair[:, -1, :].to(device)
            optimizer.zero_grad()
            seed_hat,  mean, log_var, y_hat = model(x_true, x_condition)
            forward_loss, re_loss, kld, loss = loss_all( x_true, seed_hat, log_var,mean, y_hat, y)
            x_pred = seed_hat.clone().cpu().detach()
            kld_overall += kld.item() * seed_hat.size(0)
            re_overall += re_loss.item() * seed_hat.size(0)
            total_overall += loss.item() * seed_hat.size(0)
            forward_overall += forward_loss.item() * seed_hat.size(0)

            for i in range(x_true.shape[0]):
                x_pred[i][x_pred[i] >= 0.55] = 1
                x_pred[i][x_pred[i] != 1] = 0
                precision_all += precision_score(x_true[i].cpu().detach().numpy(), x_pred[i].cpu().detach().numpy(),zero_division=0, average='macro')
                recall_all += recall_score(x_true[i].cpu().detach().numpy(), x_pred[i].cpu().detach().numpy(),zero_division=0, average='macro')
                f1_all += f1_score(x_true[i].cpu().detach().numpy(), x_pred[i].cpu().detach().numpy(),zero_division=0, average='macro')

            loss.backward()
            optimizer.step()

        print("Epoch: {}".format(epoch + 1),
              "\tReconstruction: {:.4f}".format(re_overall / sample_number),
              "\tKLD: {:.4f}".format(kld_overall / sample_number),
              "\tforward loss: {:.4f}".format(forward_overall / sample_number),
              "\tTotal: {:.4f}".format(total_overall / sample_number),
              "\tPrecision: {:.4f}".format(precision_all / sample_number),
              "\tRecall: {:.4f}".format(recall_all / sample_number),
              "\tf1: {:.4f}".format(f1_all / sample_number))

        if (recall_all / sample_number) > 0.8: #early stopping
            break

    eval_vae_model = model.vae_model
    eval_forward_model = InferenceModel(model.mlp_model, model.propagate).to(device)

    return eval_vae_model, eval_forward_model

def eval (args, train_set, eval_set, eval_vae_model, eval_forward_model, thres):
    device = next(eval_vae_model.parameters()).device
    batch_precision_all = 0
    batch_recall_all = 0
    batch_f1_all = 0
    batch_acc_all = 0

    for param in eval_vae_model.parameters():
        param.requires_grad = False

    for param in eval_forward_model.parameters():
        param.requires_grad = False

    eval_encoder = eval_vae_model.Encoder
    eval_decoder = eval_vae_model.Decoder
    x_pred_all = []

    for eval_id, eval in enumerate(eval_set):
        precision_all = 0
        recall_all = 0
        f1_all = 0
        acc_all = 0
        batch_x_true = []
        batch_x_pred = []
        for i in range(eval.shape[0]):
            x_true = torch.tensor(eval[i, 0, :]).float().unsqueeze(0).to(device)
            y_true = torch.tensor(eval[i, -1, :]).float().unsqueeze(0).to(device)
            index = cal_similarity(y_true.cpu(), train_set)
            train_x = torch.tensor(train_set[index, :, :, :][:, 0, :]).float().to(device)
            test_condition_bar = eval[i, 1:-1, :].reshape(1, -1).float().to(device)
            test_condition_all = test_condition_bar.repeat(train_x.shape[0], 1)

            with torch.no_grad():
                mean, var = eval_encoder(train_x, test_condition_all)
                train_z_all = eval_vae_model.reparameterization(mean, var).to(device)
                train_z_bar = torch.mean(train_z_all, dim=0).unsqueeze(0).to(device)
                train_z_all = eval_decoder(train_z_all, test_condition_all).to(device)
                train_z_bar = eval_decoder(train_z_bar, test_condition_bar).to(device)  # (temporal_dim, nodes_num)

                train_z_bar = train_z_bar.squeeze(0).squeeze(-1)
                x_forward = torch.sigmoid(torch.randn(train_z_bar.shape)).unsqueeze(0).to(device)

            x_forward.requires_grad = True
            x = x_true.cpu().detach().numpy()
            # initialization
            BN = nn.BatchNorm1d(1, affine=False).to(device)
            print("Getting initialization")
            initial_x, initial_x_prec = x_hat_initialization(i, eval_forward_model, x_forward, x, y_true, train_z_bar, train_z_all, BN,
                                                             threshold=thres, lr=args.inference_learning_rate, epochs=args.inference_epochs)

            with torch.no_grad():
                init_x = initial_x[initial_x_prec.index(max(initial_x_prec))]

            with torch.no_grad():
                init_x.clamp_(0, 1)

            x_pred = init_x.clone().cpu().detach().numpy()

            batch_x_true.append(x[0])
            batch_x_pred.append(init_x.clone().cpu().detach().numpy()[0])

            auc_float = roc_auc_score(x[0], x_pred[0], average='macro')
            x_pred[x_pred >= thres] = 1
            x_pred[x_pred != 1] = 0 #array

            if i==0:
                x_pred_all = [x_pred[0]]
            else:
                x_pred_all.append(x_pred[0])

            auc_binary = roc_auc_score(x[0], x_pred[0], average='macro')
            precision = precision_score(x[0], x_pred[0], average='macro')
            recall = recall_score(x[0], x_pred[0], average='macro')
            f1 = f1_score(x[0], x_pred[0], average='macro')
            accuracy = accuracy_score(x[0], x_pred[0])

            print("Test #{} ".format(i + 1),
                  "\tx Loss: {:.5f}".format(loss_seed_x(x_true, init_x, loss_type='bce').item()),
                  "\tPrec: {:.5f}".format(precision),
                  "\tRec: {:.5f}".format(recall),
                  "\tF1: {:.5f}".format(f1),
                  "\tAUC_float: {:.5f}".format(auc_float),
                  "\tAUC_binary: {:.5f}".format(auc_binary),
                  "\tACC {:.5f}".format(accuracy)
                  )

            precision_all += precision
            recall_all += recall
            f1_all += f1
            acc_all += accuracy

        print("This batch Test finished", "\tTotal Prec: {:.5f}".format(precision_all / eval.shape[0]),
              "\tTotal Rec: {:.5f}".format(recall_all / eval.shape[0]),
              "\tTotal F1: {:.5f}".format(f1_all / eval.shape[0]),
              "\tTotal Acc:{:.5f}".format(acc_all / eval.shape[0]))

        batch_precision_all += precision_all / eval.shape[0]
        batch_recall_all += recall_all / eval.shape[0]
        batch_f1_all += f1_all / eval.shape[0]
        batch_acc_all += acc_all / eval.shape[0]

    print("All batch Test finished", "\tTotal Prec: {:.5f}".format(batch_precision_all / eval_set.shape[0]),
          "\tTotal Rec: {:.5f}".format(batch_recall_all / eval_set.shape[0]),
          "\tTotal F1: {:.5f}".format(batch_f1_all / eval_set.shape[0]),
          "\tTotal Acc:{:.5f}".format(batch_acc_all / eval_set.shape[0]),)

    return batch_precision_all / eval_set.shape[0], batch_recall_all / eval_set.shape[0], batch_f1_all / eval_set.shape[0], batch_acc_all / eval_set.shape[0]

