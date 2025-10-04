
import torch.nn.functional as F
import torch

def loss_seed_x(x, x_hat, loss_type='mse'):
    if loss_type == 'bce':
        return F.binary_cross_entropy(x_hat, x, reduction='mean')
    else:
        return F.mse_loss(x_hat, x)

def loss_inverse_initial (y_true, y_hat, x_hat, f_z_bar, f_z_all, BN):
    device = x_hat.device
    forward_loss = F.mse_loss(y_hat, y_true)

    pdf_sum = 0
    for i, x_i in enumerate(x_hat[0]):
        temp = torch.pow(f_z_bar[i], x_i) * torch.pow(1 - f_z_bar[i], 1 - x_i).to(torch.double)
        pdf_sum += torch.log(1 + temp)

    log_pmf = []
    for f_z in f_z_all:
        log_likelihood_sum = torch.zeros(1).to(device)
        for i, x_i in enumerate(x_hat[0]):
            temp = torch.pow(f_z[i], x_i) * torch.pow(1 - f_z[i], 1 - x_i).to(torch.double)
            log_likelihood_sum += torch.log(1 + temp)
        log_pmf.append(log_likelihood_sum)
    log_pmf = torch.stack(log_pmf)
    log_pmf = BN(log_pmf.float())

    pmf_max = torch.max(log_pmf)
    pdf_sum_all = pmf_max + torch.log(torch.sum(torch.exp(log_pmf - pmf_max + 1), dim=0))

    return forward_loss - pdf_sum - pdf_sum_all, pdf_sum, pdf_sum_all

def loss_all( x_true, x_hat, log_var, mean, y_hat, y):
    forward_loss = F.mse_loss(y_hat, y, reduction='sum')
    monotone_loss = torch.sum(torch.relu(y_hat - y_hat[0]))
    reproduction_loss = F.binary_cross_entropy(x_hat, x_true, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    total_loss = reproduction_loss + KLD + forward_loss + monotone_loss
    return forward_loss, reproduction_loss, KLD, total_loss