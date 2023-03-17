import torch
import torch.nn as nn
import numpy as np


class NETWORK_F_MLP(nn.Module):
    def __init__(self, input_dim=784, HIDDEN=200, out_dim=200, how_many_layers=2):
        super(NETWORK_F_MLP, self).__init__()
        self.dim = out_dim
        self.many_layer = how_many_layers

        self.fc_list = []
        self.bn_list = []

        self.fc_list.append(nn.Linear(input_dim + 50, HIDDEN, bias=True))
        self.bn_list.append(nn.BatchNorm1d(HIDDEN))

        for i in range(0, self.many_layer - 1):
            self.fc_list.append(nn.Linear(HIDDEN, HIDDEN, bias=True))
            self.bn_list.append(nn.BatchNorm1d(HIDDEN))

        self.fc_list = nn.ModuleList(self.fc_list)
        self.bn_list = nn.ModuleList(self.bn_list)

        self.fc_final = nn.Linear(HIDDEN, out_dim, bias=True)

    def forward(self, x):
        same_noise = torch.zeros((x.shape[0], 50)).uniform_()
        x = torch.cat((x, same_noise), 1)

        for i in range(0, self.many_layer):
            x = self.fc_list[i](x)
            x = torch.relu(x)
            x = self.bn_list[i](x)

        x = torch.sigmoid((self.fc_final(x)))
        return x


class wf_layer_WIENERSOLUTION(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(wf_layer_WIENERSOLUTION, self).__init__()
        self.weights = torch.nn.Parameter(torch.zeros(out_dim, in_dim + 1))

    def forward(self, x):
        x_ = np.concatenate((np.ones((x.shape[0], 1)), x), 1)

        return self.weights @ x_.T

    def train_weights(self, x, desire_list):
        x_ = np.concatenate((np.ones((x.shape[0], 1)), x), 1)
        self.weights.data = desire_list.T @ x_ @ np.linalg.pinv(x_.T @ x_)
        return (x_.T @ x_), desire_list.T @ x_


def adaptive_estimation(v_t, beta, square_term, i):
    v_t = beta*v_t + (1-beta)*square_term.detach()
    return v_t, (v_t/(1-beta**i))


def produce_CC_GRAD_ALL_new(cat_vector, track_cov, i, threshold):
#     XY = cat_vector.T@cat_vector/cat_vector.shape[0] - cat_vector.mean(0).unsqueeze(1)@cat_vector.mean(0).unsqueeze(0)
    XY = cat_vector.T@cat_vector/cat_vector.shape[0]
    cov = XY + torch.eye((XY.shape[0]))*(threshold)
#     cov = XY

    track_cov, cov_estimate = adaptive_estimation(track_cov, 0.5, cov, i)
    return cov_estimate, cov, track_cov


class Model(torch.nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.in_channels = configs.enc_in
        self.threshold = 1e-6
        self.out_dim = 96
        self.f = NETWORK_F_MLP(input_dim=configs.seq_len, HIDDEN=96, out_dim=self.out_dim, how_many_layers=2)
        self.g = NETWORK_F_MLP(input_dim=configs.pred_len, HIDDEN=96, out_dim=self.out_dim, how_many_layers=2)
        self.layer = wf_layer_WIENERSOLUTION(self.out_dim, configs.pred_len)
        self.cov_estimate = None

    def get_loss(self, x, y, track_cov0=None, i=0):
        if track_cov0 is None:
            track_cov0 = torch.zeros((self.out_dim + self.out_dim, self.out_dim + self.out_dim)).to(x.device)
        output_f = self.f(x.transpose(1, 2).reshape(-1, self.seq_len))
        output_g = self.g(y.transpose(1, 2).reshape(-1, self.pred_len))

        cat_vector = torch.cat((output_f, output_g), 1)
        cov_estimate, cov, track_cov0 = produce_CC_GRAD_ALL_new(cat_vector, track_cov0, i, self.threshold)
        self.cov_estimate = cov_estimate
        cov_estimate_f = cov_estimate[:self.out_dim, :self.out_dim]
        cov_f = cov[:self.out_dim, :self.out_dim]

        cov_estimate_g = cov_estimate[self.out_dim:, self.out_dim:]
        cov_g = cov[self.out_dim:, self.out_dim:]

        loss = (torch.linalg.inv(cov_estimate) * cov).sum() - (torch.linalg.inv(cov_estimate_f) * cov_f).sum() - (
                    torch.linalg.inv(cov_estimate_g) * cov_g).sum()

        return loss, track_cov0

    def train_layer(self, loader):
        x = []
        y = []
        for (_x, _y, _, _) in loader:
            x.append(_x.float())
            y.append(_y.float())
        x = torch.cat(x, dim=0).transpose(1, 2).reshape(-1, self.seq_len)
        y = torch.cat(y, dim=0).transpose(1, 2).reshape(-1, self.pred_len)
        # todo get NaN values here
        eigen_normalized_f, eig_PF, ratio_map = self._calculate_general_ratio(x, self.cov_estimate)
        self.layer.train_weights(eigen_normalized_f, y)
#         predict = layer_f.forward(eigen_normalized_f)
# #         error = (desire-predict)**2
# #         print('error:', error.mean())

    def pred(self, x):
        o = []
        for i in range(x.shape[2]):
            eigen_normalized_f, eig_PF, ratio_map = self._calculate_general_ratio(x[:, :, i], self.cov_estimate)
            o.append(self.layer.forward(eigen_normalized_f).T)
        return torch.stack(o, dim=-1)

    def _PP_generate_quantities_gpu(self, cov_estimate):
        E1, V1 = torch.linalg.eigh(cov_estimate[:self.out_dim, :self.out_dim])
        E2, V2 = torch.linalg.eigh(cov_estimate[self.out_dim:, self.out_dim:])

        RF_NORM = V1 @ torch.diag(E1 ** (-1 / 2)) @ V1.T
        RG_NORM = V2 @ torch.diag(E2 ** (-1 / 2)) @ V2.T

        P = cov_estimate[:self.out_dim, self.out_dim:]
        P_STAR = RF_NORM @ P @ RG_NORM

        PP_F = P_STAR @ P_STAR.T
        eig_PF, eig_vec_PF = torch.linalg.eigh(PP_F)

        PP_G = P_STAR.T @ P_STAR
        eig_PG, eig_vec_PG = torch.linalg.eigh(PP_G)

        index_sort = eig_PF.argsort(descending=True)
        eig_PF = eig_PF[index_sort]
        eig_vec_PF = eig_vec_PF[:, index_sort]

        index_sort = eig_PG.argsort(descending=True)
        eig_PG = eig_PG[index_sort]
        eig_vec_PG = eig_vec_PG[:, index_sort]

        return RF_NORM, RG_NORM, P_STAR, eig_PF, eig_vec_PF, eig_PG, eig_vec_PG

    def _calculate_general_ratio(self, input_f_inter, cov_estimate):
        RF_NORM, RG_NORM, P_STAR, eig_PF, eig_vec_PF, PP_G, eig_vec_PG = self._PP_generate_quantities_gpu(cov_estimate)
        with torch.no_grad():
            BS = 100

            output_f_inter = torch.zeros((input_f_inter.shape[0], self.out_dim))
            for k in range(0, BS):
                output_f_inter = (self.f(input_f_inter) + output_f_inter * k) / (k + 1)

            normalized_f = output_f_inter @ RF_NORM
            eigen_normalized_f = normalized_f @ eig_vec_PF
            # eigen_normalized_f = eigen_normalized_f*torch.from_numpy(p_x.reshape(-1, 1)).cuda().float() # whether has density
            ratio_map = (eigen_normalized_f * eig_PF.reshape(1, -1)) @ eigen_normalized_f.T

        return eigen_normalized_f, eig_PF, ratio_map

    def calculate_general_ratio_withcov(self, input_f_inter, cov_estimate):
        RF_NORM, RG_NORM, P_STAR, eig_PF, eig_vec_PF, PP_G, eig_vec_PG = self._PP_generate_quantities_gpu(cov_estimate)
        with torch.no_grad():
            BS = 100

            output_f_inter = torch.zeros((input_f_inter.shape[0], self.seq_len))
            for k in range(0, BS):
                output_f_inter = (self.f(input_f_inter) + output_f_inter * k) / (k + 1)

            normalized_f = output_f_inter @ RF_NORM
            eigen_normalized_f = normalized_f @ eig_vec_PF
            # eigen_normalized_f = eigen_normalized_f*torch.from_numpy(p_x.reshape(-1, 1)).cuda().float() # whether has density
            ratio_map = (eigen_normalized_f * eig_PF.reshape(1, -1)) @ eigen_normalized_f.T

        return eigen_normalized_f, eig_PF, ratio_map

