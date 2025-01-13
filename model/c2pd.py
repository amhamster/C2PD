# -*- coding: utf-8 -*-

import torch
from torch import nn
import segmentation_models_pytorch as smp

device = 'cuda'


# This is the horizontal operation in PCGD,
# and the vertical part can be realized by matrix transposition.
class H_Branch(nn.Module):  # Horizontal branch of PCGD
    def __init__(self):
        super(H_Branch, self).__init__()
        self.CAPO = CAPO()

    def forward(self, I, G_feat, train=False):
        B, C, H, W = I.size()  # lr.size()

        dI = I[:, :, :, 1:] - I[:, :, :, :-1]
        dG = torch.unsqueeze(torch.mean(torch.abs(G_feat[:, :, :, 1:] - G_feat[:, :, :, :-1]), 1), 1)

        p_mask = (dI > 0).float()
        p_dI = dI * p_mask
        n_dI = dI * (1 - p_mask)

        # To improve generalization, positive and negative gradients share the same parameters.
        p_dI_ot = self.CAPO(p_dI, dG)
        n_dI_ot = self.CAPO(n_dI, dG)

        d_ot = I.clone()
        d_ot[:, :, :, 1:] = p_dI_ot + n_dI_ot

        # Right multiply the upper triangular matrix to achieve horizontal accumulation
        upper_triangular_matrix = torch.triu(torch.ones((W, W)))
        upper_triangular_matrix = upper_triangular_matrix.to(device)
        ot = torch.matmul(d_ot, upper_triangular_matrix.unsqueeze(0).unsqueeze(0))

        return ot


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CAPO(nn.Module):  # CAPO(4x1)
    def __init__(self):
        super(CAPO, self).__init__()
        self.mlp = Mlp(in_features=8, hidden_features=32, out_features=12)

    def forward(self, target, guidance):
        n = 4 * 1
        b, c, h, w = target.size()  # b,1,h,w

        operated = target.clone()  # copy a target to be operated
        con = torch.cat(
            [target[:, :, :, :-3], target[:, :, :, 1:-2],
             target[:, :, :, 2:-1], target[:, :, :, 3:],
             guidance[:, :, :, :-3], guidance[:, :, :, 1:-2],
             guidance[:, :, :, 2:-1], guidance[:, :, :, 3:]], 1)
        local_input = con.permute(0, 2, 3, 1).contiguous().view(b, h * (w - 3), 8)

        local_output = self.mlp(local_input)
        ari_1 = local_output[:, :, :4].contiguous().view(b, h * (w - 3), 2, 2)  # area relative information 1
        ari_2 = local_output[:, :, 4:8].contiguous().view(b, h * (w - 3), 2, 2)  # area relative information 2
        var = local_output[:, :, 8:]  # degree of variation
        ara = torch.matmul(ari_1, ari_2.transpose(-2, -1)).contiguous().view(b, h * (w - 3), 4)
        ara = nn.Softmax(dim=-1)(ara)  # area relative attention
        var = ara * var
        var = var.permute(2, 0, 1).contiguous().view(4, b, h, w - 3)

        sum = var[0] + var[1] + var[2] + var[3]
        var_1_norm = (var[0] - sum / n)
        var_2_norm = (var[1] - sum / n)
        var_3_norm = (var[2] - sum / n)
        var_4_norm = (var[3] - sum / n)

        operated[:, :, :, :-3] += var_1_norm.unsqueeze(1) / n
        operated[:, :, :, 1:-2] += var_2_norm.unsqueeze(1) / n
        operated[:, :, :, 2:-1] += var_3_norm.unsqueeze(1) / n
        operated[:, :, :, 3:] += var_4_norm.unsqueeze(1) / n

        return operated


class CAPO_3x3(nn.Module):  # Same as CAPO(4x1)
    def __init__(self):
        super(CAPO_3x3, self).__init__()
        self.mlp = Mlp(in_features=18, hidden_features=64, out_features=27)

    def forward(self, target, guidance):
        n = 3 * 3
        operated = target.clone()

        b, c, h, w = target.size()

        con = torch.cat([target[:, :, :-2, :-2].clone(), target[:, :, :-2, 1:-1].clone(), target[:, :, :-2, 2:].clone(),
                         target[:, :, 1:-1, :-2].clone(), target[:, :, 1:-1, 1:-1].clone(),
                         target[:, :, 1:-1, 2:].clone(),
                         target[:, :, 2:, :-2].clone(), target[:, :, 2:, 1:-1].clone(), target[:, :, 2:, 2:].clone(),
                         guidance[:, :, :-2, :-2].clone(), guidance[:, :, :-2, 1:-1].clone(),
                         guidance[:, :, :-2, 2:].clone(),
                         guidance[:, :, 1:-1, :-2].clone(), guidance[:, :, 1:-1, 1:-1].clone(),
                         guidance[:, :, 1:-1, 2:].clone(),
                         guidance[:, :, 2:, :-2].clone(), guidance[:, :, 2:, 1:-1].clone(),
                         guidance[:, :, 2:, 2:].clone(),
                         ], 1)  # b,18*c,h-2,w-2
        local_input = con.permute(0, 2, 3, 1).contiguous().view(b, (h - 2) * (w - 2), 18)

        local_output = self.mlp(local_input)

        ari_1 = local_output[:, :, :9].contiguous().view(b, (h - 2) * (w - 2), 3, 3)
        ari_2 = local_output[:, :, 9:18].contiguous().view(b, (h - 2) * (w - 2), 3, 3)
        var = local_output[:, :, 18:]

        ara = torch.matmul(ari_1, ari_2.transpose(-2, -1)).contiguous().view(b, (h - 2) * (w - 2), 9)
        ara = nn.Softmax(dim=-1)(ara)
        var = ara * var
        var = var.permute(2, 0, 1).contiguous().view(9, b, h - 2, w - 2)

        sum = var[0] + var[1] + var[2] + \
              var[3] + var[4] + var[5] + \
              var[6] + var[7] + var[8]
        var_1_norm = (var[0] - sum / n)
        var_2_norm = (var[1] - sum / n)
        var_3_norm = (var[2] - sum / n)
        var_4_norm = (var[3] - sum / n)
        var_5_norm = (var[4] - sum / n)
        var_6_norm = (var[5] - sum / n)
        var_7_norm = (var[6] - sum / n)
        var_8_norm = (var[7] - sum / n)
        var_9_norm = (var[8] - sum / n)

        operated[:, :, :-2, :-2] += var_1_norm.unsqueeze(1) / n
        operated[:, :, :-2, 1:-1] += var_2_norm.unsqueeze(1) / n
        operated[:, :, :-2, 2:] += var_3_norm.unsqueeze(1) / n
        operated[:, :, 1:-1, :-2] += var_4_norm.unsqueeze(1) / n
        operated[:, :, 1:-1, 1:-1] += var_5_norm.unsqueeze(1) / n
        operated[:, :, 1:-1, 2:] += var_6_norm.unsqueeze(1) / n
        operated[:, :, 2:, :-2] += var_7_norm.unsqueeze(1) / n
        operated[:, :, 2:, 1:-1] += var_8_norm.unsqueeze(1) / n
        operated[:, :, 2:, 2:] += var_9_norm.unsqueeze(1) / n

        return operated


class C2PD(nn.Module):
    def __init__(self):
        super(C2PD, self).__init__()
        self.Extractor1 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bicubic'),
            smp.Unet('resnet50', classes=32, in_channels=4),
            torch.nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.Extractor2 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bicubic'),
            smp.Unet('resnet50', classes=32, in_channels=4),
            torch.nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.modeling = CAPO_3x3()
        self.deform = H_Branch()

    def forward(self, rgb, lr_up, train=False):
        # ------------------ Isovolumetric Deformation ------------------
        feature = self.Extractor1(torch.cat((rgb, lr_up - lr_up.mean((1, 2, 3), keepdim=True)), 1))
        operational_map = torch.unsqueeze(torch.mean(feature, 1), 1)
        lr_up_pre = self.modeling(lr_up, operational_map)

        # ---------- Pixelwise Cross Gradient Deformance (PCGD) ----------
        # Building the horizontal branch of PCGD
        feature_h = self.Extractor2(torch.cat((rgb, lr_up_pre - lr_up_pre.mean((1, 2, 3), keepdim=True)), 1))
        h_comp = self.deform(lr_up_pre, feature_h)

        # Construct a vertical branch of PCGD.
        # To improve generalization, the two branches share same parameters.
        rgb_transpose = rgb.transpose(3, 2)
        lr_up_transpose = lr_up_pre.transpose(3, 2)
        feature_v = self.Extractor2(
            torch.cat((rgb_transpose, lr_up_transpose - lr_up_transpose.mean((1, 2, 3), keepdim=True)), 1))
        v_comp = self.deform(lr_up_transpose, feature_v)
        v_comp = v_comp.transpose(3, 2)

        out = (h_comp + v_comp) / 2

        return out
