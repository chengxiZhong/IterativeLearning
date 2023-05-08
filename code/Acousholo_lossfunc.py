import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from Acousholo_evaluation import create_window, type_trans, _ssim

def L_accuracy(y_true, y_pred):
    '''
    Lacc = 1 - ( sum(y_pred*y_true) / ( sqrt(sum(y_pred^2)) * sqrt(sum(y_true^2)) ) )
    1. y_pred does not need normalization, independent of amplitude
    2. cosine similariy-based loss
    Args:
        m, n is element size of PTA
        y_pred: tensor, size of (BatchSize, 1, m, n)
        y_true: tensor, size of (BatchSize, 1, m, n)
    Return:
        Lacc
    '''
    numer = torch.sum(y_pred * y_true, dim=(1, 2, 3))  # numer.shape is torch.Size([BatchSize])
    denom = torch.sqrt(torch.sum(torch.pow(y_pred, 2), dim=(1, 2, 3)) * torch.sum(torch.pow(y_true, 2), dim=(1, 2, 3))) # numer.shape is torch.Size([BatchSize])
    return 1-torch.mean((numer+0.001)/(denom+0.001))

def L_efficiency(y_true, y_pred):
    '''
    Leff = log ( (mean(Arecon))^-1 | Atarget=1 + (mean(Arecon)) | Atarget=0 )
    Args:
        m, n is element size of PTA
        y_pred: tensor, size of (BatchSize, 1, m, n), in the range of [0,1]
        y_true: tensor, size of (BatchSize, 1, m, n), in the range of [0,1]
        fg_mask: area whose pixel value is (Option1) greater or equal than 0.5, (Option2) not equal to 0, (Option3) equal to 1
        bg_mask: area whose pixel value is (Option1) less than 0.5
        on_target_mean: [mean] pixel value corresponding the "foreground" region of y_true
        off_target_mean: [mean] pixel value corresponding the "background" region of y_true
        on_target_std: [std] pixel value corresponding the "foreground" region of y_true
    Return:
        on_target_mean, off_target_mean, on_target_std
        or
        loge(1/on_target_mean + off_target_mean)
    '''
    # Check whether y_true's range is [0,1], and y_true.max, y_true.min
    assert y_true.max() <= 1.0 and y_true.min() >= 0.0, "y_true.max() > 1.0  or  y_true.min() < 0.0"
    # print(torch.sum((1.0 - y_true.max()) > 1e-2 ), torch.sum((y_true.min() - 0.0) > 1e-2))
    # exit()
    assert (1.0 - y_true.max()) <= 1e-2 and (y_true.min() - 0.0) <= 1e-2, "y_true.max() is far away from 1.0 or y_pred.min() is far waay from 0.0"
    # assert (1.0 - y_true.max()) <= 1e-3 and (y_true.min() - 0.0) <= 1e-3, "y_true.max() is far away from 1.0 or y_pred.min() is far waay from 0.0"
    assert y_pred.max() <= 1.0 and y_pred.min() >= 0.0, "y_pred.max() > 1.0  or  y_pred.min() < 0.0"
    # assert (1.0 - y_pred.max()) <= 1e-3 and (y_pred.min() - 0.0) <= 1e-3, "y_pred.max() is far away from 1.0 or y_pred.min() is far waay from 0.0"
    # The foreground area 
    fg_mask = torch.ge(y_true, 0.5) # Option1: The region whose value is greater or equal than 0.5
    on_target = fg_mask * y_pred
    zero_mask = torch.ne(on_target, 0)
    # on_target = y_true * y_pred # on_target.shape is torch.Size([BatchSize, 1, m, n])
    # fg_mask = torch.ne(on_target, 0.0) # Option2: The region whose value not equal to 0
    # fg_mask = torch.eq(on_target, 1.0) # Option3: The region whose value equal to 1
    # ????????????????????????????????????????????
    # This operation abbandon the background and pixels whose predicted value is zero in foreground.
    # Is it also can be (y_pred, zero_mask)?
    on_target_mean = torch.mean(torch.masked_select(on_target, zero_mask))  
    on_target_std = torch.std(torch.masked_select(on_target, zero_mask))

    bg_mask = torch.lt(y_true, 0.5) # Option1: The region whose value is less than 0.5
    off_target_mean = torch.mean(torch.masked_select(y_pred, bg_mask))
    # return on_target_mean, off_target_mean, on_target_std
    return torch.log(1/on_target_mean + off_target_mean) # natural logarithm, afterward, test torch.log10() conmmon logarithm; or torch.log2()

def L_uniformity(y_true, y_pred):  # 1-uniformity
    '''
    Luniformity = standard deviation of foreground of y_pred / mean value of foreground of y_pred
    '''
    # Check whether y_true's range is [0,1], and y_true.max, y_true.min
    assert y_true.max() <= 1.0 and y_true.min() >= 0.0, "y_true.max() > 1.0  or  y_true.min() < 0.0"
    # assert (1.0 - y_true.max()) <= 1e-3 and (y_true.min() - 0.0) <= 1e-3, "y_true.max() is far away from 1.0 or y_pred.min() is far waay from 0.0"
    assert (1.0 - y_true.max()) <= 1e-2 and (y_true.min() - 0.0) <= 1e-2, "y_true.max() is far away from 1.0 or y_pred.min() is far waay from 0.0"
    # assert y_pred.max() <= 1.0 and y_pred.min() >= 0.0, "y_pred.max() > 1.0  or  y_pred.min() < 0.0"
    # assert (1.0 - y_pred.max()) <= 1e-3 and (y_pred.min() - 0.0) <= 1e-3, "y_pred.max() is far away from 1.0 or y_pred.min() is far waay from 0.0"
    # The foreground area 
    fg_mask = torch.ge(y_true, 0.5) # Option1: The region whose value is greater or equal than 0.5
    on_target = fg_mask * y_pred
    zero_mask = torch.ne(on_target, 0)
    on_target_mean = torch.mean(torch.masked_select(on_target, zero_mask))
    on_target_std = torch.std(torch.masked_select(on_target, zero_mask))
    return on_target_std / on_target_mean

def L_npcc(y_true, y_pred):       # Negative pearson correlation coefficient (NPCC)
    cov = torch.sum((y_true - torch.mean(y_true)) * (y_pred - torch.mean(y_pred)))
    denom = torch.sqrt(torch.sum(torch.square((y_true-torch.mean(y_true))))) * torch.sqrt(torch.sum(torch.square((y_pred-torch.mean(y_pred)))))
    return (-1) * (cov / denom)

def L_percp(y_true, y_pred):   # percp means the perceptual reconstruction space, which inspiring me to try spactial transform !!!!!
    # y_pred = torch.divide(y_pred, torch.max(torch.max(y_pred, dim=-1)[0], dim=-1)[0].unsqueeze(-1).unsqueeze(-1))
    # assert y_pred.max() <= 1.0 and y_pred.min() >= 0.0, "After normalization, the y_pred still is not in the range of [0,1]"
    # return torch.mean(torch.sum(torch.square(y_true - y_pred), dim=(1,2,3))) # L2 norm square
    return torch.mean(torch.sum(torch.square(y_true - y_pred), dim=(1,2,3))/(y_true.size()[-1]*y_true.size()[-2])) # mean square error

def L_percp_freq(y_true, y_pred):
    return torch.mean(torch.sum(torch.square(torch.abs(y_true) - torch.abs(y_pred)), dim=(1,2,3))/(y_true.size()[-1]*y_true.size()[-2]))

def L_percp_fg(y_true, y_true_scaleup, y_pred):
    fg_mask = torch.ge(y_true, 0.5) # 128/255 = 0.502
    fg_mask_count = torch.sum(fg_mask, dim=(1,2,3))
    on_target_true = fg_mask * y_true_scaleup
    on_target_pred = fg_mask * y_pred
    return torch.mean(torch.sum(torch.square(on_target_true - on_target_pred), dim=(1,2,3))/fg_mask_count)

def L_psnr(y_true, y_pred):
    # y_pred = torch.divide(y_pred, torch.max(torch.max(y_pred, dim=-1)[0], dim=-1)[0].unsqueeze(-1).unsqueeze(-1))
    assert y_pred.max() <= 1.0 and y_pred.min() >= 0.0, "After normalization, the y_pred still is not in the range of [0,1]"
    MSE = torch.mean(torch.pow(y_true - y_pred, 2))
    return -10*(torch.log10(1/MSE))

def L_psnr_NoNormalization(y_true, y_pred):
    assert y_true.max() > 1.0 and y_pred.max() > 1.0, "(This is to check whether normalization is done) The normalization is done! "
    assert (torch.abs(torch.sum(y_true**2, dim=(-1,-2,-3))-2500) <= 1).all(), "The total energy of Ae_Input is not 2500"
    MAX = (torch.max(torch.max(torch.max(y_true, dim=-1)[0], dim=-1)[0], dim=-1)[0]).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    MSE = torch.pow(y_true - y_pred, 2)
    return  torch.mean(-10*(torch.log10(MAX/MSE)))

def L_ssim(y_true, y_pred):
    # y_pred = torch.divide(y_pred, torch.max(torch.max(y_pred, dim=-1)[0], dim=-1)[0].unsqueeze(-1).unsqueeze(-1))
    assert y_pred.max() <= 1.0 and y_pred.min() >= 0.0, "After normalization, the y_pred still is not in the range of [0,1]"
    _, channel, _, _ = y_true.size()
    window = create_window(window_size=11,channel=channel)
    window = type_trans(window,y_true)
    ssim_map, _ = _ssim(y_true, y_pred, window, window_size=11,channel=channel, size_average = True)
    return 1-torch.mean(ssim_map)

def L_SumSquareMAE(y_true, y_pred):
    assert y_pred.max() <= 1.0 and y_pred.min() >= 0.0, "After normalization, the y_pred still is not in the range of [0,1]"
    SSMAE = torch.sum(torch.pow(torch.abs(y_true - y_pred),2))
    return SSMAE


def L_cos(y_true, y_pred):
    assert y_true.min() >= 0.0 and y_true.max() <= 1.0, "y_true is not in the range of [0,1]"
    assert y_pred.min() >= 0.0 and y_pred.max() <= 1.0, "y_pred is not in the range of [0,1]"
    y_true, y_pred = y_true * (2*torch.pi), y_pred * (2*torch.pi)
    # assert y_true.min() >= 0.0 and y_true.max() <= (2*torch.pi), "y_true is not in the range of [0,(2pi)]"
    # assert y_pred.min() >= 0.0 and y_pred.max() <= (2*torch.pi), "y_pred is not in the range of [0,(2pi)]"
    Diff_cos = torch.cos(torch.abs(y_true - y_pred))
    Imatrix = torch.ones_like(Diff_cos)
    loss_cos_mae = Imatrix - Diff_cos
    return torch.mean(loss_cos_mae)

def L_pwl(y_true, y_pred):
    assert y_true.min() >= 0.0 and y_true.max() <= 1.0, "y_true is not in the range of [0,1]"
    assert y_pred.min() >= 0.0 and y_pred.max() <= 1.0, "y_pred is not in the range of [0,1]"
    y_true, y_pred = y_true * (2*torch.pi), y_pred * (2*torch.pi)
    # assert y_true.min() >= 0.0 and y_true.max() <= (2*torch.pi), "y_true is not in the range of [0,(2pi)]"
    # assert y_pred.min() >= 0.0 and y_pred.max() <= (2*torch.pi), "y_pred is not in the range of [0,(2pi)]"
    Diff = torch.abs(y_true - y_pred)
    Diff_gt_pi = torch.gt(Diff, torch.pi)
    Diff_le_pi = torch.le(Diff, torch.pi)
    Diff_pwl = Diff_gt_pi * (2*torch.pi-Diff) + Diff_le_pi * (Diff)
    return torch.mean(Diff_pwl)


class Lacc(nn.Module):
    def __init__(self):
        super(Lacc, self).__init__()
    def forward(self, y_true, y_pred):
        return L_accuracy(y_true, y_pred)

class Leff(nn.Module):
    def __init__(self):
        super(Leff, self).__init__()
    def forward(self, y_true, y_pred):
        return L_efficiency(y_true, y_pred)

class Lunifo(nn.Module):
    def __init__(self):
        super(Lunifo, self).__init__()
    def forward(self, y_true, y_pred):
        return L_uniformity(y_true, y_pred)

class Lnpcc(nn.Module):
    def __init__(self):
        super(Lnpcc, self).__init__()
    def forward(self, y_true, y_pred):
        return L_npcc(y_true, y_pred)

class Lpercp(nn.Module):
    def __init__(self):
        super(Lpercp, self).__init__()
    def forward(self, y_true, y_pred):
        return L_percp(y_true, y_pred)

class Lpercp_freq(nn.Module):
    def __init__(self):
        super(Lpercp_freq, self).__init__()
    def forward(self, y_true, y_pred):
        return L_percp_freq(y_true, y_pred)
    
class Lpercpfg(nn.Module):
    def __init__(self):
        super(Lpercpfg, self).__init__()
    def forward(self, y_true, y_true_scaleup, y_pred):
        return L_percp_fg(y_true, y_true_scaleup, y_pred)

class Lpsnr(nn.Module):
    def __init__(self):
        super(Lpsnr, self).__init__()
    def forward(self, y_true, y_pred):
        return L_psnr(y_true, y_pred)

class Lpsnr_NoNormalization(nn.Module):
    def __init__(self):
        super(Lpsnr_NoNormalization, self).__init__()
    def forward(self, y_true, y_pred):
        return L_psnr_NoNormalization(y_true, y_pred)
    

class Lssim(nn.Module):
    def __init__(self):
        super(Lssim, self).__init__()
    def forward(self, y_true, y_pred):
        return L_ssim(y_true, y_pred)

class LSumSquareMAE(nn.Module):
    def __init__(self):
        super(LSumSquareMAE, self).__init__()
    def forward(self, y_true, y_pred):
        return L_SumSquareMAE(y_true, y_pred)

class GradLayer(nn.Module):

    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def get_gray(self,x):
        ''' 
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):
        if x.shape[1] == 3:
            x = self.get_gray(x)
        x_v = F.conv2d(x, self.weight_v.to(x.device), padding=1)
        x_h = F.conv2d(x, self.weight_h.to(x.device), padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)
        return x

class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.grad_layer = GradLayer()

    def forward(self, y_true, y_pred):
        output_grad = self.grad_layer(y_true)
        gt_grad = self.grad_layer(y_pred)
        return self.loss(output_grad, gt_grad)
    
class GradWeightedLoss(nn.Module):
    def __init__(self):
        super(GradWeightedLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.grad_layer = GradLayer()

    def forward(self, y_true, y_pred):
        grad = self.grad_layer(torch.abs(y_true-y_pred))
        return self.loss(grad*y_true, grad*y_pred)
    
class Lcos(nn.Module):
    def __init__(self):
        super(Lcos, self).__init__()
    def forward(self, y_true, y_pred):
        return L_cos(y_true, y_pred)

class Lpwl(nn.Module):
    def __init__(self):
        super(Lpwl, self).__init__()
    def forward(self, y_true, y_pred):
        return L_pwl(y_true, y_pred)
