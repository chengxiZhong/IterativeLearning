import torch
from math import exp
from torch.autograd import Variable
import torch.nn.functional as F
import math
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from Acousholo_auxiliaryfunc import normalize_amp

def type_trans(window,img):
    if img.is_cuda:
        window = window.cuda(img.get_device())
    return window.type_as(img)
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window
def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)
    # print(mu1.shape,mu2.shape)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2
    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12   = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2
    C1 = 0.01**2
    C2 = 0.03**2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    mcs_map  = (2.0 * sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)
    # print(ssim_map.shape)
    if size_average:
        return ssim_map.mean(), mcs_map.mean()
    # else:
    #     return ssim_map.mean(1).mean(1).mean(1)

def assess(target_img,propagated_pressure,propagated_pressure_nmlzd):
    def __M_accuracy(y_true, y_pred):
        numer = torch.sum(y_pred * y_true, dim=(1, 2, 3))  # numer.shape is torch.Size([BatchSize])
        denom = torch.sqrt(torch.sum(torch.pow(y_pred, 2), dim=(1, 2, 3)) * torch.sum(torch.pow(y_true, 2), dim=(1, 2, 3))) # numer.shape is torch.Size([BatchSize])
        return torch.mean((numer+0.001)/(denom+0.001))
        
    def __M_pc(y_true, y_pred):
        cov = torch.sum((y_true - torch.mean(y_true)) * (y_pred - torch.mean(y_pred)))
        denom = torch.sqrt(torch.sum(torch.square((y_true-torch.mean(y_true))))) * torch.sqrt(torch.sum(torch.square((y_pred-torch.mean(y_pred)))))
        return cov / denom

    def __M_mse(y_true, y_pred):        # y_pred needs normalization
        # y_pred = torch.divide(y_pred, torch.max(torch.max(y_pred, dim=-1)[0], dim=-1)[0].unsqueeze(-1).unsqueeze(-1))
        assert y_pred.max() <= 1.0 and y_pred.min() >= 0.0, "After normalization, the y_pred still is not in the range of [0,1]"
        return 1-torch.mean(torch.pow(y_pred-y_true, 2))

    def __M_SSIM(y_true, y_pred):        # y_pred needs normalization
        # y_pred = torch.divide(y_pred, torch.max(torch.max(y_pred, dim=-1)[0], dim=-1)[0].unsqueeze(-1).unsqueeze(-1))
        assert y_pred.max() <= 1.0 and y_pred.min() >= 0.0, "After normalization, the y_pred still is not in the range of [0,1]"
        _, channel, _, _ = y_true.size()
        window = create_window(window_size=11,channel=channel)
        window = type_trans(window,y_true)
        ssim_map, _ = _ssim(y_true, y_pred, window, window_size=11,channel=channel, size_average = True)
        return torch.mean(ssim_map)

    def __M_efficiency(y_true, y_pred):
        fg_mask = torch.ge(y_true, 0.5)
        on_target = fg_mask * y_pred
        zero_mask = torch.ne(on_target, 0)
        on_target_mean = torch.mean(torch.masked_select(on_target, zero_mask))
        # on_target_ratio = torch.sum(torch.masked_select(on_target, zero_mask)) / torch.sum(y_pred)
        on_target_ratio = torch.mean(torch.sum(y_pred*zero_mask, dim=(1, 2, 3)) / torch.sum(y_pred, dim=(1, 2, 3)))
        return on_target_mean, on_target_ratio

    def __M_uniformity(y_true, y_pred):
        fg_mask = torch.ge(y_true, 0.5)
        on_target = fg_mask * y_pred
        zero_mask = torch.ne(on_target, 0)
        on_target_std = torch.std(torch.masked_select(on_target, zero_mask))
        on_target_mean = torch.mean(torch.masked_select(on_target, zero_mask))
        return 1 - on_target_std / on_target_mean

    def __M_sd(y_true, y_pred): # Standard deviation  # y_pred needs normalization
        # y_pred = torch.divide(y_pred, torch.max(torch.max(y_pred, dim=-1)[0], dim=-1)[0].unsqueeze(-1).unsqueeze(-1))
        fg_mask = torch.ge(y_true, 0.5)
        on_target = fg_mask * y_pred
        zero_mask = torch.ne(on_target, 0)
        on_target_mean = torch.mean(torch.masked_select(on_target, zero_mask))
        nom = torch.sqrt(torch.mean(torch.masked_select(torch.square((on_target - on_target_mean)), zero_mask)))
        # return 100*(nom/on_target_mean)
        return (nom/on_target_mean)

    def __M_psnr(y_true, y_pred):                    # y_pred needs normalization
        # y_pred = torch.divide(y_pred, torch.max(torch.max(y_pred, dim=-1)[0], dim=-1)[0].unsqueeze(-1).unsqueeze(-1))
        assert y_pred.max() <= 1.0 and y_pred.min() >= 0.0, "After normalization, the y_pred still is not in the range of [0,1]"
        MSE = torch.mean(torch.pow(y_true - y_pred, 2))
        return 10*(torch.log10(1/MSE))
        # return 10*(torch.log(1/MSE)/torch.log(10))

    def __M_psnr_NoNormalization(y_true, y_pred):
        assert y_true.max() > 1.0 and y_pred.max() > 1.0, "(This is to check whether normalization is done) The normalization is done! "
        assert (torch.abs(torch.sum(y_true**2, dim=(-1,-2,-3))-2500) <= 1).all(), "The total energy of Ae_Input is not 2500"
        MAX = (torch.max(torch.max(torch.max(y_true, dim=-1)[0], dim=-1)[0], dim=-1)[0]).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        MSE = torch.mean(torch.pow(y_true - y_pred, 2))
        return 10*(torch.log10(MAX/MSE))

    # Accuracy (ACC)
    M_accuracy = __M_accuracy(target_img,propagated_pressure).cpu().numpy()

    # Pearson's correlation coefficient (PC)
    M_pc = __M_pc(target_img,propagated_pressure).cpu().numpy()

    # NMSE
    # M_mse = __M_mse(target_img,propagated_pressure).cpu().numpy()
    M_mse = __M_mse(target_img,propagated_pressure_nmlzd).cpu().numpy()

    # Structure similarity index metric (SSIM)
    # M_ssim = __M_SSIM(target_img,propagated_pressure).cpu().numpy()
    M_ssim = __M_SSIM(target_img,propagated_pressure_nmlzd).cpu().numpy()

    # Efficiency (EFF)
    M_efficiency_mean, M_efficiency_ratio = __M_efficiency(target_img,propagated_pressure)
    M_efficiency_mean, M_efficiency_ratio = M_efficiency_mean.cpu().numpy(), M_efficiency_ratio.cpu().numpy()
    
    # Uniformity
    M_uniformity = __M_uniformity(target_img,propagated_pressure).cpu().numpy()

    # Standard deviation
    # M_sd = __M_sd(target_img,propagated_pressure).cpu().numpy()
    M_sd = __M_sd(target_img,propagated_pressure_nmlzd).cpu().numpy()

    # Peak signal noise ratio (PSNR)
    # M_psnr = __M_psnr(target_img,propagated_pressure).cpu().numpy()
    M_psnr = __M_psnr(target_img,propagated_pressure_nmlzd).cpu().numpy()

    # print("Accuracy:",M_accuracy,"PC:",M_pc," MSE:", M_mse,", SSIM:", M_ssim,", Efficiency_bgmean:", M_efficiency_mean,", Efficiency_ratio:", M_efficiency_ratio,", Uniformity:", M_uniformity,", standard deviation:", M_sd,", PSNR:", M_psnr)
    return M_accuracy, M_pc, M_mse, M_ssim, M_efficiency_mean, M_efficiency_ratio, M_uniformity, M_sd, M_psnr
    

def visualize(target_img, retrieved_phase, propagated_pressure, propagated_phase, normalize_option, bs_index, current_epoch, save=True, save_path='./test_result_visualize.png'): # propagated_pressure also means reconstructed_img
    assert retrieved_phase.min()>=0 and retrieved_phase.max() <= 2*math.pi, "retrieved_phase is out of range of [0, 2pi]"
    retrieved_phase = retrieved_phase / (2*math.pi)
    assert propagated_phase.min()>=0 and propagated_phase.max() <= 2*math.pi, "propagated_phase is out of range of [0, 2pi]"
    propagated_phase = propagated_phase / (2*math.pi)
    
    # Save by subplots
    fig, ax = plt.subplots(nrows=1, ncols=6) # target_img | propagated_pressure_origin | propagated_pressure_nmlzd | diff | retrieved_phase | propagated_phase
    axs = ax.flatten()
    customized_colorbar = mcolors.LinearSegmentedColormap.from_list("mylist",["black","red"],N = 800)
    
    target = ax[0].imshow(target_img.cpu().numpy()[bs_index,0,:,:], cmap=customized_colorbar, vmin=0.0, vmax=1.0)# cmap = 'jet'
    img_count_ones = np.sum(target_img.cpu().numpy()[bs_index,0,:,:] == 1)
    img_count_zeros = np.sum(target_img.cpu().numpy()[bs_index,0,:,:] == 0)
    ax[0].set_title("target\n (1:{},0:{}) ".format(img_count_ones, img_count_zeros))
    # ======
    # propagated_pressure_nmlzd = torch.divide(propagated_pressure, torch.max(torch.max(propagated_pressure, dim=-1)[0], dim=-1)[0].unsqueeze(-1).unsqueeze(-1))
    propagated_pressure_nmlzd = normalize_amp(target_img=target_img, propagated_pressure=propagated_pressure, normalize_option=normalize_option)
    psnr = 10*(np.log10(1/np.mean(np.power(target_img.cpu().numpy()[bs_index,0,:,:] - propagated_pressure_nmlzd.cpu().numpy()[bs_index,0,:,:], 2))))
    print("Normalization method is {} (1: Ac max, 2: Ae max, 3: Energy)".format(normalize_option))
    # ======
    reconstructed_amp = ax[1].imshow(propagated_pressure.cpu().numpy()[bs_index,0,:,:], cmap=customized_colorbar, vmin=0.0, vmax=propagated_pressure.cpu().numpy()[bs_index,0,:,:].max())
    ax[1].set_title("recon")
    reconstructed_amp_nmlzd = ax[2].imshow(propagated_pressure_nmlzd.cpu().numpy()[bs_index,0,:,:], cmap=customized_colorbar, vmin=0.0, vmax=1.0)
    ax[2].set_title("psnr:{:.2f}".format(psnr))
    diff = ax[3].imshow(np.abs(target_img.cpu().numpy()[bs_index,0,:,:] - propagated_pressure_nmlzd.cpu().numpy()[bs_index,0,:,:]), cmap=customized_colorbar, vmin=0.0, vmax=1.0)
    ax[3].set_title("diff:{:.4f}".format((np.abs(target_img.cpu().numpy()[bs_index,0,:,:] - propagated_pressure_nmlzd.cpu().numpy()[bs_index,0,:,:])).mean()))
    phase = ax[4].imshow(retrieved_phase.cpu().numpy()[bs_index,0,:,:], cmap='jet', vmin=0.0, vmax=1.0) # cmap = cm.Spectral, interpolation='bicubic'
    ax[4].set_title("phase")
    reconstructed_phs = ax[5].imshow(propagated_phase.cpu().numpy()[bs_index,0,:,:], cmap='jet', vmin=0.0, vmax=1.0) # cmap = cm.Spectral, interpolation='bicubic'
    ax[5].set_title("reconstruct")
    [axi.set_axis_off() for axi in ax.ravel()] # To turn off axes for all subplots


    fig.colorbar(target, ax=ax[0], fraction=0.046, pad=0.04)           # pad=0.005 用于colorbar与image的距离, 
    fig.colorbar(reconstructed_amp, ax=ax[1], fraction=0.046, pad=0.04)  # fraction=0.046 用于使得colorbar的长度与image一致， 
    fig.colorbar(reconstructed_amp_nmlzd, ax=ax[2], fraction=0.046, pad=0.04)
    fig.colorbar(diff, ax=ax[3], fraction=0.046, pad=0.04)                # shrink=0.4 用于colorbar的缩放
    fig.colorbar(phase, ax=ax[4], fraction=0.046, pad=0.04)
    fig.colorbar(reconstructed_phs, ax=ax[5], fraction=0.046, pad=0.04)

    plt.tight_layout() # 让子图之间的文字不重叠
    if save==True:
        if not os.path.exists(save_path[:-4] + '/' + current_epoch):
            os.mkdir(save_path[:-4] + '/' + current_epoch)
        plt.savefig(save_path[:-4] + '/' + current_epoch + '/ReconPSNR_'+ str(np.round(psnr,2)) + save_path[-4:],dpi=300,bbox_inches='tight')
    plt.clf()
    
    # Save with single imshow plot
    Single_imshow_data = [target_img.cpu().numpy()[bs_index,0,:,:], propagated_pressure_nmlzd.cpu().numpy()[bs_index,0,:,:], np.abs(target_img.cpu().numpy()[bs_index,0,:,:] - propagated_pressure_nmlzd.cpu().numpy()[bs_index,0,:,:]), retrieved_phase.cpu().numpy()[bs_index,0,:,:], propagated_phase.cpu().numpy()[bs_index,0,:,:]]
    Single_imshow_name =['_1_targetAmp', '_2_reconAmp', '_3_diffAmp', '_4_retrPhs', '_5_reconPhs']
    Single_imshow_cmp = [customized_colorbar, customized_colorbar, customized_colorbar,'jet','jet']
    for Single_imshow_i in range(len(Single_imshow_data)):
        plt.imshow(Single_imshow_data[Single_imshow_i], cmap=Single_imshow_cmp[Single_imshow_i])
        plt.axis('off')
        plt.savefig(save_path[:-4] + '/' + current_epoch + '/ReconPSNR_'+ str(np.round(psnr,2)) + Single_imshow_name[Single_imshow_i] + save_path[-4:],dpi=300,bbox_inches='tight')
        plt.clf()


def save_results_trr(target_img, retrieved_phase, propagated_pressure, bs_index, save_path='./test_result_retrieved_phase.csv'):
    # ======
    propagated_pressure_nmlzd = torch.divide(propagated_pressure, torch.max(torch.max(propagated_pressure, dim=-1)[0], dim=-1)[0].unsqueeze(-1).unsqueeze(-1))
    psnr = 10*(np.log10(1/np.mean(np.power(target_img.cpu().numpy()[bs_index,0,:,:] - propagated_pressure_nmlzd.cpu().numpy()[bs_index,0,:,:], 2))))
    # ======
    target_img_numpy = target_img.cpu().numpy()[bs_index,0,:,:]
    retrieved_phase_numpy = retrieved_phase.cpu().numpy()[bs_index,0,:,:]
    propagated_pressure_numpy = propagated_pressure.cpu().numpy()[bs_index,0,:,:]
    target_img_list = target_img_numpy.flatten().tolist()
    retrieved_phase_list = retrieved_phase_numpy.flatten().tolist()
    propagated_pressure_list = propagated_pressure_numpy.flatten().tolist()
    dataframe = pd.DataFrame({'target holo':target_img_list, 'retrieved phs holo':retrieved_phase_list, 'reconstructed holo':propagated_pressure_list})
    propagated_pressure_max = np.max(propagated_pressure_numpy)
    customized_colorbar = mcolors.LinearSegmentedColormap.from_list("mylist",["black","red"],N = 800)
    plt.imsave(save_path[:-4] + '/ReconAmpMax_'+ str(np.round(propagated_pressure_max,2)) + '.jpeg', propagated_pressure_numpy, cmap=customized_colorbar)
    # np.savetxt(save_path, propagated_pressure_numpy,delimiter=",")
    dataframe.to_csv(save_path[:-4] + '/ReconPSNR_'+ str(np.round(psnr,2)) + save_path[-4:],index=False,sep=',')
    print("Results, including Ae, Ps, and Ac, was successfully saved as csv file")