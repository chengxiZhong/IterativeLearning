import torch
import torch.optim as optim
import torchvision
from torchvision import transforms as T
from PIL import Image
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from UNet_V3_original import UNet_V3, init_weights
# from UNet_V3_original_forTJ import UNet_V3, init_weights
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd 
import matplotlib.colors as mcolors

def lr_update(lr_para, epoch_i): 
    '''
    This function is used for manual update of learning rate during training process
    para:
        lr_para: this is a dictionary maintaining the parameters related to learning rate 
                lr_para['lr'] indicates the learning rate value
                lr_para['decrease_at'] is a list indicating the learning rate decreasing epochs
                lr_para['decrease_method'] indicates the learning rate update methods, 
                                           including divide with 2 and 5 in turns, divide with 10
        epoch_i: training epoch when the learning rate update is required
    return:
        lr: updated learning rate
    '''
    lr = lr_para['lr']
    print("----------------------------------------- learning rate decreasing -----------------------------------------")
    print("Now, we are reaching the {}th epoch. It's time to decrease learning rate with method of {}! ".format(epoch_i, lr_para['decrease_method']))
    if lr_para['decrease_method'] == '2_5inturns':
        index_for_lr = lr_para['decrease_at'].index(epoch_i)
        lr = lr / (2*((index_for_lr+1)%2) + 5*(index_for_lr%2))
    elif lr_para['decrease_method'] == '10':
        lr = lr / int(lr_para['decrease_method'])
    else:
        print("The learning rate decrease using a wrong method! ")
        exit()
    print("After decrease at {}th epoch, current learning rate is {}".format(epoch_i, lr))
    print("The learning rate decrease technique has been done! ")
    print("------------------------------------------------------------------------------------------------------------")
    return lr

def net_freeze(train_para, netfreeze_para, lr_para, epoch_i):
    '''
    This function is used to freeze network for fine tuning in iterative learning with experience pools
    para:
        train_para is a dictionary maintaining training parameters such as batch size, epochs, validation time slot, etc.
            train_para['net']: current network
        netfreeze_para is a dictionary maintaining parameters including 
                        'freeze' (whether freeze net), 
                        'freeze_times' (how many times to freeze net), 
                        and 'freeze_dict' (what parts of net require frozen)
        lr_para: learning rate parameters
        epoch_i: freeze net at epoch_i
    return:
        optimizer and net after freezing
    '''
    net = train_para['net']
    print("----------------------------------------- network layer freezing -----------------------------------------")
    print("Now, we are reaching the {}th epoch. It's time to freeze network! ".format(epoch_i))
    net_freeze_list = netfreeze_para['freeze_dict'][str(epoch_i)]
    for layer in net_freeze_list:
        for _, value in layer.named_parameters():
            value.requires_grad = False
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr_para['lr'])
    print("The neural network freeze technique has been done! ")
    print("----------------------------------------------------------------------------------------------------------")
    return optimizer, net


def ASM(d=30e-3, PhsHolo=torch.zeros((1,1,50,50)), AmpHolo=torch.zeros((1,1,50,50)), Lam=6.4e-4, fs=1 / (50e-3 / 50), BatchSize=16):
    '''
    ASM (Angular Spectrum Method) is a model fomulating wave propagation between two holpgram plane
    d is positive means propagate from source hologram to target hologram
    d is negative means propagate from target hologram to source hologram
    Args:
        d: its signal determines whether the ASM or Inverse ASM is applied
           its absolute value determines the propagation distance
        PhsHolo: phase hologram
        AmpHolo: amplitude hologram
        Lam is wavelength and fs is sample frequency
        m and n are the number of meta-cells on  x and y axis of PZT
    return:
        Amp: > 0
        Phs: [-pi, pi]
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert Lam > 0, "Wavelength < 0"
    assert abs(d) >= 10e-3 and abs(d) <= 40e-3, "the d is out of range"
    
    Holo = torch.cat([PhsHolo, AmpHolo], dim=1)
    assert Holo.shape == torch.Size([BatchSize, 2, 50, 50]), "Holo.shape != (BS, 2, 50, 50)"
    m, n = Holo.shape[-2], Holo.shape[-1]
    assert m == 50 and n == 50, "The width and/or height of Holo is wrong"

    Phs = PhsHolo.squeeze(1)  # [BatchSize, 50, 50]
    Amp = AmpHolo.squeeze(1)  # [BatchSize, 50, 50]
    assert Phs.shape == torch.Size([BatchSize, 50, 50]), "phs.shape != torch.Size([BatchSize, 1, 50, 50])"
    assert Amp.shape == torch.Size([BatchSize, 50, 50]), "phs.shape != torch.Size([BatchSize, 1, 50, 50])"

    Re = Amp * torch.cos(Phs) # [BatchSize, 50, 50]
    Im = Amp * torch.sin(Phs) # [BatchSize, 50, 50]
    Complex = (Re + 1j * Im).to(device)

    # FFT
    Complex_freqdomian = torch.fft.fftshift(torch.fft.fftn(Complex))
    # Propagator
    [Freq_x, Freq_y] = torch.meshgrid((torch.arange(m)-m/2) * (fs/m), (torch.arange(n)-n/2) * (fs/n))

    # ---------------------- Original version (no consideration of evanescent waves) ---------------------- #
    assert torch.all(1 / (Lam**2) - Freq_x**2 - Freq_y**2 >= 0) == True, "[1 / (Lam**2) - Freq_x**2 - Freq_y**2 < 0] in ASM"
    w_of_Freqx_Freqy = torch.sqrt(1 / (Lam**2) - Freq_x**2 - Freq_y**2)
    Propagator = torch.zeros((m, n), dtype=torch.complex128)
    Propagator = torch.exp(1j * 2 * np.pi * w_of_Freqx_Freqy * d).to(device)
    # ----------------------------------------------------------------------------------------------------- #

    # # ------------------------------- Change 1j to -1j in Original version -------------------------------- #
    # # assert torch.all(1 / (Lam**2) - Freq_x**2 - Freq_y**2 >= 0) == True, "[1 / (Lam**2) - Freq_x**2 - Freq_y**2 < 0] in ASM"
    # if torch.all(1 / (Lam**2) - Freq_x**2 - Freq_y**2 >= 0):
    #     w_of_Freqx_Freqy = torch.sqrt(1 / (Lam**2) - Freq_x**2 - Freq_y**2)
    #     Propagator = torch.zeros((m, n), dtype=torch.complex128)
    #     Propagator = torch.exp(1j * 2 * np.pi * w_of_Freqx_Freqy * d).to(device)
    # else:
    #     # Deal with evanescent waves
    #     Propagator = torch.zeros((m, n), dtype=torch.complex128).to(device)
    #     for i in range(m):
    #         for j in range(n):
    #             fz2 = 1 / (Lam ** 2) - Freq_x[(i, j)] ** 2 - Freq_y[(i, j)] ** 2
    #             if fz2 >= 0:
    #                 Propagator[(i, j)] = torch.exp(1j * 2 * torch.pi * d * torch.sqrt(fz2)).to(device)
    #             else:
    #                 Propagator[(i, j)] = torch.exp(-2 * torch.pi * d * torch.sqrt(-fz2)).to(device)
    #     Propagator = Propagator.to(device=device, dtype=torch.float)
    # # ----------------------------------------------------------------------------------------------------- #

    # # ------------------------------------ Deal with evanescent waves ------------------------------------- #
    # Propagator = torch.zeros((m, n), dtype=torch.complex128)
    # mask_evanescent_waves = torch.lt(1 / (Lam**2) - Freq_x**2 - Freq_y**2, 0.0)
    # Propagator = torch.exp(-1j * 2 * np.pi * d * torch.sqrt((1^mask_evanescent_waves) * (1 / (Lam**2) - Freq_x**2 - Freq_y**2)) + -0j * 2 * np.pi * d * torch.sqrt(mask_evanescent_waves * (Freq_x**2 + Freq_y**2 - 1 / (Lam**2)))).to(device)
    # # ----------------------------------------------------------------------------------------------------- #


    # Transform to another hologram plane
    Complex_freqdomian2 = Complex_freqdomian * Propagator
    # IFFT
    Complex2 = torch.fft.ifftn(torch.fft.ifftshift(Complex_freqdomian2))

    Amp2 = torch.abs(Complex2)
    Phs2 = torch.angle(Complex2) # output range is [-pi, pi]

    # Phs2 = Phs2 - torch.floor(Phs2/(2*torch.pi)) * (2*torch.pi) # [0, 2pi]
    # assert torch.all(torch.abs(Phs2) <= 2 * math.pi) and torch.all(Phs2 >= 0.0), "Phs2 is out of range [0, 2pi]"
    # assert torch.all(torch.abs(Phs2) <= math.pi), "Phs2 is out of range [-pi, pi]"

    Amp2 = Amp2.unsqueeze(1)         # [BatchSize, 1, 50, 50]
    Phs2 = Phs2.unsqueeze(1)         # [BatchSize, 1, 50, 50]
    assert Amp2.shape == torch.Size([BatchSize, 1, 50, 50]), "Amp2's shape wrong"
    assert Phs2.shape == torch.Size([BatchSize, 1, 50, 50]), "Phs2's shape wrong"
    assert ((torch.sum(Amp2**2, dim=(-1, -2))-2500) < 1e-2).all(), "The reconstructed amplitude hologram's total energy is not 2500"

    return Amp2, Phs2

def central_axis_plot(list=[], label='', xlabel='', ylabel='',save_path=''):
    plt.plot(range(1,len(list)+1,1), list, c='b', marker='v', ms=4, label=label)
    plt.xticks(range(1, len(list)+1, 1), rotation = 90)
    plt.xlim(0, len(list)+1)
    plt.ylim(min(list)-0.2, max(list)+0.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()

def Propagation_Calculate(Algorithm_name = 'IB',
                            Imaging_plane_z = 23e-3, 
                            PAT_para = [50e-3, 50e-3, 50, 50],
                            Imaging_plane_para = [50e-3, 50e-3, 100, 100],
                            PhsHolo=torch.zeros((1,1,50,50)), 
                            AmpHolo=torch.zeros((1,1,50,50)), 
                            Lam = 6.4e-4,
                            L = 0.8e-3,
                            d_33 = 374e-12,
                            ρ = 1e3,
                            f = 2.32e6,
                            S = (0.8e-3)**2,
                            Vptp = 20,
                            BatchSize=16,
                            plot_results=True):
    # This function is used to simulate the acoustic field propagation
    # The used algorithm can be 'IB1', 'IB2', 'ASM1'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Coordinate of transducers on PAT
    # PAT_length_x, PAT_length_y, PAT_pixel_count_x, PAT_pixel_count_y = 50e-3, 50e-3, 50, 50
    PAT_length_x, PAT_length_y, PAT_pixel_count_x, PAT_pixel_count_y = PAT_para[0], PAT_para[1], PAT_para[2], PAT_para[3]
    PAT_pixel_size_x, PAT_pixel_size_y = PAT_length_x/PAT_pixel_count_x, PAT_length_y/PAT_pixel_count_y
    PAT_x, PAT_y = torch.meshgrid(torch.linspace(-PAT_length_x/2+PAT_pixel_size_x/2, PAT_length_x/2-PAT_pixel_size_x/2, PAT_pixel_count_x), torch.linspace(-PAT_length_y/2+PAT_pixel_size_y/2, PAT_length_y/2-PAT_pixel_size_y/2, PAT_pixel_count_y))
    
    # Coordinate of pixels on imaging plane 
    # Imaging_plane_length_x, Imaging_plane_length_y, Imaging_plane_pixel_count_x, Imaging_plane_pixel_count_y = 50e-3, 50e-3, 100, 100
    Imaging_plane_length_x, Imaging_plane_length_y, Imaging_plane_pixel_count_x, Imaging_plane_pixel_count_y = Imaging_plane_para[0], Imaging_plane_para[1], Imaging_plane_para[2], Imaging_plane_para[3]
    Imaging_plane_pixel_size_x, Imaging_plane_pixel_size_y = Imaging_plane_length_x/Imaging_plane_pixel_count_x, Imaging_plane_length_y/Imaging_plane_pixel_count_y
    Imaging_plane_x, Imaging_plane_y = torch.meshgrid(torch.linspace(-Imaging_plane_length_x/2+Imaging_plane_pixel_size_x/2, Imaging_plane_length_x/2-Imaging_plane_pixel_size_x/2, Imaging_plane_pixel_count_x), torch.linspace(-Imaging_plane_length_y/2+Imaging_plane_pixel_size_y/2, Imaging_plane_length_y/2-Imaging_plane_pixel_size_y/2, Imaging_plane_pixel_count_y))
    # print(PAT_x, PAT_y)
    # print(Imaging_plane_x, Imaging_plane_y, Imaging_plane_z)

    # Initial phase of transducers on PAT
    # phs_holo, amp_holo should be size of [BatchSize, PAT_pixel_count_x, PAT_pixel_count_y] such as [16, 50, 50]
    phs_holo, amp_holo = PhsHolo.squeeze(1), AmpHolo.squeeze(1)

    # Parameters of Piston Source
    p0 = 2 * torch.pi * d_33 * ρ * (f**2) * S * Vptp # a constant value

    if Algorithm_name == 'IB1':
        wave_number = 2 * torch.pi / Lam 
        propagator_xoy = torch.zeros((Imaging_plane_pixel_count_x, Imaging_plane_pixel_count_y, PAT_pixel_count_x, PAT_pixel_count_y), dtype = torch.complex128)
        distance = torch.zeros((Imaging_plane_pixel_count_x, Imaging_plane_pixel_count_y))
        Directivity_func = torch.zeros((Imaging_plane_pixel_count_x, Imaging_plane_pixel_count_y))
        for i in range(PAT_pixel_count_x):
            for j in range(PAT_pixel_count_y):
                distance = (torch.sqrt((Imaging_plane_x - PAT_x[i,j])**2 + (Imaging_plane_y - PAT_y[i,j])**2 + (Imaging_plane_z)**2))
                Directivity_func = torch.sinc(wave_number * (Imaging_plane_x -PAT_x[i,j]) / distance * L / 2) * torch.sinc(wave_number * (Imaging_plane_y -PAT_y[i,j]) / distance  * L / 2)
                propagator_xoy[:,:,i,j] = torch.exp(1j*(wave_number * distance)) * (p0 * Directivity_func / distance)
        # ##
        # # propagator_xoy_elementi = propagator_xoy[0][0].cpu().numpy()
        # # print(propagator_xoy_elementi.shape)
        # # exit()
        # propagator_amp, propagator_phs = torch.abs(propagator_xoy), torch.angle(propagator_xoy)
        # # for propagator_z in range(10):
        # propagator_z = 49
        # for propagator_j, propagator_i in zip([1,2,3],[0,1,-1]):
        #     plt.subplot(2,3, propagator_j)
        #     plt.imshow(propagator_amp[propagator_z][propagator_i].cpu().numpy()) # ,cmap='jet'
        #     plt.colorbar(fraction=0.046, pad=0.04)
        #     plt.axis('off')
        #     plt.subplot(2,3, propagator_j+3)
        #     plt.imshow(propagator_phs[propagator_z][propagator_i].cpu().numpy()) # ,cmap='jet'
        #     plt.colorbar(fraction=0.046, pad=0.04)
        #     plt.axis('off')
        # plt.tight_layout()
        # if not os.path.exists('./AlgorithmCheck/'):
        #     os.makedirs('./AlgorithmCheck/')
        # plt.savefig('./AlgorithmCheck/'+ Algorithm_name  +'_propagator_' + str(propagator_z) + '.png', bbox_inches='tight')
        # plt.clf()
        # exit()
        # ##

        target_holo = torch.sum(propagator_xoy * torch.exp( 1j * phs_holo.reshape(BatchSize, 1,1,PAT_pixel_count_x, PAT_pixel_count_y)), axis=(-1,-2))
        assert target_holo.shape == torch.Size([BatchSize, Imaging_plane_pixel_count_x, Imaging_plane_pixel_count_y]), "The shape of target hologram {} is wrong which should be {}".format(target_holo.shape, torch.Size([BatchSize, Imaging_plane_pixel_count_x, Imaging_plane_pixel_count_y]))
        target_amp = torch.abs(target_holo)
        target_phs = torch.angle(target_holo) # output range is [-pi, pi]
        target_phs = target_phs - torch.floor(target_phs/(2*torch.pi)) * (2*torch.pi) # [0, 2pi]
        target_amp, target_phs = target_amp.unsqueeze(1), target_phs.unsqueeze(1)

    elif Algorithm_name == 'IB2':
        propagator = torch.zeros((Imaging_plane_pixel_count_x, Imaging_plane_pixel_count_y, PAT_pixel_count_x, PAT_pixel_count_y), dtype=torch.complex128)
        target_holo = torch.zeros((Imaging_plane_pixel_count_x, Imaging_plane_pixel_count_y), dtype=torch.complex128)
        for Imaging_plane_pixel_i in range(Imaging_plane_pixel_count_x):
            for Imaging_plane_pixel_j in range(Imaging_plane_pixel_count_y):
                distance_between_Targetij2Source = torch.sqrt((Imaging_plane_x[Imaging_plane_pixel_i,Imaging_plane_pixel_j]-PAT_x)**2 + (Imaging_plane_y[Imaging_plane_pixel_i,Imaging_plane_pixel_j]-PAT_y)**2 + Imaging_plane_z**2)
                wavevector_x = (2 * torch.pi / Lam) * (Imaging_plane_x[Imaging_plane_pixel_i,Imaging_plane_pixel_j] - PAT_x) / distance_between_Targetij2Source
                wavevector_y = (2 * torch.pi / Lam) * (Imaging_plane_y[Imaging_plane_pixel_i,Imaging_plane_pixel_j] - PAT_y) / distance_between_Targetij2Source
                Directivity_func = torch.sinc(wavevector_x * L / 2) * torch.sinc(wavevector_y * L / 2)
                propagator[Imaging_plane_pixel_i,Imaging_plane_pixel_j,:,:] = torch.exp( 1j*((2 * torch.pi / Lam) * distance_between_Targetij2Source)) * (p0 * Directivity_func / distance_between_Targetij2Source)
                # target_holo[Target_Holo_i][Target_Holo_j] = torch.sum(propagator[Target_Holo_i][Target_Holo_j]* torch.exp( 1j * phs_holo))
        
        target_holo = torch.sum(propagator * torch.exp( 1j * phs_holo.reshape(BatchSize, 1,1,PAT_pixel_count_x, PAT_pixel_count_y)), axis=(-1,-2))
        assert target_holo.shape == torch.Size([BatchSize, Imaging_plane_pixel_count_x, Imaging_plane_pixel_count_y]), "The shape of target hologram {} is wrong which should be {}".format(target_holo.shape, torch.Size([BatchSize, Imaging_plane_pixel_count_x, Imaging_plane_pixel_count_y]))
        # print(target_holo.shape)
        target_amp = torch.abs(target_holo)
        target_phs = torch.angle(target_holo) # output range is [-pi, pi]
        target_phs = target_phs - torch.floor(target_phs/(2*torch.pi)) * (2*torch.pi) # [0, 2pi]
        target_amp, target_phs = target_amp.unsqueeze(1), target_phs.unsqueeze(1)
    ###
    elif Algorithm_name == 'ASM1':
        phs_holo, amp_holo = phs_holo.unsqueeze(1), amp_holo.unsqueeze(1)
        print(phs_holo.shape, amp_holo.shape)
        target_amp, target_phs = ASM(Imaging_plane_z, phs_holo, amp_holo, BatchSize=BatchSize)
        # print(target_amp.shape, target_phs.shape)
    
    else:
        print("Please specified the algorithm name for acoustic field propagation... ")

    if plot_results == True:
        # print(phs_holo[0].shape, target_amp[0].shape, target_phs[0].shape)
        phs_holo_plot, target_amp_plot, target_phs_plot = phs_holo[1].squeeze(0).cpu().numpy(), target_amp[1].squeeze(0).cpu().numpy(), target_phs[1].squeeze(0).cpu().numpy()
        # Plot the results
        plt.subplot(131)
        plt.imshow(phs_holo_plot) # ,cmap='jet'
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        plt.subplot(132)
        customized_colorbar = mcolors.LinearSegmentedColormap.from_list("mylist",["black","red"],N = 800)
        plt.imshow(target_amp_plot) # ,cmap=customized_colorbar
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(target_phs_plot) # ,cmap='jet'
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        plt.tight_layout() # 让子图之间的文字不重叠
        if not os.path.exists('./AlgorithmCheck/'):
            os.makedirs('./AlgorithmCheck/')
        plt.savefig('./AlgorithmCheck/'+ Algorithm_name  +'_code_test.png', bbox_inches='tight')
        plt.clf()
        central_axis_plot(list=list(target_amp_plot[:,int(target_amp_plot.shape[-1]/2)]), label='amplitude at central axis', xlabel='point index', ylabel='amplitude', save_path='./AlgorithmCheck/'+ Algorithm_name  +'amp_at_central_axis.png')
    
    return target_amp, target_phs

def normalize_amp(target_img, propagated_pressure, normalize_option):
    if normalize_option == 1:
    # -------------- 1 - Maximum of propagated_pressure Normalization --------------------
        propagated_pressure_nmlzd = torch.divide(propagated_pressure, torch.max(torch.max(propagated_pressure, dim=-1)[0], dim=-1)[0].unsqueeze(-1).unsqueeze(-1))
    elif normalize_option == 2:
    # -------------- 2 - Maximum of target_img Normalization -----------------------------
        # ratio = torch.round(torch.sum(torch.ones_like(target_img), dim=(1,2,3), keepdim=True) / torch.sum(target_img, dim=(1,2,3), keepdim=True), 2)
        ratio = torch.sqrt(torch.sum(torch.ones_like(target_img), dim=(1,2,3), keepdim=True) / torch.sum(target_img**2, dim=(1,2,3), keepdim=True))
        propagated_pressure_nmlzd = torch.divide(propagated_pressure, ratio)
        if propagated_pressure_nmlzd.max() > 1.0:
            propagated_pressure_nmlzd_g1 = torch.gt(propagated_pressure_nmlzd, 1.0)
            propagated_pressure_nmlzd_l1 = torch.le(propagated_pressure_nmlzd, 1.0)
            propagated_pressure_nmlzd_g1_num = torch.sum(propagated_pressure_nmlzd_g1)
            propagated_pressure_nmlzd_g1_mean = torch.mean(torch.masked_select(propagated_pressure_nmlzd, propagated_pressure_nmlzd_g1))
            # print("The sum num, pixel value of nomalized results which are greater than 1 are {} and {}, respectively. ".format(propagated_pressure_nmlzd_g1_num, propagated_pressure_nmlzd_g1_mean))
            # print(propagated_pressure_nmlzd.min(), propagated_pressure_nmlzd.max())
            # print(propagated_pressure_nmlzd_g1.min(), propagated_pressure_nmlzd_g1.max())
            propagated_pressure_nmlzd = propagated_pressure_nmlzd_g1*(2*target_img - propagated_pressure_nmlzd) + propagated_pressure_nmlzd_l1*propagated_pressure_nmlzd
            # print(propagated_pressure_nmlzd.min(), propagated_pressure_nmlzd.max())
            # exit()
            if propagated_pressure_nmlzd.min()<0.0 or propagated_pressure_nmlzd.max()>1.0:
                propagated_pressure_nmlzd = torch.divide(propagated_pressure, torch.max(torch.max(propagated_pressure, dim=-1)[0], dim=-1)[0].unsqueeze(-1).unsqueeze(-1))
    elif normalize_option == 3:
    # -------------- 3 - Enegy and Align Normalization ---------------------------------
        # ratio = torch.round(torch.sum(torch.ones_like(target_img), dim=(1,2,3), keepdim=True) / torch.sum(target_img, dim=(1,2,3), keepdim=True), 2)
        ratio = torch.sqrt(torch.sum(torch.ones_like(target_img), dim=(1,2,3), keepdim=True) / torch.sum(target_img**2, dim=(1,2,3), keepdim=True))
        propagated_pressure_nmlzd = torch.divide(propagated_pressure, ratio)
        if propagated_pressure_nmlzd.max() <= 2.0:
            propagated_pressure_nmlzd_l1 = torch.le(propagated_pressure_nmlzd, 1.0)
            propagated_pressure_nmlzd_g1 = torch.gt(propagated_pressure_nmlzd, 1.0)
            propagated_pressure_nmlzd = propagated_pressure_nmlzd*propagated_pressure_nmlzd_l1 + (2-propagated_pressure_nmlzd)*propagated_pressure_nmlzd_g1
        else:
            propagated_pressure_nmlzd_g2 = torch.gt(propagated_pressure_nmlzd, 2.0)
            propagated_pressure_nmlzd_g2_num = torch.sum(propagated_pressure_nmlzd_g2)
            propagated_pressure_nmlzd_g2_mean = torch.mean(torch.masked_select(propagated_pressure_nmlzd, propagated_pressure_nmlzd_g2))
            # print("The sum num, pixel value of nomalized results which are greater than 2 are {} and {}, respectively. ".format(propagated_pressure_nmlzd_g2_num, propagated_pressure_nmlzd_g2_mean))
            propagated_pressure_nmlzd = torch.divide(propagated_pressure, torch.max(torch.max(propagated_pressure, dim=-1)[0], dim=-1)[0].unsqueeze(-1).unsqueeze(-1))
    assert propagated_pressure_nmlzd.max() <= 1.0 and propagated_pressure_nmlzd.min() >= 0.0, "After normalization, the propagated_pressure_nmlzd still is not in the range of [0,1]"
    return propagated_pressure_nmlzd

def Init_net_from_scratch(device, my_img_ch=1, my_output_ch=1, my_output_process='', with_conv_7x7=False, with_conv6=False, with_conv_1x1_repeat=False, with_final_fc=False):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet_V3(img_ch=my_img_ch, output_ch=my_output_ch, output_process=my_output_process, with_conv_7x7=with_conv_7x7, with_conv6=with_conv6, with_conv_1x1_repeat=with_conv_1x1_repeat, with_final_fc=with_final_fc)
    init_weights(net, init_type='kaiming', gain=0.02)
    # torch.nn.DataParallel(net)
    net = net.to(device)
    return net

def Init_net_from_before(device, my_img_ch=1, my_output_ch=1, my_output_process='', net_path='', with_conv_7x7=False, with_conv6=False, with_conv_1x1_repeat=False, with_final_fc=False):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet_V3(img_ch=my_img_ch, output_ch=my_output_ch, output_process=my_output_process, with_conv_7x7=with_conv_7x7, with_conv6=with_conv6, with_conv_1x1_repeat=with_conv_1x1_repeat, with_final_fc=with_final_fc)
    init_weights(net, init_type='kaiming', gain=0.02)
    assert os.path.exists(net_path), "file: '{}' does not exist.".format(net_path)
    net.load_state_dict(torch.load(net_path, map_location = device), strict=False)
    # torch.nn.DataParallel(net)
    net = net.to(device)
    return net

def Init_net_from_pretrain(device, fixed_pretrained_model=True, my_img_ch=1, my_output_ch=1, my_output_process='', net_path='', with_conv_7x7=False, with_conv6=False, with_conv_1x1_repeat=False, with_final_fc=False):
    # net = UNet_V3(num_classes=128, img_ch=1,output_ch=1, output_process=my_output_process)
    net = UNet_V3(img_ch=my_img_ch, output_ch=my_output_ch, output_process=my_output_process, with_conv_7x7=with_conv_7x7, with_conv6=with_conv6, with_conv_1x1_repeat=with_conv_1x1_repeat, with_final_fc=with_final_fc)
    checkpoint = torch.load(net_path)
    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q'):
        # if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # print(k[len("module.encoder_q."):])
        # delete renamed or unused k
        del state_dict[k]
    net.load_state_dict(state_dict, strict=False)
    if fixed_pretrained_model == True:
        for layer in [net.Conv1, net.Conv2, net.Conv3, net.Conv4, net.Conv5]:
            for _, value in layer.named_parameters():
                value.requires_grad = False
                # print("freeze!!!!")
    # torch.nn.DataParallel(net)
    net = net.to(device)
    # print(set(msg.missing_keys))
    # {'Up4.up.2.bias', 'Up5.up.1.bias', 'Up5.up.1.weight', 'Up5.up.2.bias', 'Up3.up.1.weight', 'Up2.up.2.running_var', 'Up2.up.1.weight', 'Up3.up.2.bias', 'Up1.up.2.running_var', 'Conv_1x1.bias', 'Up3.up.2.running_var', 'Up1.up.1.weight', 'Up3.up.2.running_mean', 'Up2.up.2.bias', 'Up1.up.1.bias', 'Up3.up.1.bias', 'Up1.up.2.running_mean', 'Up1.up.2.weight', 'Up2.up.2.weight', 'Up4.up.2.running_var', 'Up2.up.1.bias', 'Up3.up.2.weight', 'Up4.up.1.bias', 'Up4.up.1.weight', 'Up1.up.2.bias', 'Up5.up.2.running_mean', 'Up4.up.2.weight', 'Up5.up.2.weight', 'Up2.up.2.running_mean', 'Up5.up.2.running_var', 'Conv_1x1.weight', 'Up4.up.2.running_mean'}
    # assert set(msg.missing_keys) == {"fc.weight", "fc.bias", ""}
    print("=> loaded pre-trained model '{}'".format(net_path))
    return net

def plot_single_curve(index, data, name, savepath, total_epochs, current_epochs, fontsize):
    savepath_fig = savepath[:-4] + '/' + current_epochs + '_epoch/' + str(index) + '_' + name + savepath[-4:]
    if not os.path.exists(savepath[:-4] + '/' + current_epochs + '_epoch/'):
        os.mkdir(savepath[:-4] + '/' + current_epochs + '_epoch/')
    plt.rcParams['figure.figsize'] = (8.0, 6.0)
    if current_epochs != 'Final':
        x = np.linspace(0, int(current_epochs)+1, len(data))
    else:
        x = np.linspace(0, total_epochs, len(data))
    plt.plot(x, data, 'ko', x, data, 'k-', label=name)
    plt.legend(loc='best')
    plt.xlabel("epochs", fontsize=fontsize)
    plt.ylabel("loss or metric", fontsize=fontsize)
    plt.title(name, fontsize=fontsize)
    plt.savefig(savepath_fig, dpi=300, bbox_inches='tight')
    plt.clf()


def plot_curves(list_packed_data, list_packed_name, savepath, total_epochs, current_epochs, fontsize):
    '''
    The length of list_packed_data is 21:
        1. plot by 1 figures (2*6): 8 train loss curves (7 items [acc, eff, unifo, npcc, percp, psnr, ssim], 1 total)
                                    8 validated loss curves (7 items [acc, eff, unifo, npcc, percp, psnr, ssim], 1 total)
        2. plot by 1 figures (3*3): 9 validated metrics curves (acc, pc, mse, ssim, effmean, effratio, unifom, sd, psnr)
    '''
    nrows1, ncols1 = 2, 8
    nrows2, ncols2 = 3, 3
    plt.rcParams['figure.figsize'] = (16.0, 12.0)
    fig1, ax = plt.subplots(nrows=nrows1, ncols=ncols1)
    axs = ax.flatten()
    savepath_fig1 = savepath[:-4] + '/' + current_epochs + '_epoch/' + 'loss_curves' + savepath[-4:]
    if not os.path.exists(savepath[:-4] + '/' + current_epochs + '_epoch/'):
        os.mkdir(savepath[:-4] + '/' + current_epochs + '_epoch/')
    dotlist = ['ko', 'gv', 'bs', 'y+', 'c>', 'mp', 'mx', 'r*', 'ko', 'gv', 'bs', 'y+', 'c>', 'mp', 'mx', 'r*']
    line_list = ['k--', 'g--', 'b--', 'y--', 'c--', 'm--', 'm--', 'r-', 'k--', 'g--', 'b--', 'y--', 'c--', 'm-', 'm-', 'r-']
    for i in range(nrows1*ncols1):
        if current_epochs != 'Final':
            x_train = np.linspace(0, int(current_epochs)+1, len(list_packed_data[i]))
        else:
            x_train = np.linspace(0, total_epochs, len(list_packed_data[i]))
        axs[i].plot(x_train, list_packed_data[i], dotlist[i], x_train, list_packed_data[i], line_list[i], label=list_packed_name[i])
        axs[i].set_xlabel('epochs', fontsize=fontsize)
        axs[i].set_ylabel('loss or metric', fontsize=fontsize)
        axs[i].set_title(list_packed_name[i], fontsize=fontsize)
    # ax.legend(bbox_to_anchor=(0.7, 0.5), loc='center left')
    plt.tight_layout() # 让子图之间的文字不重叠
    plt.savefig(savepath_fig1, dpi=300, bbox_inches='tight')
    plt.clf()

    fig2, ax = plt.subplots(nrows=nrows2, ncols=ncols2)
    axs = ax.flatten()
    savepath_fig2 = savepath[:-4] + '/' + current_epochs + '_epoch/' + 'metric_curves' + savepath[-4:]
    for i in range(nrows2*ncols2):
        if current_epochs != 'Final':
            x_val = np.linspace(0, int(current_epochs)+1, len(list_packed_data[i+(nrows1*ncols1)]))
        else:
            x_val = np.linspace(0, total_epochs, len(list_packed_data[i+(nrows1*ncols1)]))
        
        axs[i].plot(x_val, list_packed_data[i+(nrows1*ncols1)],'bo', x_val, list_packed_data[i+(nrows1*ncols1)],'k-', label=list_packed_name[i+(nrows1*ncols1)])
        axs[i].set_xlabel('epochs', fontsize=fontsize)
        axs[i].set_ylabel('loss', fontsize=fontsize)
        axs[i].set_title(list_packed_name[i+(nrows1*ncols1)], fontsize=fontsize)
    # ax.legend(bbox_to_anchor=(0.7, 0.5), loc='center left')
    plt.tight_layout()
    plt.savefig(savepath_fig2, dpi=300, bbox_inches='tight')
    plt.clf()

    print("The training and validation curves have been ploted! ")

def Sub_plot(img_data_list:list, img_name_list:list, img_position_list:list, vacant_position_list:list, plot_nrows:int, plot_ncols:int, plot_dpi:int, plot_savepath:str, plot_savename:str):
    # Save by subplots
    # Each item in img_data_list: numpy, size of (m, n)
    # Each item in img_name_list: string, amp or phs
    # # plot_nrows, plot_ncols, plot_dpi: int
    num_img = len(img_data_list)
    assert num_img <= plot_nrows * plot_ncols, "The subplot position ({}) is not enough for {} images"
    assert len(img_data_list) == len(img_name_list), "The lengths of img_data_list and img_name_list are different"
    fig, ax = plt.subplots(nrows=plot_nrows, ncols=plot_ncols)
    axs = ax.flatten()
    amp_colorbar = mcolors.LinearSegmentedColormap.from_list("mylist",["black","red"],N = 800) # Colorbar for amplitude data
    phs_colorbar = 'jet'

    # For vacant postion in the subplot canvas
    for vacant_index in vacant_position_list:
        plt.delaxes(axs[vacant_index])
    # For the position with image in the subplot canvas
    # 'amp', 'phs', 'phs', 'offset_direct', 'offset_mod', 'offset_fold', 'offset_align'
    for img_index, img_data_i, img_name_i in zip(img_position_list, img_data_list, img_name_list):
        if 'amp' in img_name_i:
            cmap_i = amp_colorbar
            # vmax_i = img_data_i.max()
            vmax_i = 1.0
            vmin_i = 0.0
            # assert np.abs(img_data_i.max() - vmax_i) <= 1e-2, "The {}'s max is not {}".format(img_name_i, vmax_i)
        elif 'phs' in img_name_i:
            cmap_i = phs_colorbar
            vmax_i = 1.0 * 2 * np.pi
            vmin_i = 0.0
            # assert np.abs(img_data_i.max() - vmax_i) <= 0.1, "The {}'s max is not {}".format(img_name_i, vmax_i)
            # assert img_data_i.max() <= (vmax_i+1e-2), "The {}'s max is not smaller than {}".format(img_name_i, vmax_i)
            # if np.abs(img_data_i.max() - vmax_i) > 1e-2:
            #     print(np.abs(img_data_i.max() - vmax_i))
            # if img_data_i.max() > vmax_i:
            #     print(img_data_i.max())
        elif img_name_i == 'offset_direct':
            cmap_i = phs_colorbar
            vmax_i = 2.0 * np.pi
            vmin_i = -2.0 * np.pi
        elif img_name_i == 'offset_mod' or img_name_i == 'offset_fold':
            cmap_i = phs_colorbar
            vmax_i = 2.0 * np.pi
            vmin_i = 0.0
        elif img_name_i == 'offset_align':
            cmap_i = phs_colorbar
            vmax_i = np.pi
            vmin_i = -np.pi
        else:
            print("The img_name is neither amp nor phs") 
            exit()
        assert img_data_i.max() <= (vmax_i+1e-4), "The {}'s max is {} which is not smaller than {}".format(img_name_i, img_data_i.max(), vmax_i)
        assert img_data_i.min() >= (vmin_i-1e-4), "The {}'s min is {} which is not larger than {}".format(img_name_i, img_data_i.min(), vmin_i)
        img = ax[img_index].imshow(img_data_i, cmap=cmap_i, vmin=vmin_i, vmax=vmax_i)
        ax[img_index].set_title(img_name_i)
        # axs[img_index].axis('off')
        fig.colorbar(img, ax=ax[img_index], fraction=0.046, pad=0.04) # pad=0.005 用于colorbar与image的距离, fraction=0.046 用于使得colorbar的长度与image一致,shrink=0.4 用于colorbar的缩放
    
    [axi.set_axis_off() for axi in ax.ravel()] # To turn off axes for all subplots
    plt.tight_layout() # 让子图之间的文字不重叠
    if not os.path.exists(plot_savepath):
        os.makedirs(plot_savepath)
    plt.savefig(plot_savepath + '/' + plot_savename,dpi=plot_dpi, bbox_inches='tight')
    plt.clf()




def EP_Collect(train_loader, train_para, savepath_para, PAT_para, propagation_para, EP_collect_para):
    '''
    This function is used for experience pools (EPs) collection
    para:
        train_loader: training loader
        train_para: maintains training parameters
        savepath_para: indicates save path of experience pools (EPs)
        PAT_para: hardware parameters related to PAT
        propagation_para: acoustic field propagation parameters
        EP_collect_para: indicates EPs' capacities, update frequency, triggering update iteration
    '''
    print("===========================================================================")
    print("Let's collect experience pools...... ")
    EP1_capacity, EP2_capacity = EP_collect_para['EP1_capacity'], EP_collect_para['EP2_capacity']
    len_trainloader_forEP = len(train_loader)
    assert EP1_capacity <= len_trainloader_forEP*train_para['batch_size'] and EP2_capacity <= len_trainloader_forEP*train_para['batch_size'], "The expected capacity of EP1 or EP2 is greater than length of dataloader"
    EP1_data, EP2_data = np.zeros((2, PAT_para['pixel_num_in_x'] , PAT_para['pixel_num_in_y'])), np.zeros((2, PAT_para['pixel_num_in_x'] , PAT_para['pixel_num_in_y']))
    device = train_para['device']
    EPcollect_net = train_para['net'].to(device)
    EPcollect_net.eval()
    with torch.no_grad():
        EPcollect_bar = tqdm(train_loader)
        for iter_i, data_i in enumerate(EPcollect_bar):
            EP_TargetAmpHolo_input = data_i.to(train_para['device'])
            EP_energy_ratio = torch.ones_like(EP_TargetAmpHolo_input)*(PAT_para['pixel_num_in_x']*PAT_para['pixel_num_in_y']) / torch.sum(EP_TargetAmpHolo_input**2, dim=(-1,-2), keepdim=True)
            # EP_amplitude_ratio = torch.round(torch.sqrt(EP_energy_ratio)*100)/100
            EP_amplitude_ratio = torch.sqrt(EP_energy_ratio)
            if train_para['input_scaleup']:
                EP_PredPhsHolo_output = EPcollect_net(EP_TargetAmpHolo_input * EP_amplitude_ratio)
            else:
                print("Please check the target holograms preparation (whether scale up) for neural network input! ")
                exit()
            EP_UniformAmpHolo = torch.ones_like(EP_PredPhsHolo_output)
            if train_para['output_process'] == 'sigmoid_0to1':
                assert EP_PredPhsHolo_output.max() <= 1.0 and EP_PredPhsHolo_output.min() >= 0.0, "The EP_PredPhsHolo_output is out of range [0,1] defined by train_para['output_process']: {}".format(train_para['output_process'])
                EP_ReconAmpHolo, EP_ReconPhsHolo = ASM(d = propagation_para['propagation_Dist'], PhsHolo = EP_PredPhsHolo_output*(2*torch.pi), AmpHolo = EP_UniformAmpHolo, fs=propagation_para['sample frequency'])
                EP_PredPhsHolo_output_nmlzd = EP_PredPhsHolo_output
                # EP_ReconPhsHolo_rearrange = EP_ReconPhsHolo - torch.floor(EP_ReconPhsHolo/(2*torch.pi)) * (2*torch.pi)
            elif train_para['output_process'] == 'tanh_minus1to1':
                assert EP_PredPhsHolo_output.max() <= 1.0 and EP_PredPhsHolo_output.min() >= -1.0, "The EP_PredPhsHolo_output is out of range [-1,1] defined by train_para['output_process']: {}".format(train_para['output_process'])
                EP_ReconAmpHolo, EP_ReconPhsHolo = ASM(d = propagation_para['propagation_Dist'], PhsHolo = EP_PredPhsHolo_output*(torch.pi), AmpHolo = EP_UniformAmpHolo, fs=propagation_para['sample frequency'])
                EP_PredPhsHolo_output_nmlzd = ((EP_PredPhsHolo_output) - torch.floor(EP_PredPhsHolo_output/2) * 2)/2
                # EP_ReconPhsHolo_rearrange = EP_ReconPhsHolo - torch.floor(EP_ReconPhsHolo/(2*torch.pi)) * (2*torch.pi)
            EP_ReconAmpHolo_nmlzd = normalize_amp(target_img=EP_TargetAmpHolo_input, propagated_pressure=EP_ReconAmpHolo, normalize_option=propagation_para['normalize option'])
            EP_BackAmpHolo, EP_BackPhsHolo = ASM(d = -propagation_para['propagation_Dist'], PhsHolo = EP_ReconPhsHolo, AmpHolo = (EP_TargetAmpHolo_input * EP_amplitude_ratio), fs=propagation_para['sample frequency'])
            EP_BackPhsHolo_nmlzd = ((EP_BackPhsHolo/torch.pi) - torch.floor((EP_BackPhsHolo/torch.pi)/2) * 2)/2 
            if (iter_i+1)*train_para['batch_size'] <= EP1_capacity:
                # Collection {EP_ReconAmpHolo_nmlzd, EP_PredPhsHolo_output_nmlzd} data pairs to EP1
                for i in range(train_para['batch_size']):
                    EP1_data[0,:,:] = EP_ReconAmpHolo_nmlzd.cpu().numpy()[i,:,:,:] # Amp
                    EP1_data[1,:,:] = EP_PredPhsHolo_output_nmlzd.cpu().numpy()[i,:,:,:] # Phs
                    if not os.path.exists(savepath_para['EP_basepath'] + "/EP1/"):
                        os.mkdir(savepath_para['EP_basepath'] + "/EP1/")
                    np.save(savepath_para['EP_basepath'] + "/EP1/" + str(iter_i*train_para['batch_size']+i) + ".npy", EP1_data)
            if (iter_i+1)*train_para['batch_size'] <= EP2_capacity:
                # Collection {EP_TargetAmpHolo_input, EP_BackPhsHolo_nmlzd} data pairs to EP2
                for i in range(train_para['batch_size']):
                    EP2_data[0,:,:] = EP_TargetAmpHolo_input.cpu().numpy()[i,:,:,:] # Amp
                    EP2_data[1,:,:] = EP_BackPhsHolo_nmlzd.cpu().numpy()[i,:,:,:] # Phs
                    if not os.path.exists(savepath_para['EP_basepath'] + "/EP2/"):
                        os.mkdir(savepath_para['EP_basepath'] + "/EP2/")
                    np.save(savepath_para['EP_basepath'] + "/EP2/" + str(iter_i*train_para['batch_size']+i) + ".npy", EP2_data)
        assert (iter_i+1) == len_trainloader_forEP, "There is something wrong with the iteration to go through train loader for Experience pool collection! "
        print("Our **expected** capacities of EP1 and EP 2 are {} and {}, respectively! ".format(EP1_capacity, EP2_capacity))
        Actual_data_EP1 = EP1_capacity*(EP1_capacity<=len_trainloader_forEP*train_para['batch_size']) + len_trainloader_forEP*train_para['batch_size']*(EP1_capacity>len_trainloader_forEP*train_para['batch_size'])
        Actual_data_EP2 = EP2_capacity*(EP2_capacity<=len_trainloader_forEP*train_para['batch_size']) + len_trainloader_forEP*train_para['batch_size']*(EP2_capacity>len_trainloader_forEP*train_para['batch_size'])
        print("After {} iters, we collected {} data pairs to EP1, while {} to EP2".format(len_trainloader_forEP, Actual_data_EP1, Actual_data_EP2))
        print("The Epxerience pool collection is done! ")
        print("===========================================================================\n")
