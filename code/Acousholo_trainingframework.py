import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd 
from tqdm import tqdm
import datetime
from Acousholo_auxiliaryfunc import ASM, normalize_amp, plot_curves, plot_single_curve, lr_update, net_freeze
from Acousholo_evaluation import assess, visualize, save_results_trr
from Acousholo_lossfunc import Lacc, Leff, Lunifo, Lnpcc, Lpercp, Lpercp_freq, Lpercpfg, Lpsnr, Lpsnr_NoNormalization, Lssim, LSumSquareMAE, Lcos, Lpwl, GradLoss, GradWeightedLoss  # for loss functions

# Commands:
    # The key of dictionary can be accessed by command '[*dictname][index]', while corresponding value can be accessedd by command 'dictname[[*dictname][index]]'
    # [1/2**(i+1) for i in range(5)] # 5, is lr decrease times, 1/2, 1/4, 1/8, 1/16, 1/32, decrease_after should times epochs and perform int operation and accumulate

def DL_train_val(train_para, data_loader, Criteria_para, lr_para, netfreeze_para, savepath_para, PAT_para, propagation_para, fontsize=10):
    '''
    Args:
        train_para: dictionary; including training relative information
        data_loader: list; [train_loader, valid_loader, test_loader] or [total_loader, train_loader, valid_loader, test_loader]
        Criteria_para: dictionary; including loss function information
        lr_para: dictionary; including learning rate information
        netfreeze_para: dictionary; including network freeze information
        savepath_para: dictionary; including save results path information
        PAT_para: hardware information
        propagation_para: ASM propagation parameter
        fontsize:plot font size
    '''
    # current_lr = lr_para['lr_decrease']
    # current_net = train_para['net']
    train_avg_err_list = []
    train_avg_ampmse_list = []
    train_avg_acc_list, train_avg_eff_list, train_avg_unifo_list = [], [], []
    train_avg_npcc_list, train_avg_percp_list, train_avg_lpercp_freq_list, train_avg_percpfg_list = [], [], [], []
    train_avg_psnr_list, train_avg_ssim_list = [], []
    train_avg_sse_list = []
    train_avg_gradloss_list, train_avg_gradweightedloss_list = [], []
    
    best_valid_psnr = 0.0
    
    val_avg_lphsmae_list, val_avg_lampmse_list = [], []
    val_avg_lacc_list, val_avg_leff_list, val_avg_lunifo_list, val_avg_err_list = [], [], [], []
    val_avg_lnpcc_list, val_avg_lpercp_list, val_avg_lpercp_freq_list, val_avg_lpercpfg_list = [], [], [], []
    val_avg_lpsnr_list, val_avg_lssim_list = [], []
    val_avg_lsse_list = []
    val_avg_gradloss_list, val_avg_gradweightedloss_list = [], []
    val_epoch_i_list = []
    val_avg_macc_list, val_avg_mpc_list, val_avg_mmse_list, val_avg_mssim_list, val_avg_meffmean_list, val_avg_meffratio_list, val_avg_munifo_list, val_avg_msd_list, val_avg_mpsnr_list = [], [], [], [], [], [], [], [], []
    # test_avg_err_list = []
    # test_avg_acc_list, test_avg_pc_list, test_avg_mse_list, test_avg_ssim_list, test_avg_effmean_list, test_avg_effratio_list, test_avg_unifo_list, test_avg_sd_list, test_avg_psnr_list = [], [], [], [], [], [], [], [], []

    criter_Lacc = Lacc() # do not need normalization of y_pred
    criter_Leff = Leff() # with nomalized y_pred
    criter_Lunifo = Lunifo() # do not need normalization of y_pred
    criter_Lnpcc = Lnpcc() # do not need normalization of y_pred
    criter_Lpercp = Lpercp() # with energy-based scale up y_true
    criter_Lpercp_freq = Lpercp_freq()
    criter_Lpercpfg = Lpercpfg() # with energy-based scale up y_true
    criter_Lpsnr = Lpsnr() # with nomalized y_pred
    criter_Lssim = Lssim() # with nomalized y_pred
    criter_LSumSquareMAE = LSumSquareMAE() 

    criter_GradLoss = GradLoss()
    criter_GradWeightedLoss = GradWeightedLoss()

    train_avg_phsmae_list, train_avg_phsmse_list, train_avg_phscos_list, train_avg_phspwl_list = [], [], [], []
    criter_Lmae = nn.L1Loss()
    criter_Lmse = nn.MSELoss()
    criter_Lcos = Lcos()
    criter_Lpwl = Lpwl() # Piecewise linear loss

    # count_loss_stability = 0.0
    loss_weigtsdecay_epoch = []


    if len(data_loader) == 4:
        train_loader, valid_loader, test_loader = data_loader[1], data_loader[2], data_loader[3]
    elif len(data_loader) == 3:
        train_loader, valid_loader, test_loader = data_loader[0], data_loader[1], data_loader[2]
    else:
        print("Train or validate or test loader does not exist. ")
        exit()
    len_trainloader = len(train_loader)
    len_validloader = len(valid_loader)
    len_testloader = len(test_loader)

    params = [p for p in train_para['net'].parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr_para['lr'])

    for epoch_i in range(train_para['epochs']):
        # Learning rate decrease technique
        if lr_para['lr_decrease'] and epoch_i in lr_para['decrease_at']:
            lr_para['lr'] = lr_update(lr_para, epoch_i)
            optimizer = optim.Adam(params, lr=lr_para['lr'])
        # Network freeze technique 
        if netfreeze_para['freeze'] and str(epoch_i) in [*netfreeze_para['freeze_dict']]:
            optimizer, train_para['net'] = net_freeze(train_para, netfreeze_para, lr_para, epoch_i)
        # Initialize the average variables for training for each epoch
        train_avg_err = 0.0
        train_avg_acc, train_avg_eff, train_avg_unifo = 0.0, 0.0, 0.0
        train_avg_npcc, train_avg_percp, train_avg_percp_freq, train_avg_percpfg = 0.0, 0.0, 0.0, 0.0
        train_avg_psnr, train_avg_ssim = 0.0, 0.0
        train_avg_sse = 0.0
        train_avg_gradloss, train_avg_gradweightedloss = 0.0, 0.0
        train_avg_phsmae, train_avg_phsmse, train_avg_phscos, train_avg_phspwl = 0.0, 0.0, 0.0, 0.0
        train_avg_ampmse = 0.0
        current_net = train_para['net']
        current_net.train()
        train_bar = tqdm(train_loader)
        for iter_i, data_i in enumerate(train_bar):
            # ===================================== Self-supervised Learning ===================================== #
            if train_para['training_method'] == 'SSL' or train_para['training_method'] == 'SSL_additionC_NFP' or train_para['training_method'] == 'SSL_downstream' or train_para['training_method'] == 'SSL_continue': # Input without annotations
                TargetAmpHolo_input = data_i.to(train_para['device'])
                energy_ratio = torch.ones_like(TargetAmpHolo_input)*(PAT_para['pixel_num_in_x']*PAT_para['pixel_num_in_y']) / torch.sum(TargetAmpHolo_input**2, dim=(-1,-2), keepdim=True)
                # amplitude_ratio = torch.round(torch.sqrt(energy_ratio)*100)/100
                amplitude_ratio = torch.sqrt(energy_ratio)
                # amplitude_ratio = torch.sqrt((PAT_para['pixel_num_in_x']*PAT_para['pixel_num_in_y']) / torch.sum(TargetAmpHolo_input**2, dim=(-1,-2,-3)))
                optimizer.zero_grad()
                if train_para['input_scaleup']:
                    TargetAmpHolo_input_scaleup = TargetAmpHolo_input * amplitude_ratio
                    # print(torch.sum(TargetAmpHolo_input_scaleup**2, dim=(-1,-2)))
                    if not (torch.abs(torch.sum(TargetAmpHolo_input_scaleup**2, dim=(-1,-2))-(PAT_para['pixel_num_in_x']*PAT_para['pixel_num_in_y'])) <= 1).all():
                        with open('./EnergyCheck.txt', 'a') as file0:
                            print(torch.sum(TargetAmpHolo_input_scaleup**2, dim=(-1,-2)), file=file0)
                    # assert (torch.abs(torch.sum(TargetAmpHolo_input_scaleup**2, dim=(-1,-2))-(PAT_para['pixel_num_in_x']*PAT_para['pixel_num_in_y'])) <= 1).all() and TargetAmpHolo_input_scaleup.min() >= 0.0, "The total energy of TargetAmpHolo_input_scaleup is not 2500"
                    if train_para['training_method'] == 'SSL' or train_para['training_method'] == 'SSL_downstream' or train_para['training_method'] == 'SSL_continue':
                        PredPhsHolo_output = current_net(TargetAmpHolo_input * amplitude_ratio)
                    if train_para['training_method'] == 'SSL_additionC_NFP':
                        assert (torch.cat(((TargetAmpHolo_input * amplitude_ratio), amplitude_ratio), 1)).size() == torch.Size([train_para['batch_size'], 2, PAT_para['pixel_num_in_x'], PAT_para['pixel_num_in_y']]), "The tensor cat is wrong, please check it  "
                        PredPhsHolo_output = current_net(torch.cat(((TargetAmpHolo_input * amplitude_ratio), amplitude_ratio), 1))
                else:
                    PredPhsHolo_output = current_net(TargetAmpHolo_input)        
                UniformAmpHolo = torch.ones_like(PredPhsHolo_output)
                if train_para['output_process'][:12] == 'sigmoid_0to1':
                    assert PredPhsHolo_output.max() <= 1.0 and PredPhsHolo_output.min() >= 0.0, "The PredPhsHolo_output is out of range [0,1] defined by train_para['output_process']: {}".format(train_para['output_process'])
                    ReconAmpHolo, ReconPhsHolo = ASM(d = propagation_para['propagation_Dist'], PhsHolo = PredPhsHolo_output*(2*torch.pi), AmpHolo = UniformAmpHolo, fs=propagation_para['sample frequency'], BatchSize=train_para['batch_size'])
                elif train_para['output_process'] == 'tanh_minus1to1':
                    assert PredPhsHolo_output.max() <= 1.0 and PredPhsHolo_output.min() >= -1.0, "The PredPhsHolo_output is out of range [-1,1] defined by train_para['output_process']: {}".format(train_para['output_process'])
                    ReconAmpHolo, ReconPhsHolo = ASM(d = propagation_para['propagation_Dist'], PhsHolo = PredPhsHolo_output*(torch.pi), AmpHolo = UniformAmpHolo, fs=propagation_para['sample frequency'], BatchSize=train_para['batch_size'])
                # Attention: the output phase of ASM function is change to the range of [-pi, pi] !!!!!!!!!!!!!!!!!!!!!!
                ReconAmpHolo_nmlzd = normalize_amp(target_img=TargetAmpHolo_input, propagated_pressure=ReconAmpHolo, normalize_option=propagation_para['normalize option'])

                iter_lacc = criter_Lacc(y_true=TargetAmpHolo_input, y_pred=ReconAmpHolo)
                iter_leff = criter_Leff(y_true=TargetAmpHolo_input, y_pred=ReconAmpHolo_nmlzd)
                iter_lunifo = criter_Lunifo(y_true=TargetAmpHolo_input, y_pred=ReconAmpHolo)
                iter_lnpcc = criter_Lnpcc(y_true=TargetAmpHolo_input, y_pred=ReconAmpHolo)
                iter_lpercp = criter_Lpercp(y_true=TargetAmpHolo_input*amplitude_ratio, y_pred=ReconAmpHolo)
                iter_lpercp_freq = criter_Lpercp_freq(y_true=torch.fft.fftshift(torch.fft.fftn((TargetAmpHolo_input*amplitude_ratio).to(torch.complex128))), y_pred=torch.fft.fftshift(torch.fft.fftn(ReconAmpHolo.to(torch.complex128))))
                iter_lpercpfg = criter_Lpercpfg(y_true=TargetAmpHolo_input, y_true_scaleup=TargetAmpHolo_input*amplitude_ratio, y_pred=ReconAmpHolo)
                iter_lpsnr = criter_Lpsnr(y_true=TargetAmpHolo_input, y_pred=ReconAmpHolo_nmlzd)
                iter_lssim = criter_Lssim(y_true=TargetAmpHolo_input, y_pred=ReconAmpHolo_nmlzd)
                iter_sse = criter_LSumSquareMAE(y_true=TargetAmpHolo_input, y_pred=ReconAmpHolo_nmlzd)# sse means sum of square error
                iter_gradloss = criter_GradLoss(y_true=TargetAmpHolo_input, y_pred=ReconAmpHolo_nmlzd)
                iter_gradweightedloss = criter_GradWeightedLoss(y_true=TargetAmpHolo_input, y_pred=ReconAmpHolo_nmlzd)
                
                if Criteria_para['losfunc'] == 'acc_eff':
                    assert len(Criteria_para['lamda']) == 1, "The weights of the total loss are not compile to the loss design! "
                    iter_loss = iter_lacc + Criteria_para['lamda'][0] * iter_leff
                elif Criteria_para['losfunc'] == 'acc_eff_unifo':
                    assert len(Criteria_para['lamda']) == 2, "The weights of the total loss are not compile to the loss design! "
                    iter_loss = iter_lacc + Criteria_para['lamda'][0] * iter_leff + Criteria_para['lamda'][1] * iter_lunifo
                elif Criteria_para['losfunc'] == 'acc_eff_unifo_with_weightdecay':
                    assert len(Criteria_para['lamda']) == 3, "The weights of the total loss are not compile to the loss design! "
                    iter_loss = Criteria_para['lamda'][0] * iter_lacc + Criteria_para['lamda'][1] * iter_leff + Criteria_para['lamda'][2] * iter_lunifo
                elif Criteria_para['losfunc'] == 'percp_psnr_percpfg_acc_eff_unifo_with_weightdecay':
                    assert len(Criteria_para['lamda']) == 6, "The weights of the total loss are not compile to the loss design! "
                    iter_loss = Criteria_para['lamda'][0] * iter_lpercp + Criteria_para['lamda'][1] * iter_lpsnr + Criteria_para['lamda'][2] * iter_lpercpfg + Criteria_para['lamda'][3] * iter_lacc + Criteria_para['lamda'][4] * iter_leff + Criteria_para['lamda'][5] * iter_lunifo
                elif Criteria_para['losfunc'] == 'npcc_percp':
                    assert len(Criteria_para['lamda']) == 1, "The weights of the total loss are not compile to the loss design! "
                    iter_loss = iter_lnpcc + Criteria_para['lamda'][0] * iter_lpercp
                elif Criteria_para['losfunc'] == 'percp_acc_eff_unifo_npcc':
                    assert len(Criteria_para['lamda']) == 5, "The weights of the total loss are not compile to the loss design! "
                    iter_loss = Criteria_para['lamda'][0] * iter_lpercp + Criteria_para['lamda'][1] * iter_lacc + Criteria_para['lamda'][2] * iter_leff + Criteria_para['lamda'][3] * iter_lunifo + Criteria_para['lamda'][4] * iter_lnpcc 
                elif Criteria_para['losfunc'] == 'percp':
                    iter_loss = iter_lpercp
                elif Criteria_para['losfunc'] == 'sse':
                    iter_loss = iter_sse
                elif Criteria_para['losfunc'] == 'percp_acc':
                    assert len(Criteria_para['lamda']) == 1, "The weights of the total loss are not compile to the loss design! "
                    iter_loss = iter_lpercp + Criteria_para['lamda'][0] * iter_lacc
                elif Criteria_para['losfunc'] == 'percp_psnr':
                    assert len(Criteria_para['lamda']) == 2, "The weights of the total loss are not compile to the loss design! "
                    iter_loss = Criteria_para['lamda'][0] * iter_lpercp + Criteria_para['lamda'][1] * iter_lpsnr
                elif Criteria_para['losfunc'] == 'percp_psnr_ssim':
                    assert len(Criteria_para['lamda']) == 3, "The weights of the total loss are not compile to the loss design! "
                    iter_loss = Criteria_para['lamda'][0] * iter_lpercp + Criteria_para['lamda'][1] * iter_lpsnr + Criteria_para['lamda'][2] * iter_lssim
                elif Criteria_para['losfunc'] == 'percp_psnr_percpfg':
                    assert len(Criteria_para['lamda']) == 3, "The weights of the total loss are not compile to the loss design! "
                    iter_loss = Criteria_para['lamda'][0] * iter_lpercp + Criteria_para['lamda'][1] * iter_lpsnr + Criteria_para['lamda'][2] * iter_lpercpfg    
                elif Criteria_para['losfunc'] == 'percp_psnr_ssim':
                    assert len(Criteria_para['lamda']) == 3, "The weights of the total loss are not compile to the loss design! "
                    iter_loss = Criteria_para['lamda'][0] * iter_lpercp + Criteria_para['lamda'][1] * iter_lpsnr + Criteria_para['lamda'][2] * iter_lssim
                elif Criteria_para['losfunc'] == 'percp_percpfreq_ssim':
                    assert len(Criteria_para['lamda']) == 3, "The weights of the total loss are not compile to the loss design! "
                    iter_loss = Criteria_para['lamda'][0] * iter_lpercp + Criteria_para['lamda'][1] * iter_lpercp_freq + Criteria_para['lamda'][2] * iter_lssim
                elif Criteria_para['losfunc'] == 'percp_percpfreq_psnr_ssim':
                    assert len(Criteria_para['lamda']) == 4, "The weights of the total loss are not compile to the loss design! "
                    iter_loss = Criteria_para['lamda'][0] * iter_lpercp + Criteria_para['lamda'][1] * iter_lpercp_freq + Criteria_para['lamda'][2] * iter_lpsnr + Criteria_para['lamda'][3] * iter_lssim
                elif Criteria_para['losfunc'] == 'percp_percpfreq_psnr_ssim_grad':
                    iter_loss = Criteria_para['lamda'][0] * iter_lpercp + Criteria_para['lamda'][1] * iter_lpercp_freq + Criteria_para['lamda'][2] * iter_lpsnr + Criteria_para['lamda'][3] * iter_lssim + Criteria_para['lamda'][4] * iter_gradloss
                elif Criteria_para['losfunc'] == 'percp_percpfreq_psnr_ssim_gradweighted':
                    iter_loss = Criteria_para['lamda'][0] * iter_lpercp + Criteria_para['lamda'][1] * iter_lpercp_freq + Criteria_para['lamda'][2] * iter_lpsnr + Criteria_para['lamda'][3] * iter_lssim + Criteria_para['lamda'][4] * iter_gradweightedloss
                else:
                    print("Please define the loss function for you Deep learning! ")
                    exit()

                iter_loss.backward()
                optimizer.step()
                train_avg_acc += iter_lacc.item()
                train_avg_eff += iter_leff.item()
                train_avg_unifo += iter_lunifo.item()
                train_avg_npcc += iter_lnpcc.item()
                train_avg_percp += iter_lpercp.item()
                train_avg_percp_freq += iter_lpercp_freq.item()
                train_avg_percpfg += iter_lpercpfg.item()
                train_avg_psnr += iter_lpsnr.item()
                train_avg_ssim += iter_lssim.item()
                train_avg_sse += iter_sse.item()
                train_avg_gradloss += iter_gradloss.item()
                train_avg_gradweightedloss += iter_gradweightedloss.item()
                train_avg_err += iter_loss.item()
                train_bar.desc = "{}, {}, train epoch [{}/{}], los: {} (lacc:{:.2f}, leff:{:.2f}, lunifo:{:.2f}, lnpcc:{:.2f}, lpercp:{:.2f}, lpercpfg:{:.2f}, lpsnr:{:.2f}, lssim:{:.2f}, gradloss:{:.2f}, gradweighted:{:.2f})".format(train_para['training_method'], Criteria_para['losfunc'], epoch_i, train_para['epochs'], iter_loss, iter_lacc, iter_leff, iter_lunifo, iter_lnpcc, iter_lpercp, iter_lpercpfg, iter_lpsnr, iter_lssim, iter_gradloss, iter_gradweightedloss)

            # ===================================== Supervised Learning ===================================== # 
            elif train_para['training_method'] == 'SL' or train_para['training_method'] == 'SL_NIPS': # Input with retrieved phase annotations from the eperience pools
                # Attention: for training dataset loader, we should change orginal expected hologram with the data pairs collected in experience pools
                TargetAmpHolo_input, SourcePhsHolo_label = data_i
                TargetAmpHolo_input, SourcePhsHolo_label = TargetAmpHolo_input.to(train_para['device']), SourcePhsHolo_label.to(train_para['device'])
                energy_ratio = torch.ones_like(TargetAmpHolo_input)*(PAT_para['pixel_num_in_x']*PAT_para['pixel_num_in_y']) / torch.sum(TargetAmpHolo_input**2, dim=(-1,-2), keepdim=True)
                # amplitude_ratio = torch.round(torch.sqrt(energy_ratio)*100)/100
                amplitude_ratio = torch.sqrt(energy_ratio)
                optimizer.zero_grad()
                if train_para['input_scaleup']:
                    PredPhsHolo_output = current_net(TargetAmpHolo_input * amplitude_ratio)
                else:
                    PredPhsHolo_output = current_net(TargetAmpHolo_input)
                iter_lmse = criter_Lmse(SourcePhsHolo_label, PredPhsHolo_output)
                iter_lmae = criter_Lmae(SourcePhsHolo_label, PredPhsHolo_output)
                iter_lcos = criter_Lcos(SourcePhsHolo_label, PredPhsHolo_output)
                iter_lpwl = criter_Lpwl(SourcePhsHolo_label, PredPhsHolo_output)

                if Criteria_para['losfunc'] == 'Phse_mae':
                    iter_loss = iter_lmae
                elif Criteria_para['losfunc'] == 'Phse_mse':
                    iter_loss = iter_lmse
                elif Criteria_para['losfunc'] == 'Phse_cos':
                    iter_loss = iter_lcos
                elif Criteria_para['losfunc'] == 'Phse_pwl':
                    iter_loss = iter_lpwl
                else:
                    print("Please define the loss function for you Deep learning! ")
                    exit()

                iter_loss.backward()
                optimizer.step()
                train_avg_phsmae += iter_lmae.item()
                train_avg_phsmse += iter_lmse.item()
                train_avg_phscos += iter_lcos.item()
                train_avg_phspwl += iter_lpwl.item()
                train_avg_err += iter_loss.item()
                train_bar.desc = "{}, {}, train epoch [{}/{}], los: {} (Phsmae:{:.2f}, mse:{:.2f}, cos:{:.2f}, pwl:{:.2f})".format(train_para['training_method'], Criteria_para['losfunc'], epoch_i, train_para['epochs'], iter_loss, iter_lmae, iter_lmse, iter_lcos, iter_lpwl)

            # ===================================== Catch wrong setup ===================================== # 
            else:
                print("The defined training method is wrong! ")
                exit()


        if train_para['training_method'] == 'SSL' or train_para['training_method'] == 'SSL_continue':
            train_avg_acc /= len_trainloader
            train_avg_eff /= len_trainloader
            train_avg_unifo /= len_trainloader
            train_avg_err /= len_trainloader
            train_avg_npcc /= len_trainloader
            train_avg_percp /= len_trainloader
            train_avg_percp_freq /= len_trainloader
            train_avg_percpfg /= len_trainloader
            train_avg_psnr /= len_trainloader
            train_avg_ssim /= len_trainloader
            train_avg_sse /= len_trainloader
            train_avg_gradloss /= len_trainloader
            train_avg_gradweightedloss /= len_trainloader
            train_avg_acc_list.append(train_avg_acc)
            train_avg_eff_list.append(train_avg_eff)
            train_avg_unifo_list.append(train_avg_unifo)
            train_avg_err_list.append(train_avg_err)
            train_avg_npcc_list.append(train_avg_npcc)
            train_avg_percp_list.append(train_avg_percp)
            train_avg_lpercp_freq_list.append(train_avg_percp_freq)
            train_avg_percpfg_list.append(train_avg_percpfg)
            train_avg_psnr_list.append(train_avg_psnr)
            train_avg_ssim_list.append(train_avg_ssim)
            train_avg_sse_list.append(train_avg_sse)
            train_avg_gradloss_list.append(train_avg_gradloss)
            train_avg_gradweightedloss_list.append(train_avg_gradweightedloss)
        elif train_para['training_method'] == 'SL':
            train_avg_phsmae /= len_trainloader
            train_avg_phsmse /= len_trainloader
            train_avg_phscos /= len_trainloader
            train_avg_phspwl /= len_trainloader
            train_avg_err /= len_trainloader
            train_avg_phsmae_list.append(train_avg_phsmae)
            train_avg_phsmse_list.append(train_avg_phsmse)
            train_avg_phscos_list.append(train_avg_phscos)
            train_avg_phspwl_list.append(train_avg_phspwl)
            
            

        # Validate neural networks for each certain epochs
        # Save visualization and trr (target_image, retrieved phase, and recosntructed image) results 
        # Save the best, each i epochs and final neural network parameters by .pth files
        if epoch_i % train_para['validate_eachi'] == 0 or epoch_i in train_para['vidualize_specifici']:
            val_epoch_i_list.append(epoch_i)
            val_avg_lphsmae, val_avg_lampmse = 0.0, 0.0
            val_avg_lacc, val_avg_leff, val_avg_lunifo = 0.0, 0.0, 0.0
            val_avg_lnpcc, val_avg_lpercp, val_avg_lpercp_freq, val_avg_lpercpfg = 0.0, 0.0, 0.0, 0.0
            val_avg_lpsnr, val_avg_lssim = 0.0, 0.0
            val_avg_lsse = 0.0
            val_avg_gradloss, val_avg_gradweightedloss = 0.0, 0.0
            val_avg_err = 0.0
            val_avg_macc, val_avg_mpc, val_avg_mmse, val_avg_mssim, val_avg_meffmean, val_avg_meffratio, val_avg_munifo, val_avg_msd, val_avg_mpsnr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            current_net.eval()
            with torch.no_grad():
                val_bar = tqdm(valid_loader)
                for iter_i, data_i in enumerate(val_bar):
                    if train_para['training_method'] == 'OffsetLearning':
                        val_AePs_Input, val_Offset_Label = data_i
                        val_AePs_Input, val_Offset_Label = val_AePs_Input.to(train_para['device']), val_Offset_Label.to(train_para['device'])
                        # val_Ae_Input, val_Ps_Input = val_AePs_Input[:,0,:,:], val_AePs_Input[:,1,:,:]
                        val_Ae_Input, val_Ps_Input = val_AePs_Input[:,0,:,:].unsqueeze(1), val_AePs_Input[:,1,:,:].unsqueeze(1)
                        # assert (torch.abs(torch.sum(val_AePs_Input[:,0,:,:]**2, dim=(-1,-2))-(PAT_para['pixel_num_in_x']*PAT_para['pixel_num_in_y'])) <= 1e-2).all() and val_AePs_Input[:,0,:,:].min() >= 0.0, "The total energy of Ae is not 2500"
                        assert val_AePs_Input[:,1,:,:].max() <= 1.0 and val_AePs_Input[:,1,:,:].min() >= 0.0, "The Ps is out of range [0,1]"
                        assert val_Offset_Label.max() <= 1.0 and val_Offset_Label.min() >= -1.0, "The Ps is out of range [-1,1]"
                        # Feed AePs to the network
                        optimizer.zero_grad()
                        val_Offset_Output = current_net(val_AePs_Input)
                        val_PredPhsHolo_output = val_Ps_Input + val_Offset_Output
                        val_TargetAmpHolo_input = val_Ae_Input / (torch.max(torch.max(torch.max(val_Ae_Input, dim=-1)[0], dim=-1)[0], dim=-1)[0]).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                        # val_TargetAmpHolo_input = val_Ae_Input / val_Ae_Input.max()

                

                    elif train_para['training_method'] == 'SL_Datafrom2folder' or train_para['training_method'] == 'SSL_Datafrom2folder' or train_para['training_method'] == 'SL_Datafrom2folder_ResNet' or train_para['training_method'] == 'SSL_Datafrom2folder_ResNet':
                        val_TargetAmpHolo_input, val_SourcePhsHolo_label = data_i
                        val_TargetAmpHolo_input, val_SourcePhsHolo_label = val_TargetAmpHolo_input.to(train_para['device']), val_SourcePhsHolo_label.to(train_para['device'])
                        # val_TargetAmpHolo_input, _ = data_i
                        # val_TargetAmpHolo_input = val_TargetAmpHolo_input.to(train_para['device'])
                    elif train_para['training_method'] == 'SL' or train_para['training_method'] == 'SL_NIPS':
                        val_TargetAmpHolo_input, val_SourcePhsHolo_label = data_i
                        val_TargetAmpHolo_input, val_SourcePhsHolo_label = val_TargetAmpHolo_input.to(train_para['device']), val_SourcePhsHolo_label.to(train_para['device'])
                    else:
                        val_TargetAmpHolo_input = data_i.to(train_para['device'])
                    val_energy_ratio = torch.ones_like(val_TargetAmpHolo_input)*(PAT_para['pixel_num_in_x']*PAT_para['pixel_num_in_y']) / torch.sum(val_TargetAmpHolo_input**2, dim=(-1,-2), keepdim=True)
                    # val_amplitude_ratio = torch.round(torch.sqrt(val_energy_ratio)*100)/100
                    val_amplitude_ratio = torch.sqrt(val_energy_ratio)
                    if train_para['input_scaleup']:
                        if train_para['training_method'] in ['SSL', 'SL', 'SL_NIPS', 'SSL_continue', 'SSL_downstream', 'SL_Datafrom2folder', 'SSL_Datafrom2folder', 'SL_Datafrom2folder_ResNet', 'SSL_Datafrom2folder_ResNet']:
                            val_PredPhsHolo_output = current_net(val_TargetAmpHolo_input * val_amplitude_ratio)
                            # val_PredPhsHolo_output = val_PredPhsHolo_output*0 + val_LabelPhsHolo_input
                        elif train_para['training_method'] in ['SSL_additionC_NFP']:
                            assert (torch.cat(((val_TargetAmpHolo_input * val_amplitude_ratio), val_amplitude_ratio), 1)).size() == torch.Size([train_para['batch_size'], 2, PAT_para['pixel_num_in_x'], PAT_para['pixel_num_in_y']]), "The tensor cat is wrong, please check it  "
                            val_PredPhsHolo_output = current_net(torch.cat(((val_TargetAmpHolo_input * val_amplitude_ratio), val_amplitude_ratio), 1))
                        else:
                            print("{} is not in the consideration in implemented codes, please re-specify! ".format(train_para['training_method']))
                            exit()
                    elif train_para['training_method'] != 'OffsetLearning':
                        val_PredPhsHolo_output = current_net(val_TargetAmpHolo_input)
                    else:
                        val_PredPhsHolo_output = current_net(val_TargetAmpHolo_input)
                    if train_para['training_method'] == 'SL_Datafrom2folder_ResNet' or train_para['training_method'] == 'SSL_Datafrom2folder_ResNet':
                        val_PredPhsHolo_output = val_PredPhsHolo_output.reshape((-1,1,PAT_para['pixel_num_in_x'],PAT_para['pixel_num_in_y']))
                    val_UniformAmpHolo = torch.ones_like(val_PredPhsHolo_output)
                    if train_para['output_process'][:12] == 'sigmoid_0to1':
                        assert val_PredPhsHolo_output.max() <= 1.0 and val_PredPhsHolo_output.min() >= 0.0, "The val_PredPhsHolo_output is out of range [0,1] defined by train_para['output_process']: {}".format(train_para['output_process'])
                        val_PredPhsHolo_output_rearrange = val_PredPhsHolo_output*(2*torch.pi)
                        val_ReconAmpHolo, val_ReconPhsHolo = ASM(d = propagation_para['propagation_Dist'], PhsHolo = val_PredPhsHolo_output_rearrange, AmpHolo = val_UniformAmpHolo, fs=propagation_para['sample frequency'], BatchSize=train_para['batch_size'])
                    elif train_para['output_process'] == 'tanh_minus1to1' and train_para['training_method'] != 'OffsetLearning':
                        assert val_PredPhsHolo_output.max() <= 1.0 and val_PredPhsHolo_output.min() >= -1.0, "The val_PredPhsHolo_output is out of range [-1,1] defined by train_para['output_process']: {}".format(train_para['output_process'])
                        val_PredPhsHolo_output_rearrange = (val_PredPhsHolo_output*torch.pi) - torch.floor(val_PredPhsHolo_output/2) * (2*torch.pi) 
                        val_ReconAmpHolo, val_ReconPhsHolo = ASM(d = propagation_para['propagation_Dist'], PhsHolo = val_PredPhsHolo_output_rearrange, AmpHolo = val_UniformAmpHolo, fs=propagation_para['sample frequency'], BatchSize=train_para['batch_size'])
                    elif train_para['training_method'] == 'OffsetLearning':
                        # assert val_PredPhsHolo_output.max() <= 1.0 and val_PredPhsHolo_output.min() >= 0.0, "The val_PredPhsHolo_output is out of range [0,1] defined by train_para['output_process']: {}".format(train_para['output_process'])
                        if val_PredPhsHolo_output.max() > 1.0 or val_PredPhsHolo_output.min() < 0.0:
                            val_PredPhsHolo_output = torch.clamp(val_PredPhsHolo_output, min=0.0, max=1.0)
                            with open('./valPredPhsHoloCheck.txt', 'a') as file0:
                                print("val_PredPhsHolo max {:.4f}, min {:.4f}".format(val_PredPhsHolo_output.max(), val_PredPhsHolo_output.min()), file=file0)
                        val_PredPhsHolo_output_rearrange = val_PredPhsHolo_output*(2*torch.pi)
                        val_ReconAmpHolo, val_ReconPhsHolo = ASM(d = propagation_para['propagation_Dist'], PhsHolo = val_PredPhsHolo_output_rearrange, AmpHolo = val_UniformAmpHolo, fs=propagation_para['sample frequency'], BatchSize=train_para['batch_size'])
                    val_ReconAmpHolo_nmlzd = normalize_amp(target_img=val_TargetAmpHolo_input, propagated_pressure=val_ReconAmpHolo, normalize_option=propagation_para['normalize option'])
                    val_ReconPhsHolo_rearrange = val_ReconPhsHolo - torch.floor(val_ReconPhsHolo/(2*torch.pi)) * (2*torch.pi)

                    if iter_i == 0 and epoch_i in train_para['vidualize_specifici']:
                        print("At epoch {}, machince is validating on the image......".format(epoch_i+1))
                        visualize(target_img=val_TargetAmpHolo_input,retrieved_phase=val_PredPhsHolo_output_rearrange, propagated_pressure=val_ReconAmpHolo, propagated_phase=val_ReconPhsHolo_rearrange, normalize_option=propagation_para['normalize option'], bs_index=0, current_epoch=str(epoch_i), save=True, save_path=savepath_para['visualize_basepath'])

                    if train_para['training_method'] in ['SL']:
                        val_iter_lphsmae = criter_Lmae(val_SourcePhsHolo_label, val_PredPhsHolo_output)
                        val_iter_lphsmse = criter_Lmse(val_SourcePhsHolo_label, val_PredPhsHolo_output)
                        val_iter_lphscos = criter_Lcos(val_SourcePhsHolo_label, val_PredPhsHolo_output)
                        val_iter_lphspwl = criter_Lpwl(val_SourcePhsHolo_label, val_PredPhsHolo_output)
                    val_iter_lampmse = criter_Lmse(val_TargetAmpHolo_input, val_ReconAmpHolo_nmlzd)
                    val_iter_lacc = criter_Lacc(y_true=val_TargetAmpHolo_input, y_pred=val_ReconAmpHolo)
                    val_iter_leff = criter_Leff(y_true=val_TargetAmpHolo_input, y_pred=val_ReconAmpHolo_nmlzd)
                    val_iter_lunifo = criter_Lunifo(y_true=val_TargetAmpHolo_input, y_pred=val_ReconAmpHolo)
                    val_iter_lnpcc = criter_Lnpcc(y_true=val_TargetAmpHolo_input, y_pred=val_ReconAmpHolo)
                    val_iter_lpercp = criter_Lpercp(y_true=val_TargetAmpHolo_input*val_amplitude_ratio, y_pred=val_ReconAmpHolo)
                    val_iter_lpercp_freq = criter_Lpercp_freq(y_true=torch.fft.fftshift(torch.fft.fftn((val_TargetAmpHolo_input*val_amplitude_ratio).to(torch.complex128))), y_pred=torch.fft.fftshift(torch.fft.fftn(val_ReconAmpHolo.to(torch.complex128))))
                    val_iter_lpercpfg = criter_Lpercpfg(y_true=val_TargetAmpHolo_input, y_true_scaleup=val_TargetAmpHolo_input*val_amplitude_ratio, y_pred=val_ReconAmpHolo)
                    val_iter_lpsnr = criter_Lpsnr(y_true=val_TargetAmpHolo_input, y_pred=val_ReconAmpHolo_nmlzd)
                    val_iter_lssim = criter_Lssim(y_true=val_TargetAmpHolo_input, y_pred=val_ReconAmpHolo_nmlzd)
                    val_iter_lsse = criter_LSumSquareMAE(y_true=val_TargetAmpHolo_input, y_pred=val_ReconAmpHolo_nmlzd)
                    val_iter_gradloss = criter_GradLoss(y_true=val_TargetAmpHolo_input, y_pred=val_ReconAmpHolo_nmlzd)
                    val_iter_gradweightedloss = criter_GradWeightedLoss(y_true=val_TargetAmpHolo_input, y_pred=val_ReconAmpHolo_nmlzd)

                    if Criteria_para['losfunc'] == 'acc_eff':
                        val_iter_loss = val_iter_lacc + Criteria_para['lamda'][0] * val_iter_leff
                    elif Criteria_para['losfunc'] == 'acc_eff_unifo':
                        val_iter_loss = val_iter_lacc + Criteria_para['lamda'][0] * val_iter_leff + Criteria_para['lamda'][1] * val_iter_lunifo
                    elif Criteria_para['losfunc'] == 'acc_eff_unifo_with_weightdecay':
                        val_iter_loss = Criteria_para['lamda'][0] * val_iter_lacc + Criteria_para['lamda'][1] * val_iter_leff + Criteria_para['lamda'][2] * val_iter_lunifo
                    elif Criteria_para['losfunc'] == 'percp_psnr_percpfg_acc_eff_unifo_with_weightdecay':
                        val_iter_loss = Criteria_para['lamda'][0] * val_iter_lpercp + Criteria_para['lamda'][1] * val_iter_lpsnr + Criteria_para['lamda'][2] * val_iter_lpercpfg + Criteria_para['lamda'][3] * val_iter_lacc + Criteria_para['lamda'][4] * val_iter_leff + Criteria_para['lamda'][5] * val_iter_lunifo
                    elif Criteria_para['losfunc'] == 'npcc_percp':
                        val_iter_loss = val_iter_lnpcc + Criteria_para['lamda'][0] * val_iter_lpercp
                    elif Criteria_para['losfunc'] == 'percp_acc_eff_unifo_npcc':
                        val_iter_loss = Criteria_para['lamda'][0] * val_iter_lpercp + Criteria_para['lamda'][1] * val_iter_lacc + Criteria_para['lamda'][2] * val_iter_leff + Criteria_para['lamda'][3] * val_iter_lunifo + Criteria_para['lamda'][4] * val_iter_lnpcc 
                    elif Criteria_para['losfunc'] == 'percp':
                        val_iter_loss = val_iter_lpercp
                    elif Criteria_para['losfunc'] == 'sse':
                        val_iter_loss = val_iter_lsse
                    elif Criteria_para['losfunc'] == 'percp_acc':
                        assert len(Criteria_para['lamda']) == 1, "The weights of the total loss are not compile to the loss design! "
                        val_iter_loss = val_iter_lpercp + Criteria_para['lamda'][0] * val_iter_lacc
                    elif Criteria_para['losfunc'] == 'percp_psnr':
                        assert len(Criteria_para['lamda']) == 2, "The weights of the total loss are not compile to the loss design! "
                        val_iter_loss = Criteria_para['lamda'][0] * val_iter_lpercp + Criteria_para['lamda'][1] * val_iter_lpsnr
                    elif Criteria_para['losfunc'] == 'percp_psnr_percpfg':
                        assert len(Criteria_para['lamda']) == 3, "The weights of the total loss are not compile to the loss design! "
                        val_iter_loss = Criteria_para['lamda'][0] * val_iter_lpercp + Criteria_para['lamda'][1] * val_iter_lpsnr + Criteria_para['lamda'][2] * val_iter_lpercpfg
                    elif Criteria_para['losfunc'] == 'percp_psnr_ssim':
                        assert len(Criteria_para['lamda']) == 3, "The weights of the total loss are not compile to the loss design! "
                        val_iter_loss = Criteria_para['lamda'][0] * val_iter_lpercp + Criteria_para['lamda'][1] * val_iter_lpsnr + Criteria_para['lamda'][2] * val_iter_lssim
                    elif Criteria_para['losfunc'] in ['percp_acc_and_Phse_mae', 'percp_acc_and_Phse_mse', 'percp_acc_and_Phse_cos', 'percp_acc_and_Phse_pwl']:
                        assert len(Criteria_para['lamda']) == 3, "The weights of the total loss are not compile to the loss design! "
                        val_iter_loss = val_iter_lpercp + Criteria_para['lamda'][0] * val_iter_lacc
                    elif Criteria_para['losfunc'] in ['percp_and_Phse_mae', 'percp_and_Phse_mse', 'percp_and_Phse_cos', 'percp_and_Phse_pwl']:
                        assert len(Criteria_para['lamda']) == 2, "The weights of the total loss are not compile to the loss design! "
                        val_iter_loss = val_iter_lpercp
                    elif Criteria_para['losfunc'] in ['Phse_mae', 'Phse_mse', 'Phse_cos', 'Phse_pwl']:
                        assert len(Criteria_para['lamda']) == 0, "The weights of the total loss are not compile to the loss design! "
                        # val_iter_loss = 0
                        val_iter_loss = val_iter_lphsmae
                    elif Criteria_para['losfunc'] == 'percp_psnr_ssim':
                        assert len(Criteria_para['lamda']) == 2, "The weights of the total loss are not compile to the loss design! "
                        val_iter_loss = Criteria_para['lamda'][0] * val_iter_lpercp + Criteria_para['lamda'][1] * val_iter_lpsnr + Criteria_para['lamda'][2] * val_iter_lssim
                    elif Criteria_para['losfunc'] == 'percp_percpfreq_ssim':
                        assert len(Criteria_para['lamda']) == 3, "The weights of the total loss are not compile to the loss design! "
                        val_iter_loss = Criteria_para['lamda'][0] * val_iter_lpercp + Criteria_para['lamda'][1] * val_iter_lpercp_freq + Criteria_para['lamda'][2] * val_iter_lssim
                    elif Criteria_para['losfunc'] == 'percp_percpfreq_psnr_ssim':
                        assert len(Criteria_para['lamda']) == 4, "The weights of the total loss are not compile to the loss design! "
                        val_iter_loss = Criteria_para['lamda'][0] * val_iter_lpercp + Criteria_para['lamda'][1] * val_iter_lpercp_freq + Criteria_para['lamda'][2] * val_iter_lpsnr + Criteria_para['lamda'][3] * val_iter_lssim
                    elif Criteria_para['losfunc'] == 'percp_percpfreq_psnr_ssim_grad':
                        val_iter_loss = Criteria_para['lamda'][0] * val_iter_lpercp + Criteria_para['lamda'][1] * val_iter_lpercp_freq + Criteria_para['lamda'][2] * val_iter_lpsnr + Criteria_para['lamda'][3] * val_iter_lssim + Criteria_para['lamda'][4] * val_iter_gradloss
                    elif Criteria_para['losfunc'] == 'percp_percpfreq_psnr_ssim_gradweighted':
                        val_iter_loss = Criteria_para['lamda'][0] * val_iter_lpercp + Criteria_para['lamda'][1] * val_iter_lpercp_freq + Criteria_para['lamda'][2] * val_iter_lpsnr + Criteria_para['lamda'][3] * val_iter_lssim + Criteria_para['lamda'][4] * val_iter_gradweightedloss
                    else:
                        print("Please define the loss function for you Deep learning! ")
                        exit()
                    iter_M_acc, iter_M_pc, iter_M_mse, iter_M_ssim, iter_M_effmean, iter_M_effratio, iter_M_unifo, iter_M_sd, iter_M_psnr = assess(target_img=val_TargetAmpHolo_input,propagated_pressure=val_ReconAmpHolo,propagated_pressure_nmlzd=val_ReconAmpHolo_nmlzd)
                    
                    if train_para['training_method'] in ['SL']:
                        val_avg_lphsmae += val_iter_lphsmae.item()
                    val_avg_lampmse += val_iter_lampmse.item()
                    val_avg_lacc += val_iter_lacc.item()
                    val_avg_leff += val_iter_leff.item()
                    val_avg_lunifo += val_iter_lunifo.item()
                    val_avg_lnpcc += val_iter_lnpcc.item()
                    val_avg_lpercp += val_iter_lpercp.item()
                    val_avg_lpercp_freq += val_iter_lpercp_freq.item()
                    val_avg_lpercpfg += val_iter_lpercpfg.item()
                    val_avg_lpsnr += val_iter_lpsnr.item()
                    val_avg_lssim += val_iter_lssim.item()
                    val_avg_lsse += val_iter_lsse.item()
                    val_avg_gradloss += val_iter_gradloss.item()
                    val_avg_gradweightedloss += val_iter_gradweightedloss.item()
                    val_avg_err += val_iter_loss.item()
                    val_avg_macc += iter_M_acc
                    val_avg_mpc += iter_M_pc
                    val_avg_mmse += iter_M_mse
                    val_avg_mssim += iter_M_ssim
                    val_avg_meffmean += iter_M_effmean
                    val_avg_meffratio += iter_M_effratio
                    val_avg_munifo += iter_M_unifo
                    val_avg_msd += iter_M_sd
                    val_avg_mpsnr += iter_M_psnr
                    val_bar.desc = "Valid epoch [{}], {} loss: {}, psnr: {}".format(epoch_i, Criteria_para['losfunc'], val_iter_loss, iter_M_psnr)

                val_avg_lphsmae /= len_validloader
                val_avg_lampmse /= len_validloader
                val_avg_lacc /= len_validloader
                val_avg_leff /= len_validloader
                val_avg_lunifo /= len_validloader
                val_avg_lnpcc /= len_validloader
                val_avg_lpercp /= len_validloader
                val_avg_lpercp_freq /= len_validloader
                val_avg_lpercpfg /= len_validloader
                val_avg_lpsnr /= len_validloader
                val_avg_lssim /= len_validloader
                val_avg_lsse /= len_validloader
                val_avg_gradloss /= len_validloader
                val_avg_gradweightedloss /= len_validloader
                val_avg_err /= len_validloader
                val_avg_macc /= len_validloader
                val_avg_mpc /= len_validloader
                val_avg_mmse /= len_validloader
                val_avg_mssim /= len_validloader
                val_avg_meffmean /= len_validloader
                val_avg_meffratio /= len_validloader
                val_avg_munifo /= len_validloader
                val_avg_msd /= len_validloader
                val_avg_mpsnr /= len_validloader
                
                val_avg_lphsmae_list.append(val_avg_lphsmae)
                val_avg_lampmse_list.append(val_avg_lampmse)
                val_avg_lacc_list.append(val_avg_lacc)
                val_avg_leff_list.append(val_avg_leff)
                val_avg_lunifo_list.append(val_avg_lunifo)
                val_avg_lnpcc_list.append(val_avg_lnpcc)
                val_avg_lpercp_list.append(val_avg_lpercp)
                val_avg_lpercp_freq_list.append(val_avg_lpercp_freq)
                val_avg_lpercpfg_list.append(val_avg_lpercpfg)
                val_avg_lpsnr_list.append(val_avg_lpsnr)
                val_avg_lssim_list.append(val_avg_lssim)
                val_avg_lsse_list.append(val_avg_lsse)
                val_avg_gradloss_list.append(val_avg_gradloss)
                val_avg_gradweightedloss_list.append(val_avg_gradweightedloss)
                val_avg_err_list.append(val_avg_err)
                val_avg_macc_list.append(val_avg_macc)
                val_avg_mpc_list.append(val_avg_mpc)
                val_avg_mmse_list.append(val_avg_mmse)
                val_avg_mssim_list.append(val_avg_mssim)
                val_avg_meffmean_list.append(val_avg_meffmean)
                val_avg_meffratio_list.append(val_avg_meffratio)
                val_avg_munifo_list.append(val_avg_munifo)
                val_avg_msd_list.append(val_avg_msd)
                val_avg_mpsnr_list.append(val_avg_mpsnr)

                print("Epoch [{}], {}. avgtrainlpercp: {:.2f}, avgtrainlpercp_freq: {:.2f}, avgtrainpsnr: {:.2f}, avgtrainlampmse: {:.2f}, avgtrainphsmae: {:.2f}; avgvallpercp: {:.2f}, avgvalpercp_freq: {:.2f}, avgvalpsnr: {:.2f}, avgvallampmse: {:.2f}, avgvalphsmae: {:.2f}, avggradloss: {:.2f}, avggradweightedloss: {:.2f}".format(epoch_i, Criteria_para['losfunc'], train_avg_percp, train_avg_percp_freq, -train_avg_psnr, train_avg_ampmse, train_avg_phsmae, val_avg_lpercp, val_avg_lpercp_freq, val_avg_mpsnr, val_avg_lampmse, val_avg_lphsmae, val_avg_gradloss, val_avg_gradweightedloss))

            if val_avg_mpsnr > best_valid_psnr:
                train_para['net'] = current_net # The train_para['net'] maintains the best network parameters
                torch.save(train_para['net'].state_dict(), savepath_para['model_pth_basepath'][:-4] + '/Best' + savepath_para['model_pth_basepath'][-4:])
                best_valid_psnr = val_avg_mpsnr
            

        # if epoch_i % train_para['save_pth_eachi'] == 0:
        #     torch.save(current_net.state_dict(), savepath_para['model_pth_basepath'][:-4] + '/' + str(epoch_i) + savepath_para['model_pth_basepath'][-4:])
        if epoch_i == (train_para['epochs']-1):
            torch.save(current_net.state_dict(), savepath_para['model_pth_basepath'][:-4] + '/Final' + savepath_para['model_pth_basepath'][-4:])

        if (epoch_i+1) % train_para['plotcurve_eachi'] == 0: # each train_para['plotcurve_eachi'] epochs, plot and visualize terminated results
            list_packed_data = [train_avg_acc_list, train_avg_eff_list, train_avg_unifo_list, train_avg_npcc_list, train_avg_percp_list, train_avg_lpercp_freq_list, train_avg_psnr_list, train_avg_ssim_list, train_avg_err_list, val_avg_lacc_list, val_avg_leff_list, val_avg_lunifo_list, val_avg_lnpcc_list, val_avg_lpercp_list, val_avg_lpsnr_list, val_avg_lssim_list, val_avg_err_list, val_avg_macc_list, val_avg_mpc_list,val_avg_mmse_list, val_avg_mssim_list, val_avg_meffmean_list, val_avg_meffratio_list, val_avg_munifo_list, val_avg_msd_list, val_avg_mpsnr_list]
            list_packed_name = ['train_acc', 'train_eff', 'train_unifo', 'train_npcc', 'train_percp', 'train_percp_freq', 'train_psnr', 'train_ssim', 'train_loss', 'val_lacc', 'val_leff', 'val_lunifo', 'val_lnpcc', 'val_lpercp', 'val_lpsnr', 'val_lssim', 'val_loss', 'val_macc', 'val_mpc', 'val_mmse', 'val_mssim', 'val_meffmean', 'val_meffratio', 'val_munifo', 'val_msd', 'val_mpsnr']
            assert len(list_packed_data) == len(list_packed_name), "The length of list_packed_data and that of list_packed_name are not the same! "
            plot_curves(list_packed_data, list_packed_name, savepath_para['curves_basepath'], train_para['epochs'], str(epoch_i), fontsize=fontsize)
            for index, data, name in zip(range(len(list_packed_data)), list_packed_data, list_packed_name):
                plot_single_curve(index=index, data=data, name=name, savepath=savepath_para['curves_basepath'], total_epochs=train_para['epochs'], current_epochs=str(epoch_i), fontsize=fontsize)

            with open(savepath_para['testmetric_basepath'], 'a') as file0:
                print("============================  {} epoch  ====================================".format(epoch_i+1), file=file0)
                print("The test (validation dataset) image number is {}".format(len_validloader*train_para['batch_size']), file=file0)
                print("On thes test (validation dataset) image:\n The best avg Accuracy (ACC) = {:.4f}, at {} epoch ".format(max(val_avg_macc_list), val_epoch_i_list[val_avg_macc_list.index(max(val_avg_macc_list))]+1), file=file0)
                print("The best Pearson's correlation coefficient (PC) = {:.4f}, at {} epoch ".format(max(val_avg_mpc_list), val_epoch_i_list[val_avg_mpc_list.index(max(val_avg_mpc_list))]+1), file=file0)
                print("The best NMSE = {:.4f}, at {} epoch ".format(max(val_avg_mmse_list), val_epoch_i_list[val_avg_mmse_list.index(max(val_avg_mmse_list))]+1), file=file0)
                print("The best Structure similarity index metric (SSIM) = {:.4f}, at {} epoch ".format(max(val_avg_mssim_list), val_epoch_i_list[val_avg_mssim_list.index(max(val_avg_mssim_list))]+1), file=file0)
                print("Efficiency (EFF) mean = {:.4f}, at {} epoch ".format(max(val_avg_meffmean_list), val_epoch_i_list[val_avg_meffmean_list.index(max(val_avg_meffmean_list))]+1), file=file0)
                print("Efficiency (EFF) ratio = {:.4f}, at {} epoch ".format(max(val_avg_meffratio_list), val_epoch_i_list[val_avg_meffratio_list.index(max(val_avg_meffratio_list))]+1), file=file0)
                print("Uniformity = {:.4f}, at {} epoch ".format(max(val_avg_munifo_list), val_epoch_i_list[val_avg_munifo_list.index(max(val_avg_munifo_list))]+1), file=file0)
                print("Standard deviation = {:.4f}, at {} epoch ".format(max(val_avg_msd_list), val_epoch_i_list[val_avg_msd_list.index(max(val_avg_msd_list))]+1), file=file0)
                print("Peak signal noise ratio (PSNR) = {:.4f}, at {} epoch ".format(max(val_avg_mpsnr_list), val_epoch_i_list[val_avg_mpsnr_list.index(max(val_avg_mpsnr_list))]+1), file=file0)
                print("==============================================================================\n\n", file=file0)
        
        if 'weightdecay' in Criteria_para['losfunc'] and len(train_avg_err_list) > train_para['weights_decay_condtion'][0]:
            # print("~~~~~~~~~~~~ Let's check the stability of loss ~~~~~~~~~~~~~~")

            # stable_count = 0
            # for loss_stable_times_record in range(train_para['weights_decay_condtion'][0]):
            #     if np.abs(train_avg_err_list[-1-loss_stable_times_record] - train_avg_err_list[-2-loss_stable_times_record]) <= train_para['weights_decay_condtion'][1]:
            #         stable_count += 1
            # if stable_count == train_para['weights_decay_condtion'][0]:
            #     loss_weigtsdecay_epoch.append(epoch_i) # record the weight decay epoch_i
            #     Criteria_para['lamda'][train_para['weight_decay_index']] += 0.5
            #     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            #     print("At the {}-th epoch, the loss has decreased less than or equal to {} for {} times ~~~".format(epoch_i, train_para['weights_decay_condtion'][1], train_para['weights_decay_condtion'][0]))
            #     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            
            train_avg_err_list_array = np.array(train_avg_err_list[-(train_para['weights_decay_condtion'][0]+1):])
            loss_change = train_avg_err_list_array[1:] - train_avg_err_list_array[:-1]
            assert len(loss_change) == train_para['weights_decay_condtion'][0], "The length of loss_change is wrong! "
            if (loss_change <= train_para['weights_decay_condtion'][1]).all() == True:
                loss_weigtsdecay_epoch.append(epoch_i) # record the weight decay epoch_i
                Criteria_para['lamda'][train_para['weight_decay_index']] += train_para['weight_decay_amplifier']
                print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print("At the {}-th epoch,".format(epoch_i))
                print("The weights of {}-th item in the loss function [{}] increases by amplifier {}".format(train_para['weight_decay_index'], Criteria_para['losfunc'], train_para['weight_decay_amplifier']))
                print("The loss has decreased less than or equal to {} for {} times ~~~".format(train_para['weights_decay_condtion'][1], train_para['weights_decay_condtion'][0]))
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

            # print("~~~~~~~~~~ Finish checking the stability of loss ~~~~~~~~~~~~")

    list_packed_data = [train_avg_acc_list, train_avg_eff_list, train_avg_unifo_list, train_avg_npcc_list, train_avg_percp_list, train_avg_lpercp_freq_list, train_avg_psnr_list, train_avg_ssim_list, train_avg_err_list, val_avg_lacc_list, val_avg_leff_list, val_avg_lunifo_list, val_avg_lnpcc_list, val_avg_lpercp_list, val_avg_lpsnr_list, val_avg_lssim_list, val_avg_err_list, val_avg_macc_list, val_avg_mpc_list,val_avg_mmse_list, val_avg_mssim_list, val_avg_meffmean_list, val_avg_meffratio_list, val_avg_munifo_list, val_avg_msd_list, val_avg_mpsnr_list]
    list_packed_name = ['train_acc', 'train_eff', 'train_unifo', 'train_npcc', 'train_percp', 'train_percp_freq', 'train_psnr', 'train_ssim', 'train_loss', 'val_lacc', 'val_leff', 'val_lunifo', 'val_lnpcc', 'val_lpercp', 'val_lpsnr', 'val_lssim', 'val_loss', 'val_macc', 'val_mpc', 'val_mmse', 'val_mssim', 'val_meffmean', 'val_meffratio', 'val_munifo', 'val_msd', 'val_mpsnr']
    assert len(list_packed_data) == len(list_packed_name), "The length of list_packed_data and that of list_packed_name are not the same! "
    plot_curves(list_packed_data, list_packed_name, savepath_para['curves_basepath'], train_para['epochs'], 'Final', fontsize=fontsize)
    for index, data, name in zip(range(len(list_packed_data)), list_packed_data, list_packed_name):
        plot_single_curve(index=index, data=data, name=name, savepath=savepath_para['curves_basepath'], total_epochs=train_para['epochs'], current_epochs='Final', fontsize=fontsize)

    with open(savepath_para['testmetric_basepath'], 'a') as file0:
        print("+++++++++++++++++++++++++++++++++ Validation Set +++++++++++++++++++++++++++++++++", file=file0)
        print("\n\n==============================================================================", file=file0)
        print("The test (validation dataset) image number is {}".format(len_validloader*train_para['batch_size']), file=file0)
        print("On thes test (validation dataset) image:\n The best avg Accuracy (ACC) = {:.4f}, at {} epoch ".format(max(val_avg_macc_list), val_epoch_i_list[val_avg_macc_list.index(max(val_avg_macc_list))]+1), file=file0)
        print("The best Pearson's correlation coefficient (PC) = {:.4f}, at {} epoch ".format(max(val_avg_mpc_list), val_epoch_i_list[val_avg_mpc_list.index(max(val_avg_mpc_list))]+1), file=file0)
        print("The best NMSE = {:.4f}, at {} epoch ".format(max(val_avg_mmse_list), val_epoch_i_list[val_avg_mmse_list.index(max(val_avg_mmse_list))]+1), file=file0)
        print("The best Structure similarity index metric (SSIM) = {:.4f}, at {} epoch ".format(max(val_avg_mssim_list), val_epoch_i_list[val_avg_mssim_list.index(max(val_avg_mssim_list))]+1), file=file0)
        print("Efficiency (EFF) mean = {:.4f}, at {} epoch ".format(max(val_avg_meffmean_list), val_epoch_i_list[val_avg_meffmean_list.index(max(val_avg_meffmean_list))]+1), file=file0)
        print("Efficiency (EFF) ratio = {:.4f}, at {} epoch ".format(max(val_avg_meffratio_list), val_epoch_i_list[val_avg_meffratio_list.index(max(val_avg_meffratio_list))]+1), file=file0)
        print("Uniformity = {:.4f}, at {} epoch ".format(max(val_avg_munifo_list), val_epoch_i_list[val_avg_munifo_list.index(max(val_avg_munifo_list))]+1), file=file0)
        print("Standard deviation = {:.4f}, at {} epoch ".format(max(val_avg_msd_list), val_epoch_i_list[val_avg_msd_list.index(max(val_avg_msd_list))]+1), file=file0)
        print("Peak signal noise ratio (PSNR) = {:.4f}, at {} epoch ".format(max(val_avg_mpsnr_list), val_epoch_i_list[val_avg_mpsnr_list.index(max(val_avg_mpsnr_list))]+1), file=file0)
        print("==============================================================================\n\n", file=file0)
    
    # print(len(train_avg_psnr_list), len(train_avg_err_list), len(val_avg_err_list))
    # dataframe = pd.DataFrame({'train psnr':[-1*train_avg_psnr_list[i] for i in range(len(train_avg_psnr_list))], 'train loss':train_avg_err_list, 'validation loss':val_avg_err_list})
    print(len(train_avg_psnr_list), len(train_avg_err_list))
    dataframe = pd.DataFrame({'train psnr':[-1*train_avg_psnr_list[i] for i in range(len(train_avg_psnr_list))], 'train loss':train_avg_err_list})
    dataframe.to_csv(savepath_para['curves_basepath'][:-4] + '/Final_Main_Metrics_train.csv',index=False,sep=',')
    print(len(val_avg_mpsnr_list), len(val_avg_err_list))
    dataframe = pd.DataFrame({'train psnr':val_avg_mpsnr_list, 'train loss':val_avg_err_list})
    dataframe.to_csv(savepath_para['curves_basepath'][:-4] + '/Final_Main_Metrics_val.csv',index=False,sep=',')
    print("The final main metrics have been saved to folder named {}".format(savepath_para['curves_basepath'][:-4] + '/Final_Main_Metrics_train(or val).csv'))

    # Save each loss item for the study of how loss function affects the neural network training
    if train_para['training_method'] == 'SL_Datafrom2folder' or train_para['training_method'] == 'SSL_Datafrom2folder' or train_para['training_method'] == 'SL_Datafrom2folder_ResNet' or train_para['training_method'] == 'SSL_Datafrom2folder_ResNet':
        print(len(train_avg_phsmae_list), len(train_avg_phsmse_list), len(train_avg_phscos_list), len(train_avg_phspwl_list))
        print(len(train_avg_acc_list), len(train_avg_eff_list), len(train_avg_unifo_list), len(train_avg_npcc_list), len(train_avg_percp_list), len(train_avg_percpfg_list), len(train_avg_psnr_list), len(train_avg_ssim_list))
        print(len(train_avg_err_list))
        print()
        print(len(val_avg_lphsmae_list), len(val_avg_lampmse_list), len(val_avg_mpsnr_list), len(val_avg_mssim_list), len(val_avg_macc_list), len(val_avg_meffmean_list), len(val_avg_meffratio_list))
        print(len(val_avg_err_list))
        print()
        dataframe = pd.DataFrame({'train phsmae':train_avg_phsmae_list, 'train phsmse':train_avg_phsmse_list, 'train phscos':train_avg_phscos_list, 'train phspwl':train_avg_phspwl_list, 'train acc':train_avg_acc_list, 'train eff':train_avg_eff_list, 'train unifo':train_avg_unifo_list, 'train npcc':train_avg_npcc_list, 'train percp':train_avg_percp_list, 'train percpfg':train_avg_percpfg_list, 'train psnr':[-1*train_avg_psnr_list[i] for i in range(len(train_avg_psnr_list))], 'train ssim':train_avg_ssim_list, 'train loss':train_avg_err_list})
        dataframe.to_csv(savepath_para['curves_basepath'][:-4] + '/' + train_para['training_method'] + '_Study_How_Loss_Affects_Training.csv',index=False,sep=',')
        dataframe = pd.DataFrame({'val phsmae':val_avg_lphsmae_list, 'val ampmse':val_avg_lampmse_list, 'val psnr':val_avg_mpsnr_list, 'val ssim':val_avg_mssim_list, 'val acc':val_avg_macc_list, 'val effmean':val_avg_meffmean_list, 'val effratio':val_avg_meffratio_list, 'val loss': val_avg_err_list})
        dataframe.to_csv(savepath_para['curves_basepath'][:-4] + '/' + train_para['training_method'] + '_Study_How_Loss_Affects_Validation.csv',index=False,sep=',')
        print("The final main metrics have been saved to folder named {}".format(savepath_para['curves_basepath'][:-4] + '/' + train_para['training_method'] + '_Study_How_Loss_Affects_Training.csv'))

    
    return train_para['net']