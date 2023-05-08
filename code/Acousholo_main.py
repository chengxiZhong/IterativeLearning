import time
import os
import torch
import numpy as np
import tifffile as tiff
from Acousholo_data import load_data  # for dataset
from UNet_V3_original import UNet_V3  # for network
# from UNet_V3_original_forTJ import UNet_V3  # for network
from Acousholo_trainingframework import DL_train_val, DL_train_val_MultiTaskLearning, DL_test, Test_MultiTaskLearning_Offset     # for training strategy
from Acousholo_auxiliaryfunc import Init_net_from_scratch, Init_net_from_before, Init_net_from_pretrain, EP_Collect  # for other useful auxiliary functions


Criteria_para = {
    'losfunc': 'acc_eff_unifo',
    'lamda': [0.1]
}
lr_para = {
    'lr': 1e-3,
    'lr_decrease': True,
    'decrease_method': '10', # 2_5inturns
    'decrease_times': 5,
    'decrease_at': []
}
netfreeze_para = {
    'freeze': False,
    'freeze_times': 2,
    'freeze_dict': {}
}
savepath_para = {
    'model_pth_basepath': './Results_newOrgamized/'+ time.strftime("%Y-%m-%d",time.localtime(time.time())) + '/model_pth/',
    'visualize_basepath': './Results_newOrgamized/' + time.strftime("%Y-%m-%d",time.localtime(time.time())) + '/visualize/',
    'trrData_basepath': './Results_newOrgamized/' + time.strftime("%Y-%m-%d",time.localtime(time.time())) + '/trrData/',
    'trainpara_basepath': './Results_newOrgamized/' + time.strftime("%Y-%m-%d",time.localtime(time.time())) + '/trainparams/',
    'testmetric_basepath': './Results_newOrgamized/' + time.strftime("%Y-%m-%d",time.localtime(time.time())) + '/testmetrics/',
    'curves_basepath': './Results_newOrgamized/' + time.strftime("%Y-%m-%d",time.localtime(time.time())) + '/curves/',
    'EP_basepath': '/public/home/zhongchx/Dataset_2D/ExperiencePools/' + time.strftime("%Y-%m-%d",time.localtime(time.time())) + '/',
    'Offset_basepath': '/public/home/zhongchx/Dataset_2D/MultiTaskLearning/'
}
train_para = {
    'net': UNet_V3(img_ch=1, output_ch=1, output_process='sigmoid_0to1'),
    'with_conv_7x7': bool,
    'conv_7x7_double': bool,
    'with_conv6': bool,
    'with_conv_1x1_repeat': bool,
    'with_final_fc': bool,
    'pretrained_pth_file': str,
    'fixed_pretrained_model': True,
    'output_process': 'sigmoid_0to1',
    'training_method': str,
    'dataset': '',
    'batch_size': 16,
    'epochs': 2000,
    'loss_weights_decay': False,
    'weights_decay_condtion': [],
    'weight_decay_index': int,
    'weight_decay_amplifier': 0.5,
    'l2_lambda': 0.01,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'validate_eachi': 20,
    'save_pth_eachi': 50,
    'plotcurve_eachi': 200,
    'vidualize_specifici': [1, 20, 50, 100, 200, 500, 1000],
    'input_scaleup': bool,
    'training_time': time.strftime("%Y-%m-%d",time.localtime(time.time()))
}
PAT_para = {
    'pixel_num_in_x': 50, 
    'pixel_num_in_y': 50, # 50 x 50 phased array of transducer
    'width_x': 50e-3, # [m], 50 mm = 5 cm
    'width_y': 50e-3, # [m], 50 mm = 5 cm
}
propagation_para = {
    'propagation_Dist': 30e-3, # [m] 30mm
    'sample frequency': 1/ (50e-3/50), # the sample frequency for acoustic wave propagation by ASM
    'normalize option':1 # 1 means the Maximum of propagated_pressure Normalization
}
dataloader_para = {
    'Dataset_Root_Path': '/public/home/zhongchx/Dataset_2D/', 
    # 'Subfolder': 'ExpectedAmpHolo/',
    'Subfolder': '',
    'shuffle': True,
    'nw': 0,
    'data_split_ratio': [0.8,0.1,0.1],
    'used_dataset_num': int
}
test_para = {
    'evalontestset_visual_trr_num':1
}
EP_collect_para = {
    'EP1_capacity': 1600,
    'EP2_capacity': 1600,
    'EP_update_eachi': 200,
    'EP_subfolder':'ExperiencePools/',
    'EP_iteration': 50,
    'EP_iteration_list': ['EP2','EP1']
}

# User defined parameters
# ============================ You can change parameters here ============================ #
case_index = 0
my_output_process = 'sigmoid_0to1'  # 'sigmoid_0to1', 'sigmoid_0to1_DIV2', 'sigmoid_0to1_DIV5', 'sigmoid_0to1_DIV10', 'sigmoid_0to1_DIV20', 'sigmoid_0to1_DIV42', sigmoid_0to1_DIV84', 'tanh_minus1to1'
with_conv_7x7, conv_7x7_double, with_conv6, with_conv_1x1_repeat, with_final_fc = False, False, False, False, False
training_method, input_scaleup, bs, epochs = 'SSL', True, 16, 1000 # 'SSL', 'SL', 'SSL_continue'
# SSL: 'acc_eff', [0.1]; 'acc_eff_unifo', [0.1,0.1]; 'npcc_percp', [1.0 or 1e-4];  'percp_acc_eff_unifo_npcc', [1.0, 1.0, 1.0, 1.0, 1.0 or 1e-4]; 'percp', []; 'percp_acc', [0.5]; 'percp_psnr_ssim', [1.0, 1.0, 1.0]; 'percp_psnr', [1.0, 1.0]
# SL: 'Phse_mae', []; 'Phse_mse', []; 'Phse_cos', []; 'Phse_pwl', []
loss_func, lamda_list = 'percp_psnr_ssim', [1.0, 0.2, 0.8] # 'Phse_mae', []  # 'percp_psnr_percpfg_acc_eff_unifo_with_weightdecay', [1.0, 1.0, 0.5, 1.0, 0.1, 0.0] or 'percp_psnr_ssim', [1.0, 0.2, 0.8]
l2_lambda = 0.01
if 'with_weightdecay' in loss_func:
    loss_weights_decay, weights_decay_condtion, weight_decay_index, weight_decay_amplifier = True, [20, 1e-4], -1, 0.1 # True, [20, 1e-4], -1 or True, [20, 1e-8], -1; True, [20, 1e-2], -1 or True, [20, 1e-4], -1; True, [10, 1e-2], -1; True, [5, 1e-2], -1
else:
    loss_weights_decay, weights_decay_condtion, weight_decay_index, weight_decay_amplifier = False, [], -100, 0.0
initial_lr, lr_decrease, decrease_method, decrease_times = 1e-3, False, '10', 4
freeze, freeze_time1, freeze_times2 = False, 0.8, 0.95
validate_eachi, save_pth_eachi, plotcurve_eachi, vidualize_specifici = 20, 20, epochs//5, [1, 50, 100, 200, 500, 1000, int(epochs/2), epochs] # [1, 1, 1, list or 2, 5, 20 (for codes correctness test), list] or 20, 50, 200, [1, 20, 50, 100, 200, 500, 1000]
pixel_m, pixel_n, width_m, width_n = 50, 50, 50e-3, 50e-3
propagation_dist = 30e-3 # 30e-3 or 25e-3 or 20e-3 or 10e-3 or 40e-3
for_sample_freq = 50 # 50 or 100 or 1/(0.64e-2)=156.25 [refers to half of wavelenght]
normalize_option = 3 # 1:MAximum normalization; 2:Normalization and Alignment with Ae energy; 3:Normalization and Alignment with 2
# ##
# print("Checking on sample frequency and normalize option ......")
# print("The used sample frequency is {}, while the normalize option is {}".format(for_sample_freq, normalize_option))
# ##
used_dataset_num = 60000 # T&J: 2854   17562
Dataset_Root_Path, Subfolder = './dataset/', 'DS_TNNLS/' 
shuffle, nw, data_split_ratio = True, 0, [0.8,0.1,0.1]
if training_method == 'SSL_continue':
    pretrained_file = "./Results_newOrgamized/2023-01-17/model_pth/0_SSL_percp_psnr_percpfg_acc_eff_unifo_with_weightdecay/Best.pth"
    fixed_pretrained_model = False
else:
    pretrained_file = ''
    fixed_pretrained_model = False
fontsize = 10
evalontestset_visual_trr_num = 10
# Attention: Before running 'SL' case, we fistly need to check the EP_subfolder parameter !!!!!!!!!!!!!!!!!!!!!!!!!!!
EP1_capacity, EP2_capacity, EP_update_eachi, EP_iteration, EP_subfolder = 800, 800, 100, 25, 'ExperiencePools/'+train_para['training_time']+'/'+str(case_index)+'_'+training_method+'_'+loss_func+'/EP1/' # 1600, 1600, 200, 50; 800, 800, 100, 25; '/ExperiencePools/**time**/**save_note**/EP1 or EP2' 
Iteration_list = ['EP2', 'EP1']
# ======================================================================================== #

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if training_method == 'SSL_continue':
    net = Init_net_from_before(device, my_img_ch=1, my_output_ch=1, my_output_process=my_output_process,net_path=pretrained_file,with_conv_7x7=with_conv_7x7, with_conv6=with_conv6, with_conv_1x1_repeat=with_conv_1x1_repeat, with_final_fc=with_final_fc)
else:
    net = Init_net_from_scratch(device, my_img_ch=1, my_output_ch=1, my_output_process=my_output_process,with_conv_7x7=with_conv_7x7, with_conv6=with_conv6, with_conv_1x1_repeat=with_conv_1x1_repeat, with_final_fc=with_final_fc)

if lr_decrease:
    decrease_for = [int((1/2**(i+1))*epochs) for i in range(decrease_times)]
    decrease_at = [i*0 for i in range(decrease_times)]
    for i in range(len(decrease_for)):
        for j in range(i+1):
            decrease_at[i] += decrease_for[j]
else:
    decrease_at = []

if freeze:
    freeze_dict = {
        str(int(epochs*freeze_time1)): [net.Conv1, net.Conv2, net.Conv3, net.Conv4, net.Conv5],
        str(int(epochs*freeze_times2)): [net.Conv1, net.Conv2, net.Conv3, net.Conv4, net.Conv5, net.Up5, net.Up_conv5, net.Up4, net.Up_conv4, net.Up3, net.Up_conv3, net.Up2, net.Up_conv2]
    }
    freeze_times = 2
else:
    freeze_times, freeze_dict = 0, {}
assert freeze_times == len([*freeze_dict]), "the freeze times is not compile with freeze dictionary! "

Save_notes = str(case_index) + '_' + training_method + '_' + loss_func
suffix_list, suffix_index = ['.pth', '.png', '.csv', '.txt', '.txt', '.png', '', ''], 0
need_mkdir_list = ['model_pth_basepath', 'visualize_basepath', 'trrData_basepath', 'curves_basepath', 'EP_basepath']
for key, value in savepath_para.items():
    if key not in need_mkdir_list and (not os.path.exists(value)):
        os.makedirs(value)
    if key in need_mkdir_list and (not os.path.exists(value + Save_notes)):
        os.makedirs(value + Save_notes)
    savepath_para[key] = value + Save_notes + suffix_list[suffix_index]
    suffix_index += 1
Criteria_para['losfunc'] = loss_func
Criteria_para['lamda'] = lamda_list
lr_para['lr'] = initial_lr
lr_para['lr_decrease'] = lr_decrease
lr_para['decrease_method'] = decrease_method
lr_para['decrease_times'] =decrease_times
lr_para['decrease_at'] = decrease_at
netfreeze_para['freeze'] = freeze
netfreeze_para['freeze_times'] = freeze_times
netfreeze_para['freeze_dict'] = freeze_dict
train_para['net'] = net
train_para['with_conv_7x7'] = with_conv_7x7
train_para['conv_7x7_double'] = conv_7x7_double
train_para['with_conv6'] = with_conv6
train_para['with_conv_1x1_repeat'] = with_conv_1x1_repeat
train_para['with_final_fc'] = with_final_fc
train_para['pretrained_pth_file'] = pretrained_file
train_para['fixed_pretrained_model'] = fixed_pretrained_model
train_para['output_process'] = my_output_process
train_para['training_method'] = training_method
train_para['dataset'] = Dataset_Root_Path + Subfolder
train_para['batch_size'] = bs
train_para['epochs'] = epochs
train_para['loss_weights_decay'] = loss_weights_decay
train_para['weights_decay_condtion'] = weights_decay_condtion
train_para['weight_decay_index'] = weight_decay_index
train_para['weight_decay_amplifier'] = weight_decay_amplifier
train_para['l2_lambda'] = l2_lambda
train_para['device'] = device
train_para['validate_eachi'] = validate_eachi
train_para['save_pth_eachi'] = save_pth_eachi
train_para['plotcurve_eachi'] = plotcurve_eachi
train_para['vidualize_specifici'] = vidualize_specifici
train_para['input_scaleup'] = input_scaleup
dataloader_para['Dataset_Root_Path'] = Dataset_Root_Path
dataloader_para['Subfolder'] = Subfolder
dataloader_para['data_split_ratio'] = data_split_ratio
dataloader_para['nw'] = nw
dataloader_para['shuffle'] = shuffle
dataloader_para['used_dataset_num'] = used_dataset_num
PAT_para['pixel_num_in_x'] = pixel_m
PAT_para['pixel_num_in_y'] = pixel_n
PAT_para['width_x'] = width_m
PAT_para['width_y'] = width_n
propagation_para['propagation_Dist'] = propagation_dist
propagation_para['normalize option'] = normalize_option
test_para['evalontestset_visual_trr_num'] = evalontestset_visual_trr_num
EP_collect_para['EP1_capacity'] = EP1_capacity  
EP_collect_para['EP2_capacity'] = EP2_capacity
EP_collect_para['EP_update_eachi'] = EP_update_eachi
EP_collect_para['EP_subfolder'] = EP_subfolder
EP_collect_para['EP_iteration'] = EP_iteration
EP_collect_para['EP_iteration_list'] = Iteration_list
if train_para['training_method'] == 'SL':
    train_para['epochs'] = EP_collect_para['EP_update_eachi']
    print("******************************")
    print("Due to the SL training method, the training epochs has been changed by EP update frequency! ")
    print("Thus, the original set of train_para['epochs'] is invalidated! ")
    print("******************************")
if PAT_para['width_x'] != PAT_para['width_y']:
    print("We should specify the sample frequency for x and y directions, respectively! ")
    exit()
propagation_para['sample frequency'] = 1/ (PAT_para['width_x']/for_sample_freq)

# Save the user defined parameters, including criteria, learning rate, network freeze, other train para, and save path information
user_defined_para = [Criteria_para, lr_para, netfreeze_para, train_para, savepath_para, PAT_para, propagation_para]
with open(savepath_para['trainpara_basepath'], 'a') as file0:
    print("------------------------------------------------------------------------------", file=file0)
    for user_defined_para_i in user_defined_para:
        for key, value in user_defined_para_i.items():
            print("\n==========", file=file0)
            print("+ {}: {}".format(key, value), file=file0)
            print("==========\n", file=file0)
    print("\n\n", file=file0)

total_loader, train_loader, valid_loader, test_loader = load_data(dataloader_para, train_para)

# --------------------- For EP collection in iterative learning based on experience pools --------------------- #
if train_para['training_method'] == 'SL':
    Iteration_list = EP_collect_para['EP_iteration_list']
    train_para['training_method'] = 'SSL'
    total_loader, train_loader, valid_loader, test_loader = load_data(dataloader_para, train_para)
    EP_Collect(train_loader, train_para, savepath_para, PAT_para, propagation_para, EP_collect_para)
    dataloader_para['Subfolder'] = EP_collect_para['EP_subfolder']
    dataloader_para['data_split_ratio'] = [1.0, 0.0, 0.0]
    train_para['training_method'] = 'SL'
    _, train_loader, _, _ = load_data(dataloader_para, train_para)
    print("=============================================")
    print("It is the 1 iteration of iterative learning. The superivsed learning conducted on {} is running...... ".format(Iteration_list[1]))
# -------------------------------------------------------------------------------------------------------------- #  
    
data_loader = [total_loader, train_loader, valid_loader, test_loader]
train_para['net'] = DL_train_val(train_para, data_loader, Criteria_para, lr_para, netfreeze_para, savepath_para, PAT_para, propagation_para, fontsize=fontsize)

# ------------------------------ For iterative learning based on experience pools ------------------------------ #
# If the training method is 'SL', the iterative learning is required between SL on EP1 and EP2
if train_para['training_method'] == 'SL':
    for ep_iteration_i in range(EP_collect_para['EP_iteration']):
        # Enter supervised learning on experience pool Iteration_list[0]
        print("=============================================")
        print("It is the {} iteration of iterative learning. The superivsed learning conducted on {} is running...... ".format(ep_iteration_i+1, Iteration_list[ep_iteration_i%2]))
        EP_collect_para['EP_subfolder'] = 'ExperiencePools/'+train_para['training_time']+'/'+Save_notes+'/'+Iteration_list[ep_iteration_i%2]+'/'
        dataloader_para['Subfolder'] = EP_collect_para['EP_subfolder']
        dataloader_para['data_split_ratio'] = [1.0, 0.0, 0.0]
        train_para['training_method'] = 'SL'
        _, train_loader, _, _ = load_data(dataloader_para, train_para)
        data_loader[-3] = train_loader
        train_para['net'] = DL_train_val(train_para, data_loader, Criteria_para, lr_para, netfreeze_para, savepath_para, PAT_para, propagation_para, fontsize=fontsize)
        print("It is the {} iteration of iterative learning. The superivsed learning conducted on {} is finished ".format(ep_iteration_i+1, Iteration_list[ep_iteration_i%2]))
        print("=============================================")
        # After iteration between EP1 and EP2, we need upade them
        train_para['training_method'] = 'SSL'
        dataloader_para['Subfolder'] = 'ExpectedAmpHolo/'
        _, train_loader, _, _ = load_data(dataloader_para, train_para)
        EP_Collect(train_loader, train_para, savepath_para, PAT_para, propagation_para, EP_collect_para)
        # Enter supervised learning  on experience pool Iteration_list[1]
        if ep_iteration_i != (EP_collect_para['EP_iteration']-1): # SL on EP2 is not needed
            print("=============================================")
            print("It is the {} iteration of iterative learning. The superivsed learning conducted on {} is running...... ".format(ep_iteration_i+2, Iteration_list[(ep_iteration_i+1)%2]))
            EP_collect_para['EP_subfolder'] = 'ExperiencePools/'+train_para['training_time']+'/'+Save_notes+'/'+Iteration_list[(ep_iteration_i+1)%2]+'/'
            dataloader_para['Subfolder'] = EP_collect_para['EP_subfolder']
            dataloader_para['data_split_ratio'] = [1.0, 0.0, 0.0]
            train_para['training_method'] = 'SL'
            _, train_loader, _, _ = load_data(dataloader_para, train_para)
            data_loader[-3] = train_loader
            train_para['net'] = DL_train_val(train_para, data_loader, Criteria_para, lr_para, netfreeze_para, savepath_para, PAT_para, propagation_para, fontsize=fontsize)
            print("It is the {} iteration of iterative learning. The superivsed learning conducted on {} is running...... ".format(ep_iteration_i+2, Iteration_list[(ep_iteration_i+1)%2]))
            print("=============================================")
# -------------------------------------------------------------------------------------------------------------- #   


# Test using well-trained network on testing set
batch_num = len(test_loader)
img_num = int(batch_num * train_para['batch_size'])
test_img = torch.zeros((img_num, 1, PAT_para['pixel_num_in_x'], PAT_para['pixel_num_in_y']))
for batch_num_i, test_loader_i in zip(range(batch_num), test_loader):
    if train_para['training_method'] in ['SSL', 'SSL_continue']:
        test_img_i = test_loader_i
    else:
        test_img_i, _ = test_loader_i
    # print(test_img_i.shape)  # torch.Size([16, 1, 50, 50])
    # print(test_img_i.size()[0]) # 16 or others
    test_img[batch_num_i*train_para['batch_size']:(batch_num_i+1)*train_para['batch_size']] = test_img_i
# ##
# net_path = "./Results_newOrgamized/2023-02-20/model_pth/4_SSL_continue_sse/Best.pth"
# net.load_state_dict(torch.load(net_path, map_location = device), strict=False)
# train_para['net'] = net.to(device)
# ##
test_avg_acc, test_avg_pc, test_avg_mse, test_avg_ssim, test_avg_effmean, test_avg_effratio, test_avg_unifo, test_avg_sd, test_avg_psnr, time_cost_seconds = DL_test(test_img, train_para, savepath_para, PAT_para, propagation_para, test_para)

print("------------------------------------------------------------------------------")
print("The test image number is {}".format((test_img.size()[0]//train_para['batch_size'])*train_para['batch_size']))
print("On thes test image:\navg Accuracy (ACC) = {}".format(test_avg_acc))
print("Pearson's correlation coefficient (PC) = {}".format(test_avg_pc))
print("NMSE = {}".format(test_avg_mse))
print("Structure similarity index metric (SSIM) = {}".format(test_avg_ssim))
print("Efficiency (EFF) mean = {}".format(test_avg_effmean))
print("Efficiency (EFF) ratio = {}".format(test_avg_effratio))
print("Uniformity = {}".format(test_avg_unifo))
print("Standard deviation = {}".format(test_avg_sd))
print("Peak signal noise ratio (PSNR) = {}".format(test_avg_psnr))
print("Average computing time is {} seconds for a batch size of {}".format(time_cost_seconds, train_para['batch_size']))
print("------------------------------------------------------------------------------")
print("({}  {}) case running done! ".format(train_para['training_time'], Save_notes))


