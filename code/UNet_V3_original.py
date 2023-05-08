import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchsummary import summary

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            # nn.Tanh(),
            # nn.PReLU(),
            # nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
            # nn.Tanh()
            # nn.PReLU()
            # nn.LeakyReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out, size_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            nn.Upsample(size=size_out),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
            # nn.Tanh()
            # nn.PReLU()
            # nn.LeakyReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class conv_7x7(nn.Module):
    def __init__(self,ch_in,ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x


class UNet_V3(nn.Module):
    def __init__(self,img_ch=3,output_ch=1, output_process='sigmoid', with_conv_7x7=False, with_conv6=False, with_conv_1x1_repeat=False, with_final_fc=False):
        super(UNet_V3,self).__init__()
        
        self.output_ch = output_ch
        self.output_process = output_process
        self.with_conv_7x7 = with_conv_7x7
        self.with_conv6 = with_conv6
        self.with_conv_1x1_repeat = with_conv_1x1_repeat
        self.with_final_fc = with_final_fc

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        # self.Maxpool = nn.AvgPool2d(kernel_size=2,stride=2)
        # Pool -> Conv
        # self.Pool2Conv1 = nn.Conv2d(64, 64, kernel_size=3,stride=2,padding=1,bias=True)
        # self.Pool2Conv2 = nn.Conv2d(128, 128, kernel_size=3,stride=2,padding=1,bias=True)
        # self.Pool2Conv3 = nn.Conv2d(256, 256, kernel_size=3,stride=2,padding=1,bias=True)
        # self.Pool2Conv4 = nn.Conv2d(512, 512, kernel_size=3,stride=2,padding=1,bias=True)

        if self.with_conv_7x7:
            self.Conv0 = conv_7x7(ch_in=img_ch,ch_out=32)
            self.Conv1 = conv_block(ch_in=32,ch_out=64)
        elif not self.with_conv_7x7:
            self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)

        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        if self.with_conv6:
            self.Conv6 = conv_block(ch_in=1024,ch_out=2048)
            self.Up6 = up_conv(ch_in=2048,ch_out=1024, size_out=3)
            self.Up_conv6 = conv_block(ch_in=2048, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512, size_out=6)
        # self.Up5 = up_conv(ch_in=1024,ch_out=512, size_out=7)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256, size_out=12)
        # self.Up4 = up_conv(ch_in=512,ch_out=256, size_out=13)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128, size_out=25)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64, size_out=50)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        # self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        if self.with_conv_1x1_repeat:
            self.Conv_1x1 = nn.Conv2d(64,32,kernel_size=1,stride=1,padding=0)
            self.Conv_1x1_2 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)
        elif not self.with_conv_1x1_repeat:
            self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        
        if self.with_final_fc:
            self.fc = nn.Linear(2500, 2500)


    def forward(self,x):
        # encoding path
        if self.with_conv_7x7:
            x0 = self.Conv0(x)
            x1 = self.Conv1(x0)
        elif not self.with_conv_7x7:
            x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        # x2 = self.Pool2Conv1(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        # x3 = self.Pool2Conv2(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        # x4 = self.Pool2Conv3(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        # x5 = self.Pool2Conv4(x4)
        x5 = self.Conv5(x5)

        if self.with_conv6:
            x6 = self.Maxpool(x5)
            x6 = self.Conv6(x6)
            d6 = self.Up6(x6)
            d6 = torch.cat((x5,d6),dim=1)
            d6 = self.Up_conv6(x6)

            d5 = self.Up5(d6)


        # decoding + concat path
        elif not self.with_conv6:
            d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        # d1 = self.Conv_1x1(d2)

        if self.with_conv_1x1_repeat:
            d1 = self.Conv_1x1(d2)
            d1 = self.Conv_1x1_2(d1)
        elif not self.with_conv_1x1_repeat:
            d1 = self.Conv_1x1(d2)
        if self.with_final_fc:
            d1 = self.fc(torch.flatten(d1, 1))
            d1 = d1.reshape((-1,1, 50, 50))

        if self.output_ch == 1:
            if self.output_process == 'sigmoid_0to1':
                d1 = torch.sigmoid(d1)
            elif self.output_process == 'sigmoid_0to1_time10':
                d1 = torch.sigmoid(d1*10)
            elif self.output_process == 'sigmoid_0to1_time20':
                d1 = torch.sigmoid(d1*20)
            elif self.output_process == 'sigmoid_0to1_DIV2':
                d1 = torch.sigmoid(d1/2)
            elif self.output_process == 'sigmoid_0to1_DIV5':
                d1 = torch.sigmoid(d1/5)
            elif self.output_process == 'sigmoid_0to1_DIV10':
                d1 = torch.sigmoid(d1/10)
            elif self.output_process == 'sigmoid_0to1_DIV20':
                d1 = torch.sigmoid(d1/20)
            elif self.output_process == 'sigmoid_0to1_DIV42':
                d1 = torch.sigmoid(d1/42)
            elif self.output_process == 'sigmoid_0to1_DIV84':
                d1 = torch.sigmoid(d1/84)
            elif self.output_process == 'sigmoid':
                d1 = torch.sigmoid(d1) * 2 * math.pi # phs be in the range of [0, 2pi].
            elif self.output_process == 'tanh_minus1to1':
                d1 = torch.tanh(d1)
                # d1 = (torch.tanh(d1) + 1) / 2
            elif self.output_process == 'periodic nature':
                d1 = d1 - torch.floor(d1/(2*torch.pi)) * (2*torch.pi) # re-arrange to [0, 2pi]
            elif self.output_process == 'direct output':
                d1 = d1
        elif self.output_ch == 2:
            if self.output_process == 'sigmoid':
                d1[:,0] = torch.sigmoid(d1[:,0]) * 2 * math.pi # phs be in the range of [0, 2pi].
            if self.output_process == 'periodic nature':
                d1[:,0] = d1[:,0] - torch.floor(d1[:,0]/(2*torch.pi)) * (2*torch.pi) # re-arrange to [0, 2pi]
            if self.output_process == 'direct output':
                d1[:,0] = d1[:,0]
            d1[:,1] = torch.sigmoid(d1[:,1])    # re-arrange to [0, 1]
        
    # def forward(self,x):
    #     # encoding path
    #     x1 = self.Conv1(x)

    #     x2 = self.Maxpool(x1)
    #     x2 = self.Conv2(x2)
        
    #     x3 = self.Maxpool(x2)
    #     x3 = self.Conv3(x3)

    #     # x4 = self.Maxpool(x3)
    #     # x4 = self.Conv4(x4)

    #     # x5 = self.Maxpool(x4)
    #     # x5 = self.Conv5(x5)

    #     # d5 = self.Up5(x5)
    #     # d5 = torch.cat((x4,d5),dim=1)
        
    #     # d5 = self.Up_conv5(d5)
        
    #     # d4 = self.Up4(d5)
    #     # d4 = torch.cat((x3,d4),dim=1)
    #     # d4 = self.Up_conv4(d4)

    #     # d3 = self.Up3(d4)
    #     d3 = self.Up3(x3)
    #     d3 = torch.cat((x2,d3),dim=1)
    #     d3 = self.Up_conv3(d3)

    #     d2 = self.Up2(d3)
    #     d2 = torch.cat((x1,d2),dim=1)
    #     d2 = self.Up_conv2(d2)

    #     d1 = self.Conv_1x1(d2)

    #     if self.output_ch == 1:
    #         if self.output_process == 'sigmoid_0to1':
    #             d1 = torch.sigmoid(d1)
    #         elif self.output_process == 'sigmoid_0to1_time10':
    #             d1 = torch.sigmoid(d1*10)
    #         elif self.output_process == 'sigmoid_0to1_time20':
    #             d1 = torch.sigmoid(d1*20)
    #         elif self.output_process == 'sigmoid':
    #             d1 = torch.sigmoid(d1) * 2 * math.pi # phs be in the range of [0, 2pi].
    #         elif self.output_process == 'tanh_minus1to1':
    #             d1 = torch.tanh(d1)
    #             # d1 = (torch.tanh(d1) + 1) / 2
    #         elif self.output_process == 'periodic nature':
    #             d1 = d1 - torch.floor(d1/(2*torch.pi)) * (2*torch.pi) # re-arrange to [0, 2pi]
    #         elif self.output_process == 'direct output':
    #             d1 = d1
    #     elif self.output_ch == 2:
    #         if self.output_process == 'sigmoid':
    #             d1[:,0] = torch.sigmoid(d1[:,0]) * 2 * math.pi # phs be in the range of [0, 2pi].
    #         if self.output_process == 'periodic nature':
    #             d1[:,0] = d1[:,0] - torch.floor(d1[:,0]/(2*torch.pi)) * (2*torch.pi) # re-arrange to [0, 2pi]
    #         if self.output_process == 'direct output':
    #             d1[:,0] = d1[:,0]
    #         d1[:,1] = torch.sigmoid(d1[:,1])    # re-arrange to [0, 1]
    #     return d1

    # def forward(self,x):
    #     # encoding path
    #     x1 = self.Conv1(x)

    #     x2 = self.Maxpool(x1)
    #     x2 = self.Conv2(x2)  # [-1, 128, 25, 25]
        
    #     # x3 = self.Maxpool(x2)
    #     # x3 = self.Conv3(x3)

    #     # x4 = self.Maxpool(x3)
    #     # x4 = self.Conv4(x4)

    #     # x5 = self.Maxpool(x4)
    #     # x5 = self.Conv5(x5)

    #     # d5 = self.Up5(x5)
    #     # d5 = torch.cat((x4,d5),dim=1)
        
    #     # d5 = self.Up_conv5(d5)
        
    #     # d4 = self.Up4(d5)
    #     # d4 = torch.cat((x3,d4),dim=1)
    #     # d4 = self.Up_conv4(d4)

    #     # d3 = self.Up3(d4)
    #     # d3 = self.Up3(x2)
    #     # d3 = torch.cat((x2,d3),dim=1)
    #     # d3 = self.Up_conv3(d3)

        
    #     # d2 = self.Up2(d3)
    #     d2 = self.Up2(x2)
    #     d2 = torch.cat((x1,d2),dim=1)
    #     d2 = self.Up_conv2(d2)

    #     d1 = self.Conv_1x1(d2)

    #     if self.output_ch == 1:
    #         if self.output_process == 'sigmoid_0to1':
    #             d1 = torch.sigmoid(d1)
    #         elif self.output_process == 'sigmoid_0to1_time10':
    #             d1 = torch.sigmoid(d1*10)
    #         elif self.output_process == 'sigmoid_0to1_time20':
    #             d1 = torch.sigmoid(d1*20)
    #         elif self.output_process == 'sigmoid':
    #             d1 = torch.sigmoid(d1) * 2 * math.pi # phs be in the range of [0, 2pi].
    #         elif self.output_process == 'tanh_minus1to1':
    #             d1 = torch.tanh(d1)
    #             # d1 = (torch.tanh(d1) + 1) / 2
    #         elif self.output_process == 'periodic nature':
    #             d1 = d1 - torch.floor(d1/(2*torch.pi)) * (2*torch.pi) # re-arrange to [0, 2pi]
    #         elif self.output_process == 'direct output':
    #             d1 = d1
    #     elif self.output_ch == 2:
    #         if self.output_process == 'sigmoid':
    #             d1[:,0] = torch.sigmoid(d1[:,0]) * 2 * math.pi # phs be in the range of [0, 2pi].
    #         if self.output_process == 'periodic nature':
    #             d1[:,0] = d1[:,0] - torch.floor(d1[:,0]/(2*torch.pi)) * (2*torch.pi) # re-arrange to [0, 2pi]
    #         if self.output_process == 'direct output':
    #             d1[:,0] = d1[:,0]
    #         d1[:,1] = torch.sigmoid(d1[:,1])    # re-arrange to [0, 1]
        return d1
    
# #########################################################
# myUNet = UNet_V3(img_ch=2,with_conv_7x7=False, with_conv6=False, with_conv_1x1_repeat=False, with_final_fc=False)
# myUNet = nn.DataParallel(myUNet)
# CUDA = torch.cuda.is_available()
# if CUDA:
#     myUNet = myUNet.cuda()
# summary(myUNet, input_size=(2, 50, 50))
# #########################################################
