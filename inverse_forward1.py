import xlrd
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torch
import xlwt
import torch.utils.data as Data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer,scale
from sklearn.metrics import r2_score
from  matplotlib import pyplot as plt
import os
from PIL import Image
import matplotlib.image as mpimg
import torch.nn.functional as F
from math import exp
import numpy as np
from torch.optim import lr_scheduler

#check if CUDA is available
use_cuda=torch.cuda.is_available()
print("cuda:",use_cuda)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

path = "F:/zjl/attention-cnn/data-big-label.xlsx"
data = xlrd.open_workbook(path)
table = data.sheet_by_index(0)
pictur_list = []
lable_list = []
lable_list_times=[]
loader=transforms.Compose([transforms.ToTensor()])
unloader=transforms.ToPILImage()
image_path="F:/zjl/attention-cnn/data-big-zip"
for file in os.listdir(image_path):
    name,file_path=os.path.splitext(file)
    try:
        I = Image.open("F:/zjl/attention-cnn/data-big-zip/" + name + ".jpg").convert('RGB')
        lable_list.append(table.row_values(int(name), start_colx=0, end_colx=6))
        # lable_list_times.append(table.row_values(int(name), start_colx=5, end_colx=6))
        # lable_list=loader(lable_list)
        I=loader(I)
        pictur_list.append(np.array(I))
    except FileNotFoundError:
        continue
# print(lable_list)
# #最大最小值归一化
# mm=MinMaxScaler()
# lable_list=mm.fit_transform(lable_list)

# #手动设置放缩
# factor=0.1
# lable_list = [factor * i for i in list(lable_list)]
# print(lable_list)
#标准化
ss=StandardScaler()
lable_list=ss.fit_transform(lable_list)
# print(lable_list)
picture_array = np.array(pictur_list)
# print("shape1",picture_array.shape)
lable_array = np.array(lable_list)
# label1=lable_array[:,0]
# label1=label1/90
# print("label1",label1)
# label2=lable_array[:,1]
# label2=label2/360
# print("label2",label2)
# label3=lable_array[:,2]
# label3=label3/270
# print("label3",label3)
# label4=lable_array[:,3]
# label4=label4/360
# print("label4",label4)
# label5=lable_array[:,4]
# label5=label5/360
# print("label5",label5)


# print("shape label",lable_array.shape)
picture_array = picture_array.reshape(-1,3,138,80)

# print("shape2",picture_array.shape)


index = [i for i in range(13912)]
np.random.shuffle(index)
lable_array = lable_array[index]
picture_array = picture_array[index]

picture_tensor = torch.from_numpy(picture_array).float()
# picture_tensor = picture_tensor.unsqueeze(1).float()
lable_tensor = torch.from_numpy(lable_array).float()


train_data = picture_tensor[0:10800]
train_lable = lable_tensor[0:10800]


validata_data = picture_tensor[10800:13812]
validata_lable = lable_tensor[10800:13812]


test_data = picture_tensor[13812:13912]
test_lable = lable_tensor[13812:13912]


# path = "F:/zjl/attention-cnn/all_label_add.xlsx"
# data = xlrd.open_workbook(path)
# table = data.sheet_by_index(0)
# pictur_list = []
# lable_list = []
# lable_list_times=[]
# loader=transforms.Compose([transforms.ToTensor()])
# unloader=transforms.ToPILImage()
# image_path="F:/zjl/attention-cnn/times3-add-zip"
# for file in os.listdir(image_path):
#     name,file_path=os.path.splitext(file)
#     try:
#         I = Image.open("F:/zjl/attention-cnn/total_data-small-zip1/" + name + ".jpg").convert('RGB')
#         lable_list.append(table.row_values(int(name), start_colx=0, end_colx=6))
#         # lable_list_times.append(table.row_values(int(name), start_colx=5, end_colx=6))
#         # lable_list=loader(lable_list)
#         I=loader(I)
#         pictur_list.append(np.array(I))
#     except FileNotFoundError:
#         continue
# # print(lable_list)
# # #最大最小值归一化
# # mm=MinMaxScaler()
# # lable_list=mm.fit_transform(lable_list)
#
# # #手动设置放缩
# # factor=0.1
# # lable_list = [factor * i for i in list(lable_list)]
# # print(lable_list)
# #标准化
# ss=StandardScaler()
# lable_list=ss.fit_transform(lable_list)
# # print(lable_list)
# picture_array = np.array(pictur_list)
# # print("shape1",picture_array.shape)
# lable_array = np.array(lable_list)
# # label1=lable_array[:,0]
# # label1=label1/90
# # print("label1",label1)
# # label2=lable_array[:,1]
# # label2=label2/360
# # print("label2",label2)
# # label3=lable_array[:,2]
# # label3=label3/270
# # print("label3",label3)
# # label4=lable_array[:,3]
# # label4=label4/360
# # print("label4",label4)
# # label5=lable_array[:,4]
# # label5=label5/360
# # print("label5",label5)
#
#
# # print("shape label",lable_array.shape)
# picture_array = picture_array.reshape(-1,3,138,80)
#
# # print("shape2",picture_array.shape)
#
#
# index = [i for i in range(140330)]
# np.random.shuffle(index)
# lable_array = lable_array[index]
# picture_array = picture_array[index]
#
# picture_tensor = torch.from_numpy(picture_array).float()
# # picture_tensor = picture_tensor.unsqueeze(1).float()
# lable_tensor = torch.from_numpy(lable_array).float()
#
#
# train_data = picture_tensor[0:110000]
# train_lable = lable_tensor[0:110000]
#
#
# validata_data = picture_tensor[110000:140230]
# validata_lable = lable_tensor[110000:140230]
#
#
# test_data = picture_tensor[140230:140330]
# test_lable = lable_tensor[140230:140330]


# path = "F:/zjl/attention-cnn/big_sample3.xlsx"
# data = xlrd.open_workbook(path)
# table = data.sheet_by_index(0)
# pictur_list = []
# lable_list = []
# loader=transforms.Compose([transforms.ToTensor()])
# unloader=transforms.ToPILImage()
# image_path="F:/zjl/attention-cnn/big-sample3-zip"
# for file in os.listdir(image_path):
#     name,file_path=os.path.splitext(file)
#     I = Image.open("F:/zjl/attention-cnn/big-sample3-zip/" + name + ".jpg").convert('RGB')
#     lable_list.append(table.row_values(int(name), start_colx=0, end_colx=6))
#     I=loader(I)
#     pictur_list.append(np.array(I))
# picture_array = np.array(pictur_list)
# # print("shape1",picture_array.shape)
#
# #decrease the value of label in proportional
# lable_array = np.array(lable_list)
# label1=lable_array[:,0]
# label1=label1/90
# print("label1",label1)
# label2=lable_array[:,1]
# label2=label2/360
# print("label2",label2)
# label3=lable_array[:,2]
# label3=label3/270
# print("label3",label3)
# label4=lable_array[:,3]
# label4=label4/3
# print("label4",label4)
# label5=lable_array[:,4]
# print("label5",label5)
# label6=lable_array[:,5]
# print("label6",label6)
# # print("shape label",lable_array.shape)
# picture_array = picture_array.reshape(-1,3,138,80)
# # print("shape2",picture_array.shape)
#
#
# index = [i for i in range(24192)]
# np.random.shuffle(index)
# lable_array = lable_array[index]
# picture_array = picture_array[index]
#
# picture_tensor = torch.from_numpy(picture_array).float()
# # picture_tensor = picture_tensor.unsqueeze(1).float()
# lable_tensor = torch.from_numpy(lable_array).float()
#
# train_data = picture_tensor[0:18000]
# train_lable = lable_tensor[0:18000]
#
# validata_data = picture_tensor[18000:24100]
# validata_lable = lable_tensor[18000:24100]
#
# test_data = picture_tensor[24100:24192]
# test_lable = lable_tensor[24100:24192]

# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
#搭建网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        # 组合的卷积模块
        self.conv = nn.Sequential(
            nn.Conv2d(3, 10, 3), #卷积层，输入通道数为3，输出通道数为10，卷积核大小为5X5
            nn.BatchNorm2d(10),
            nn.LeakyReLU(), #激活函数层
            # nn.MaxPool2d(2, 2), #最大池化层，卷积核大小为2X2，图片减小一半
            nn.Conv2d(10, 16, 3), #卷积层，输入通道数为10，输出通道数为16，卷积核大小为5X5
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            # nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 21, 3),
            nn.BatchNorm2d(21),
            nn.LeakyReLU(),
            # nn.MaxPool2d(2, 2),
            nn.Conv2d(21, 24, 3),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
            nn.Conv2d(24, 32, 3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 36, 3),
            nn.BatchNorm2d(36),
            nn.LeakyReLU(),
            # nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(36*68*126, 120), #线性层
            torch.nn.BatchNorm1d(120), #归一化层
            nn.Dropout(0.2),
            nn.LeakyReLU(),#激活函数层
            nn.Linear(120, 84),
            torch.nn.BatchNorm1d(84),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(84, 20),
            torch.nn.BatchNorm1d(20),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(20, 6) #有三种标签，分别预测
        )
    def forward(self, x):
        y = self.conv(x)
        output = self.fc(y.view(x.shape[0], -1))
        return output
train_set = Data.TensorDataset(train_data,train_lable)
val_set=Data.TensorDataset(validata_data,validata_lable)
test_set=Data.TensorDataset(test_data,test_lable)
dataiter_tra = Data.DataLoader(dataset = train_set,
                           batch_size =200,
                           num_workers=0,
                           shuffle = True)
dataiter_val = Data.DataLoader(dataset = val_set,
                           batch_size =200,
                           num_workers=0,
                           shuffle = True)
dataiter_test = Data.DataLoader(dataset = test_set,
                           batch_size =200,
                           num_workers=0,
                           shuffle = True)
class ForwardCNN(nn.Module):
    def __init__(self):
        super( ForwardCNN,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(6, 20),  # 有三种标签，分别预测
            nn.Linear(20, 84),
            torch.nn.BatchNorm1d(84),
            nn.LeakyReLU(),
            nn.Linear(84, 120),
            torch.nn.BatchNorm1d(120),
            nn.LeakyReLU(),
            nn.Linear(120,36 * 68 * 126),  # 线性层
            torch.nn.BatchNorm1d(36 * 68 * 126),  # 归一化层
            nn.LeakyReLU(),  # 激活函数层
        )
        # 组合的卷积模块
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(36, 32, 3),  # 卷积层，输入通道数为3，输出通道数为10，卷积核大小为5X5
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),  # 激活函数层
            nn.ConvTranspose2d(32, 24, 3),  # 卷积层，输入通道数为3，输出通道数为10，卷积核大小为5X5
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),  # 激活函数层
            nn.ConvTranspose2d(24, 21, 3), #卷积层，输入通道数为3，输出通道数为10，卷积核大小为5X5
            nn.BatchNorm2d(21),
            nn.LeakyReLU(), #激活函数层
            # nn.MaxUnpool2d(2, 2), #最大池化层，卷积核大小为2X2，图片减小一半
            nn.ConvTranspose2d(21, 16, 3), #卷积层，输入通道数为10，输出通道数为16，卷积核大小为5X5
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            # nn.MaxUnpool2d(2, 2),
            nn.ConvTranspose2d(16, 10, 3),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),
            # nn.MaxUnpool2d(2, 2),
            nn.ConvTranspose2d(10, 3, 3),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(),
            # nn.MaxUnpool2d(2, 2)
        )

    def forward(self, x):
        y = self.fc(x.view(x.shape[0], -1))
        y=torch.unsqueeze(y,dim=2)
        y=torch.unsqueeze(y,dim=3)
        y=torch.reshape(y,(-1,36,126,68))
        output= self.conv(y)
        return output
# model_inverse = CNN()
# model_forward=ForwardCNN()
model_inverse=torch.load("model-new/model_inverse_big.pkl")
model_forward=torch.load("model-new/model_forward_big.pkl")
model_inverse=model_inverse.to(device)
model_forward=model_forward.to(device)
# loss_fun = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
# loss_fun1 = torch.nn.L1Loss()
# loss_fun2 = torch.nn.L1Loss()
loss_fun1 = torch.nn.L1Loss()
loss_fun2 = torch.nn.L1Loss()
# keep learning rate stable
opt1= torch.optim.Adam(model_inverse.parameters(),lr=0.001,weight_decay=1E-5)
opt2= torch.optim.Adam(model_forward.parameters(), lr=0.001,weight_decay=1E-5)

# #changing learning rate with epoch
# lr=0.01
# opt1= torch.optim.Adam(model_inverse.parameters(),lr=lr,weight_decay=1E-5)
# opt2= torch.optim.Adam(model_forward.parameters(), lr=lr,weight_decay=1E-5)
# lr_list=[]
# lambda1=lambda epoch:0.95 ** epoch
# scheduler1=lr_scheduler.LambdaLR(opt1,lr_lambda=lambda1)
# scheduler2=lr_scheduler.LambdaLR(opt2,lr_lambda=lambda1)

# #zishiying
# lr=0.01
# opt1= torch.optim.Adam(model_inverse.parameters(),lr=lr)
# opt2= torch.optim.Adam(model_forward.parameters(), lr=lr)
# lr_list=[]
# scheduler1=lr_scheduler.ReduceLROnPlateau(opt1,mode='min',factor=0.5,patience=15000,verbose=False,threshold=1e-4,threshold_mode='rel',
#                                           cooldown=0,min_lr=0,eps=1e-8)
# scheduler2=lr_scheduler.ReduceLROnPlateau(opt2,mode='min',factor=0.5,patience=15000,verbose=False,threshold=1e-4,threshold_mode='rel',
#                                           cooldown=0,min_lr=0,eps=1e-8)
val_loss=[]
val_loss_total=[]
tra_loss=[]
tra_forward_loss=[]
r_2_total=[]
r_2_1=[]
ssim_val1=[]
ssim_val_total=[]
epochs = 800
for epoch in range(epochs):

    for step, (x, y) in enumerate(dataiter_tra):
        x = x.to(device)
        y = y.to(device)
        out = model_inverse(x)
        stru = model_forward(out)
        loss_inverse = loss_fun1(out, y)
        loss_forward = loss_fun2(stru,x)
        loss_all = loss_inverse+loss_forward
        opt1.zero_grad()
        opt2.zero_grad()
        # loss_inverse.backward(retain_graph=True)
        loss_forward.backward(retain_graph=True)
        loss_all.backward()
        opt1.step()
        opt2.step()
        # scheduler1.step(loss_forward)
        # scheduler2.step(loss_all)
        # print("loss",loss_all.size())
    print("epoch:{},loss:{:.4f}".format(epoch, np.mean(loss_all.item())))
    #     loss_all=loss_all.cpu()
    #     tra_loss.append(loss_all.detach())
    #     # print("epoch:{},step:{},loss:{}".format(epoch, step, loss))
    # train_loss=np.mean(tra_loss)
    model_inverse.eval()
    model_forward.eval()
    for step, (x1, y1) in enumerate(dataiter_val):
        # move to GPU
        x1 = x1.to(device)
        y1 = y1.to(device)
        validata_out = model_inverse(x1)
        val_stru=model_forward(validata_out)
        validata_inverse_loss = loss_fun1(validata_out, y1)
        validata_loss_forward=loss_fun2(val_stru,x1)
        validata_loss = validata_inverse_loss + validata_loss_forward
        if epoch%50==0:
            y1 = y1.cpu()
            x1=x1.cpu()
            val_stru=val_stru.cpu()
            validata_out = validata_out.cpu()
            r_2 = r2_score(y1.detach(), validata_out.detach())
            validata_loss=validata_loss.cpu()
            val_loss.append(validata_loss.detach())
            r_2_1.append(r_2)
            ssim_val = ssim(x1.detach(), val_stru.detach(), size_average=True)
            ssim_val1.append(ssim_val)
            val_loss.append(validata_loss.detach())

    val_loss1 = np.mean(val_loss)
    ssim_val_total.append(np.mean(ssim_val1))
    val_loss_total.append(val_loss1)
    r_2 = np.mean(r_2_1)
    r_2_total.append(r_2)
    if epoch%50==0:
        print("epoch:{},验证损失:{:.4f},绝对系数:{:.4f},SSIM距离:{:.4f}".format(epoch, np.mean(val_loss), r_2, np.mean(ssim_val1)))
    # validata_data=validata_data.to(device)
    # validata_lable=validata_lable.to(device)
    # validata_out = model(validata_data)
    # validata_loss = loss_fun(validata_out, validata_lable)
    # validata_out=validata_out.cpu()
    # validata_lable=validata_lable.cpu()
    # validata_loss=validata_loss.cpu()
    # r_2 = r2_score(validata_lable.detach(), validata_out.detach())
    # val_loss.append(validata_loss.detach())
    # r_2_total.append(r_2)
    # print("epoch:{},验证损失:{},绝对系数:{}".format(epoch,validata_loss, r_2))
torch.save(model_inverse,"model-new/model_inverse_big1.pkl")
torch.save(model_forward,"model-new/model_forward_big1.pkl")

test_lable=test_lable.to(device)
test_data=test_data.to(device)
test_out = model_inverse(test_data)
test_stru = model_forward(test_out)
test_out=test_out.cpu()
test_lable=test_lable.cpu()
test_stru=test_stru.cpu()
test_data=test_data.cpu()
ssim_test = ssim(test_stru.detach(), test_data.detach(),size_average=True)
print("测试的SSIM距离:{:.4f}".format(ssim_test))

true_pic="result-big/true1/"
pre_pic="result-big/pre1/"
toPIL=transforms.ToPILImage()
for i in range (100):
    pic=toPIL(test_data[i])
    pic.save(true_pic+str(i) +'.jpg')
    pic1 = toPIL(test_stru[i])
    pic1.save(pre_pic + str(i) + '.jpg')
# test_lable=test_lable.detach().numpy()
# test_out=test_out.detach().numpy()
# factor1=10
# test_lable=[factor1 * i for i in list(test_lable.detach().numpy())]
# test_out=[factor1 * i for i in list(test_out.detach().numpy())]

# test_lable=mm.inverse_transform(test_lable.detach().numpy())
# test_out=mm.inverse_transform(test_out.detach().numpy())

test_lable=ss.inverse_transform(test_lable.detach().numpy())
test_out=ss.inverse_transform(test_out.detach().numpy())
pred_inc_angle=test_out[:,0]
pred_azi_angle1=test_out[:,1]
pred_rad=test_out[:,2]
pred_times=test_out[:,3]
pred_azi_angle2=test_out[:,4]
pred_azi_angle3=test_out[:,5]
True_inc_angle=test_lable[:,0]
True_azi_angle1=test_lable[:,1]
True_rad=test_lable[:,2]
True_times=test_lable[:,3]
True_azi_angle2=test_lable[:,4]
True_azi_angle3=test_lable[:,5]
# pred_inc_angle=test_out[:,0].detach()
# pred_azi_angle1=test_out[:,1].detach()
# pred_rad=test_out[:,2].detach()
# pred_times=test_out[:,3].detach()
# pred_azi_angle2=test_out[:,4].detach()
# pred_azi_angle3=test_out[:,5].detach()
# True_inc_angle=test_lable[:,0].detach()
# True_azi_angle1=test_lable[:,1].detach()
# True_rad=test_lable[:,2].detach()
# True_times=test_lable[:,3].detach()
# True_azi_angle2=test_lable[:,4].detach()
# True_azi_angle3=test_lable[:,5].detach()
val_loss1=np.ravel(val_loss)
r_2_total1=np.ravel(r_2_total)
# #对沉积次数四舍五入取整保存 tensor数据
# pred_rad=torch.round_(pred_rad)
# pred_dep_times=torch.round_(pred_times)
#对沉积次数四舍五入取整保存 numpy数据
pred_rad=np.around(pred_rad)
pred_dep_times=np.around(pred_times)
#保存参数
np.savetxt("F:/zjl/attention-cnn/result-big/loss-1.txt", val_loss1, fmt="%f")
np.savetxt("F:/zjl/attention-cnn/result-big/r_2-1.txt", r_2_total1, fmt="%f")
np.savetxt("F:/zjl/attention-cnn/result-big/pred_inc_angle-1.txt", pred_inc_angle, fmt="%d")
np.savetxt("F:/zjl/attention-cnn/result-big/true_inc_angle-1.txt", True_inc_angle, fmt="%d")
np.savetxt("F:/zjl/attention-cnn/result-big/pred_azi_angle1-1.txt", pred_azi_angle1, fmt="%d")
np.savetxt("F:/zjl/attention-cnn/result-big/true_azi_angle1-1.txt", True_azi_angle1, fmt="%d")
np.savetxt("F:/zjl/attention-cnn/result-big/pred_azi_angle2-1.txt", pred_azi_angle2, fmt="%d")
np.savetxt("F:/zjl/attention-cnn/result-big/true_azi_angle2-1.txt", True_azi_angle2, fmt="%d")
np.savetxt("F:/zjl/attention-cnn/result-big/pred_azi_angle3-1.txt", pred_azi_angle3, fmt="%d")
np.savetxt("F:/zjl/attention-cnn/result-big/true_azi_angle3-1.txt", True_azi_angle3, fmt="%d")
np.savetxt("F:/zjl/attention-cnn/result-big/pred_rad-1.txt", pred_rad, fmt="%d")
np.savetxt("F:/zjl/attention-cnn/result-big/true_rad-1.txt", True_rad, fmt="%d")
np.savetxt("F:/zjl/attention-cnn/result-big/pred_times-1.txt", pred_dep_times, fmt="%d")
np.savetxt("F:/zjl/attention-cnn/result-big/true_times-1.txt", True_times, fmt="%d")

#画图
plt.title("测试")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.xlabel("样本数")
plt.ylabel("结构参数")
plt.subplot(3, 2, 1)
plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_out[0:100][:, 0], label='入射角预测')
plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_lable[0:100][:, 0], label='入射角真实')
plt.legend()
plt.subplot(3, 2, 2)
plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_out[0:100][:, 1], label='方位角1预测')
plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_lable[0:100][:, 1], label='方位角1真实')
plt.legend()
plt.subplot(3, 2, 3)
plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_out[0:100][:, 2], label='半径预测')
plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_lable[0:100][:, 2], label='半径真实')
plt.legend()
plt.subplot(3, 2, 4)
plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_out[0:100][:, 3], label='次数预测')
plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_lable[0:100][:, 3], label='次数真实')
plt.legend()
plt.subplot(3, 2, 5)
plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_out[0:100][:, 4], label='方位角2预测')
plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_lable[0:100][:, 4], label='方位角2真实')
plt.legend()
plt.subplot(3, 2, 6)
plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_out[0:100][:, 5], label='方位角3预测')
plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_lable[0:100][:, 5], label='方位角3真实')
plt.legend()
plt.show()

# #画图
# plt.title("测试")
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.xlabel("样本数")
# plt.ylabel("结构参数")
# plt.subplot(3, 2, 1)
# plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_out[0:100][:, 0].detach(), label='入射角预测')
# plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_lable[0:100][:, 0].detach(), label='入射角真实')
# plt.legend()
# plt.subplot(3, 2, 2)
# plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_out[0:100][:, 1].detach(), label='方位角1预测')
# plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_lable[0:100][:, 1].detach(), label='方位角1真实')
# plt.legend()
# plt.subplot(3, 2, 3)
# plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_out[0:100][:, 2].detach(), label='半径预测')
# plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_lable[0:100][:, 2].detach(), label='半径真实')
# plt.legend()
# plt.subplot(3, 2, 4)
# plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_out[0:100][:, 3].detach(), label='次数预测')
# plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_lable[0:100][:, 3].detach(), label='次数真实')
# plt.legend()
# plt.subplot(3, 2, 5)
# plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_out[0:100][:, 4].detach(), label='方位角2预测')
# plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_lable[0:100][:, 4].detach(), label='方位角2真实')
# plt.legend()
# plt.subplot(3, 2, 6)
# plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_out[0:100][:, 5].detach(), label='方位角3预测')
# plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_lable[0:100][:, 5].detach(), label='方位角3真实')
# plt.legend()
# plt.show()













