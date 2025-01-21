import xlrd
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torch
import xlwt
import torch.utils.data as Data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import r2_score
from  matplotlib import pyplot as plt
import os
from PIL import Image
import matplotlib.image as mpimg
import torch.nn.functional as F
from math import exp
import numpy as np
import pandas as pd

#check if CUDA is available
use_cuda=torch.cuda.is_available()
print("cuda:",use_cuda)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

path = ("/home/dell/Downloads/attention-cnn/result.xlsx")
df = pd.read_excel(path)

label_dict = {}
for i, row in df.iterrows():
    image_name = str(row['ImageName'])
    label_values = [row['Times'], row['IA'], row['SphereRad'], row['OADphi1'], row['OADphi2'], row['OADphi3']]
    label_dict[image_name] = label_values

pictur_list = []
label_list=[]
loader=transforms.Compose([transforms.ToTensor()])
unloader=transforms.ToPILImage()
image_path="/home/dell/Downloads/attention-cnn/123-gray-zip/"

def extract_number(file_name):
    try:
        return int(os.path.splitext(file_name)[0])
    except ValueError:
        return float('inf')  # 如果文件名不含数字，将其放到排序最后

files = sorted(os.listdir(image_path), key=extract_number)

selected_files=files[:9896]
print(selected_files)

for file in selected_files:

    name, ext = os.path.splitext(file)
    if ext.lower() not in ['.jpg', '.png', '.jpeg']:
        continue
    # 根据图片名称获取标签
    if name in label_dict:
        label = label_dict[name]
        label=label[:5]
    else:
        # 若找不到对应标签则跳过或处理异常
        continue

    img = Image.open(os.path.join(image_path, file)).convert('RGB')
    img = loader(img)
    pictur_list.append(np.array(img))
    label_list.append(label)

picture_array = np.array(pictur_list)
# print("shape1",picture_array.shape)
lable_array = np.array(label_list)
# print("shape label",lable_array.shape)
picture_array = picture_array.reshape(-1,3,138,80)
# print("shape2",picture_array.shape)


index = [i for i in range(9896)]
np.random.shuffle(index)
lable_array = lable_array[index]
picture_array = picture_array[index]

picture_tensor = torch.from_numpy(picture_array).float()
# picture_tensor = picture_tensor.unsqueeze(1).float()
lable_tensor = torch.from_numpy(lable_array).float()

# 数据集划分
total_count = len(picture_tensor)
train_count = int(total_count * 0.7)    # 70%
val_test_count = total_count - train_count  # 30%

train_data = picture_tensor[:train_count]
train_label = lable_tensor[:train_count]

val_test_data = picture_tensor[train_count:]
val_test_label = lable_tensor[train_count:]

# 从验证集中划分测试集，验证集的 20% 作为测试集
test_count = int(val_test_count * 0.05)
val_count = val_test_count - test_count

valid_data = val_test_data[:val_count]
valid_label = val_test_label[:val_count]

test_data = val_test_data[val_count:]
test_label = val_test_label[val_count:]

# 至此完成数据的加载与划分
print("训练集大小:", train_data.shape, train_label.shape)
print("验证集大小:", valid_data.shape, valid_label.shape)
print("测试集大小:", test_data.shape, test_label.shape)

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
            nn.ReLU(), #激活函数层
            # nn.MaxPool2d(2, 2), #最大池化层，卷积核大小为2X2，图片减小一半
            nn.Conv2d(10, 16, 3), #卷积层，输入通道数为10，输出通道数为16，卷积核大小为5X5
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 21, 3),
            nn.BatchNorm2d(21),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            nn.Conv2d(21, 24, 3),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 36, 3),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(36*68*126, 120), #线性层
            nn.BatchNorm1d(120), #归一化层
            nn.ReLU(), #激活函数层
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, 20),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Linear(20, 5) #有三种标签，分别预测
        )
    def forward(self, x):
        y = self.conv(x)
        output = self.fc(y.view(x.shape[0], -1))
        return output
train_set = Data.TensorDataset(train_data,train_label)
val_set=Data.TensorDataset(valid_data,valid_label)
test_set=Data.TensorDataset(test_data,test_label)
dataiter_tra = Data.DataLoader(dataset = train_set,
                           batch_size =500,
                           num_workers=0,
                           shuffle = True)
dataiter_val = Data.DataLoader(dataset = val_set,
                           batch_size =500,
                           num_workers=0,
                           shuffle = True)
dataiter_test = Data.DataLoader(dataset = test_set,
                           batch_size =500,
                           num_workers=0,
                           shuffle = True)
class ForwardCNN(nn.Module):
    def __init__(self):
        super( ForwardCNN,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(5, 20),  # 有三种标签，分别预测
            nn.Linear(20, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Linear(120,36 * 68 * 126),  # 线性层
            nn.BatchNorm1d(36 * 68 * 126),  # 归一化层
            nn.ReLU(),  # 激活函数层
        )
        # 组合的卷积模块
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(36, 32, 3),  # 卷积层，输入通道数为3，输出通道数为10，卷积核大小为5X5
            nn.BatchNorm2d(32),
            nn.ReLU(),  # 激活函数层
            nn.ConvTranspose2d(32, 24, 3),  # 卷积层，输入通道数为3，输出通道数为10，卷积核大小为5X5
            nn.BatchNorm2d(24),
            nn.ReLU(),  # 激活函数层
            nn.ConvTranspose2d(24, 21, 3), #卷积层，输入通道数为3，输出通道数为10，卷积核大小为5X5
            nn.BatchNorm2d(21),
            nn.ReLU(), #激活函数层
            # nn.MaxUnpool2d(2, 2), #最大池化层，卷积核大小为2X2，图片减小一半
            nn.ConvTranspose2d(21, 16, 3), #卷积层，输入通道数为10，输出通道数为16，卷积核大小为5X5
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.MaxUnpool2d(2, 2),
            nn.ConvTranspose2d(16, 10, 3),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            # nn.MaxUnpool2d(2, 2),
            nn.ConvTranspose2d(10, 3, 3),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            # nn.MaxUnpool2d(2, 2)
        )

    def forward(self, x):
        y = self.fc(x.view(x.shape[0], -1))
        y=torch.unsqueeze(y,dim=2)
        y=torch.unsqueeze(y,dim=3)
        y=torch.reshape(y,(-1,36,126,68))
        output= self.conv(y)
        return output
model_inverse = CNN()
model_forward=ForwardCNN()
# model_inverse=torch.load("model/model_inversecnn-2.pkl")
# model_forward=torch.load("model/model_forwordcnn-2.pkl")
model_inverse=model_inverse.to(device)
model_forward=model_forward.to(device)
# loss_fun = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_fun1 = torch.nn.L1Loss()
loss_fun2 = torch.nn.L1Loss()
# loss_fun1 = torch.nn.MSELoss()
# loss_fun2 = torch.nn.MSELoss()
opt1= torch.optim.Adam(model_inverse.parameters(),lr=0.01,weight_decay=1E-5)
opt2= torch.optim.Adam(model_forward.parameters(), lr=0.01,weight_decay=1E-5)
val_loss=[]
val_loss_total=[]
tra_loss=[]
tra_forward_loss=[]
r_2_total=[]
r_2_1=[]
ssim_val1=[]
ssim_val_total=[]
epochs = 2000
numbers='cnn-'+str(3)
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
        if epoch%20==0:
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
    val_loss1 = np.mean(val_loss)
    val_loss_total.append(val_loss1)
    r_2 = np.mean(r_2_1)
    r_2_total.append(r_2)
    if epoch%20==0:
        print("epoch:{},验证损失:{:.4f},绝对系数:{:.4f},SSIM距离:{:.4f}".format(epoch, val_loss1, r_2,np.mean(ssim_val1)))
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
torch.save(model_inverse,"model/model_inverse"+numbers+".pkl")
torch.save(model_forward,"model/model_forward"+numbers+".pkl")

test_lable=test_label.to(device)
test_data=test_data.to(device)
test_out = model_inverse(test_data)
test_stru = model_forward(test_out)
test_out=test_out.cpu()
test_lable=test_lable.cpu()
test_stru=test_stru.cpu()
test_data=test_data.cpu()
ssim_test = ssim(test_stru.detach(), test_data.detach(),size_average=True)
print("测试的SSIM距离:{:.4f}".format(ssim_test))

# true_pic="result/true1/"
# pre_pic="result/pre1/"
# toPIL=transforms.ToPILImage()
# for i in range (80):
#     pic=toPIL(test_data[i])
#     pic.save(true_pic+str(i) +'.jpg')
#     pic1 = toPIL(test_stru[i])
#     pic1.save(pre_pic + str(i) + '.jpg')
S=[]
pred_inc_angle=test_out[:,0].detach()
pred_azi_angle=test_out[:,1].detach()
pred_rad=test_out[:,2].detach()
pred_times=test_out[:,3].detach()
True_inc_angle=test_lable[:,0].detach()
True_azi_angle=test_lable[:,1].detach()
True_rad=test_lable[:,2].detach()
True_times=test_lable[:,3].detach()
val_loss1=np.ravel(val_loss)
r_2_total1=np.ravel(r_2_total)
#保存参数
np.savetxt("./result/loss_"+numbers+".txt", val_loss1, fmt="%f")
np.savetxt("./result/r_2_"+numbers+".txt", r_2_total1, fmt="%f")
np.savetxt("./result/pred_inc_angle_"+numbers+".txt", pred_inc_angle, fmt="%d")
np.savetxt("./result/true_inc_angle_"+numbers+".txt", True_inc_angle, fmt="%d")
np.savetxt("./result/pred_azi_angle1_"+numbers+".txt", pred_azi_angle, fmt="%d")
np.savetxt("./result/true_azi_angle1_"+numbers+".txt", True_azi_angle, fmt="%d")
np.savetxt("./result/pred_rad_"+numbers+".txt", pred_rad, fmt="%d")
np.savetxt("./result/true_rad_"+numbers+".txt", True_rad, fmt="%d")
np.savetxt("./result/pred_times_"+numbers+".txt", pred_times, fmt="%d")
np.savetxt("./result/true_times_"+numbers+".txt", True_times, fmt="%d")

#画图
plt.title("测试")
plt.xlabel("样本数")
plt.ylabel("结构参数")
plt.subplot(2, 2, 1)
plt.plot(np.array([i for i in range(1, 81)]).reshape(80, 1), test_out[0:80][:, 0].detach(), label='入射角预测')
plt.plot(np.array([i for i in range(1, 81)]).reshape(80, 1), test_lable[0:80][:, 0].detach(), label='入射角真实')
plt.legend()
plt.subplot(2, 2, 2)
plt.plot(np.array([i for i in range(1, 81)]).reshape(80, 1), test_out[0:80][:, 1].detach(), label='方位角预测')
plt.plot(np.array([i for i in range(1, 81)]).reshape(80, 1), test_lable[0:80][:, 1].detach(), label='方位角真实')
plt.legend()
plt.subplot(2, 2, 3)
plt.plot(np.array([i for i in range(1, 81)]).reshape(80, 1), test_out[0:80][:, 2].detach(), label='半径预测')
plt.plot(np.array([i for i in range(1, 81)]).reshape(80, 1), test_lable[0:80][:, 2].detach(), label='半径真实')
plt.legend()
plt.subplot(2, 2, 4)
plt.plot(np.array([i for i in range(1, 81)]).reshape(80, 1), test_out[0:80][:, 3].detach(), label='次数预测')
plt.plot(np.array([i for i in range(1, 81)]).reshape(80, 1), test_lable[0:80][:, 3].detach(), label='次数真实')
plt.legend()
plt.show()













