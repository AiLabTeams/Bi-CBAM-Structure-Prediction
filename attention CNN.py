import xlrd
import numpy as np
import torch.nn as nn
import torch
import shutil
import torchvision.transforms as transforms
import xlwt
import torch.utils.data as Data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from numpy import *
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
import os
from PIL import Image
import matplotlib.image as mpimg
import os
import sys
import cv2
from multiprocessing import cpu_count
import torch.nn.functional as F

cpu_num = cpu_count() # 自动获取最大核心数目
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,BASE_DIR)

#check if CUDA is available
use_cuda=torch.cuda.is_available()
print("cuda:",use_cuda)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# #对图片进行等比例压缩之后保存在“压缩之后的图片文件夹中”
# for i in range(1,607501):
#     I = Image.open(r"F:/zjl/total_sample/" + str(i) + ".jpg")
#     print("I:",I)
#     img_deal = I.resize((60,80),Image.ANTIALIAS)
#     img_deal.save("F:/zjl/attention-cnn/total_sample_zip/"+str(i)+".jpg")

# path = "F:/zjl/attention-cnn/label-crop.xlsx"
# data = xlrd.open_workbook(path)
# table = data.sheet_by_index(0)
# pictur_list = []
# lable_list = []
#
# image_path="F:/zjl/attention-cnn/total-sample-crop"
# for file in os.listdir(image_path):
#     name,file_path=os.path.splitext(file)
#     I = Image.open("F:/zjl/attention-cnn/total-sample-crop/" + name + ".jpg")
#     lable_list.append(table.row_values(int(name), start_colx=0, end_colx=4))
#     pictur_list.append(np.array(I))
# picture_array = np.array(pictur_list)
# lable_array = np.array(lable_list)
#
# picture_array = picture_array.reshape(-1,3,46,80)
#
#
# index = [i for i in range(10000)]
# np.random.shuffle(index)
# lable_array = lable_array[index]
# picture_array = picture_array[index]
#
# picture_tensor = torch.from_numpy(picture_array).float()
# # picture_tensor = picture_tensor.unsqueeze(1).float()
# lable_tensor = torch.from_numpy(lable_array).float()
#
# train_data = picture_tensor[0:8000]
# train_lable = lable_tensor[0:8000]
#
# validata_data1 = picture_tensor[8000:9700]
# validata_lable1 = lable_tensor[8000:9700]
#
# test_data = picture_tensor[9700:10000]
# test_lable = lable_tensor[9700:10000]

# path = "F:/zjl/attention-cnn/all_label_add1.xlsx"
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
#     I = Image.open("F:/zjl/attention-cnn/times3-add-zip/" + name + ".jpg").convert('RGB')
#     lable_list.append(table.row_values(int(name), start_colx=0, end_colx=5))
#     lable_list_times.append(table.row_values(int(name), start_colx=5, end_colx=6))
#     # lable_list=loader(lable_list)
#     I=loader(I)
#     pictur_list.append(np.array(I))
# # print(lable_list)
# # #最大最小值归一化
# mm=MinMaxScaler()
# lable_list=mm.fit_transform(lable_list)
# # #标准化
# # ss=StandardScaler()
# # lable_list=ss.fit_transform(lable_list)
# # print(lable_list)
# picture_array = np.array(pictur_list)
# # print("shape1",picture_array.shape)
#
# lable_list_times=np.ravel(lable_list_times)
# # print("label_times:",lable_list_times)
# #decrease the value of label in proportional
# lable_array = np.array(lable_list)
# lable_array1=np.array(lable_list_times)
# lable_array1=lable_array1-1
# # lable_array1.astype(int)
# lable_array1=np.eye(3)[lable_array1.astype(int)]
# # print("label:",lable_array1)
# # label1=lable_array[:,0]
# # label1=label1/10
# # # print("label1",label1)
# # label2=lable_array[:,1]
# # label2=label2/100
# # # print("label2",label2)
# # label3=lable_array[:,2]
# # label3=label3/100
# # # print("label3",label3)
# # label4=lable_array[:,3]
# # label4=label4/100
# # # print("label4",label4)
# # label5=lable_array[:,4]
# # label5=label5/100
#
# # print("shape label",lable_array.shape)
# picture_array = picture_array.reshape(-1,3,138,80)
#
# # print("shape2",picture_array.shape)
#
#
# index = [i for i in range(142662)]
# #设置随机种子，保证下次运行该代码时划分的数据一样
# np.random.seed(seed=0)
# np.random.shuffle(index)
# lable_array = lable_array[index]
# lable_array1=lable_array1[index]
# picture_array = picture_array[index]
#
# picture_tensor = torch.from_numpy(picture_array).float()
# # picture_tensor = picture_tensor.unsqueeze(1).float()
# lable_tensor = torch.from_numpy(lable_array).float()
# lable_times_tensor=torch.from_numpy(lable_array1).float()
#
#
# train_data = picture_tensor[0:110000]
# train_lable1 = lable_tensor[0:110000]
# train_lable2=lable_times_tensor[0:110000]
#
# validata_data = picture_tensor[110000:142562]
# validata_lable1 = lable_tensor[110000:142562]
# validata_lable2=lable_times_tensor[110000:142562]
#
# test_data = picture_tensor[142562:142662]
# test_lable1 = lable_tensor[142562:142662]
# test_lable2=lable_times_tensor[142562:142662]

path = "F:/zjl/attention-cnn/data-big-label.xlsx"
data = xlrd.open_workbook(path)
table = data.sheet_by_index(0)
pictur_list = []
lable_list = []
lable_list_times=[]
loader=transforms.Compose([transforms.ToTensor()])
unloader=transforms.ToPILImage()
image_path="F:/zjl/attention-cnn/data-small-zip"
for file in os.listdir(image_path):
    name,file_path=os.path.splitext(file)
    try:
        I = Image.open("F:/zjl/attention-cnn/data-small-zip/" + name + ".jpg").convert('RGB')
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


index = [i for i in range(13804)]
np.random.shuffle(index)
lable_array = lable_array[index]
picture_array = picture_array[index]

picture_tensor = torch.from_numpy(picture_array).float()
# picture_tensor = picture_tensor.unsqueeze(1).float()
lable_tensor = torch.from_numpy(lable_array).float()


train_data = picture_tensor[0:10800]
train_lable = lable_tensor[0:10800]


validata_data = picture_tensor[10800:13704]
validata_lable = lable_tensor[10800:13704]


test_data = picture_tensor[13704:13804]
test_lable = lable_tensor[13704:13804]

def save_ckp(state,is_best,checkpoint_path,best_model_path):
    '''

    :param state: checkpoint we want to save
    :param is_best: the best checkpoint;min validation loss
    :param checkpoint_path: path to save checkpoint
    :param best_model_path: path to save best model
    :return:
    '''
    f_path=checkpoint_path
    torch.save(state,f_path)
    if is_best:
        best_fpath=best_model_path
        shutil.copyfile(f_path,best_fpath)
        pass
    pass
def load_ckp(checkpoint_fpath,model,optimizer):
    '''

    :param checkpoint_fpath:path to save checkpoint
    :param model: model that we want to load checkpoint parameters into
    :param optimizer: optimizer we defined in previous training
    :return:
    '''
    #load check point
    checkpoint=torch.load(checkpoint_fpath)
    #initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to model
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to model
    valid_loss_min=checkpoint['valid_loss_min']
    r_2_max=checkpoint['r_2_max']
    return model,optimizer,checkpoint['epoch'],valid_loss_min.item(),r_2_max.item()

class ChannelAttention(nn.Module):
    def __init__(self, in_channel, ratio=16):
        """ 通道注意力机制 同最大池化和平均池化两路分别提取信息，后共用一个多层感知机mlp,再将二者结合

        :param in_channel: 输入通道
        :param ratio: 通道降低倍率
        """
        super(ChannelAttention, self).__init__()
        # 平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 通道先降维后恢复到原来的维数
        self.fc1 = nn.Conv2d(in_channel, in_channel // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channel // ratio, in_channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print("x.shape",x.shape)
        # 平均池化
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # 最大池化
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # out = avg_out + max_out
        # return x*self.sigmoid(out)

        # 平均池化一支 (2,512,8,8) -> (2,512,1,1) -> (2,512/ration,1,1) -> (2,512,1,1)
        # (2,512,8,8) -> (2,512,1,1)
        avg = self.avg_pool(x)
        # 多层感知机mlp (2,512,8,8) -> (2,512,1,1) -> (2,512/ration,1,1) -> (2,512,1,1)
        # (2,512,1,1) -> (2,512/ratio,1,1)
        avg = self.fc1(avg)
        avg = self.relu1(avg)
        # (2,512/ratio,1,1) -> (2,512,1,1)
        avg_out = self.fc2(avg)

        # 最大池化一支
        # (2,512,8,8) -> (2,512,1,1)
        max = self.max_pool(x)
        # 多层感知机
        # (2,512,1,1) -> (2,512/ratio,1,1)
        max = self.fc1(max)
        max = self.relu1(max)
        # (2,512/ratio,1,1) -> (2,512,1,1)
        max_out = self.fc2(max)

        # (2,512,1,1) + (2,512,1,1) -> (2,512,1,1)
        out = avg_out + max_out
        output=self.sigmoid(out)
        # print("output.shape",output.shape)
        return x*output

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        """ 空间注意力机制 将通道维度通过最大池化和平均池化进行压缩，然后合并，再经过卷积和激活函数，结果和输入特征图点乘

        :param kernel_size: 卷积核大小
        """
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print('x shape', x.shape)
        # (2,512,8,8) -> (2,1,8,8)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # (2,512,8,8) -> (2,1,8,8)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # (2,1,8,8) + (2,1,8,8) -> (2,2,8,8)
        cat = torch.cat([avg_out, max_out], dim=1)
        # (2,2,8,8) -> (2,1,8,8)
        out = self.conv1(cat)
        output=self.sigmoid(out)
        # print("output.shape", output.shape)
        return x*output

class CNN(nn.Module):
    def __init__(self,ratio,kernel_size):
        super(CNN,self).__init__()
        self.ca1=ChannelAttention(in_channel=6,ratio=ratio)
        self.sa1=SpatialAttention(kernel_size=kernel_size)
        self.ca2 = ChannelAttention(in_channel=10, ratio=ratio)
        self.sa2 = SpatialAttention(kernel_size=kernel_size)
        self.ca3 = ChannelAttention(in_channel=16, ratio=ratio)
        self.sa3 = SpatialAttention(kernel_size=kernel_size)
        self.ca4 = ChannelAttention(in_channel=24, ratio=ratio)
        self.sa4 = SpatialAttention(kernel_size=kernel_size)
        self.conv1=nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1)
        self.bn1=nn.BatchNorm2d(6)
        self.re1=nn.ReLU()
        # self.maxp1=nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=10, kernel_size=3,stride=1)
        self.bn2 = nn.BatchNorm2d(10)
        self.re2 = nn.ReLU()
        # self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3,stride=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.re3 = nn.ReLU()
        # self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=1, stride=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.re4 = nn.ReLU()
        # self.maxp4 = nn.MaxPool2d(2, 2)
        # self.conv4 = nn.Conv2d(in_channels=24, out_channels=16, kernel_size=3, stride=1)
        # self.bn4 = nn.BatchNorm2d(16)
        # self.re4 = nn.ReLU()


        self.fc1 = nn.Sequential(
            # nn.Linear(24*8*15, 120),#加入最大池化层之后的参数 80-138尺寸大小
            # nn.Linear(24 * 4 * 8, 120), #加入最大池化层之后的参数 46-80尺寸大小
            nn.Linear(24 * 74 * 132, 120),  #80-138 ，不加最大池化层
            torch.nn.BatchNorm1d(120),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(120, 84),
            torch.nn.BatchNorm1d(84),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(84, 20),
            torch.nn.BatchNorm1d(20),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 6)
        )
        self.fc2 = nn.Sequential(
            # nn.Linear(24*8*15, 120),#加入最大池化层之后的参数 80-138尺寸大小
            # nn.Linear(24 * 4 * 8, 120), #加入最大池化层之后的参数 46-80尺寸大小
            nn.Linear(24 * 74 * 132, 120),  # 80-138 ，不加最大池化层
            torch.nn.BatchNorm1d(120),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(120, 84),
            torch.nn.BatchNorm1d(84),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(84, 20),
            torch.nn.BatchNorm1d(20),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 3),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        # y1 = self.re1(self.bn1(self.conv1(x)))
        # y_1=y1*self.ca1(y1)
        # y_out1=y_1*self.sa1(y_1)
        # y2 = self.re2(self.bn2(self.conv2(y_out1)))
        # y_2 = y2 * self.ca2(y2)
        # y_out2 =  y_2 * self.sa2(y_2)
        # y3 = self.re3(self.bn3(self.conv3(y_out2)))
        # y_3 = y3 * self.ca3(y3)
        # y_out3 = y_3 * self.sa3(y_3)
        # y4 = self.re4(self.bn4(self.conv4(y_out3)))
        # y_4 = y4 * self.ca4(y4)
        # y_out4 =y_4 * self.sa4(y_4)
        # output = self.fc(y_out4.view(x.shape[0], -1))

        #专用空间注意力机制
        y1 = self.re1(self.bn1(self.conv1(x)))
        y_out1 = y1 * self.sa1(y1)
        y2 = self.re2(self.bn2(self.conv2(y_out1)))
        y_out2 = y2 * self.sa2(y2)
        y3 = self.re3(self.bn3(self.conv3(y_out2)))
        y_out3 = y3 * self.sa3(y3)
        y4 = self.re4(self.bn4(self.conv4(y_out3)))
        y_out4 = y4 * self.sa4(y4)
        output1 = self.fc1(y_out4.view(x.shape[0], -1))
        # output2=self.fc2(y_out4.view(x.shape[0], -1))
        # # 专用空间注意力机制---每一层加入最大池化层
        # y1 = self.maxp1(self.re1(self.bn1(self.conv1(x))))
        # y_out1 = y1 * self.sa1(y1)
        # y2 = self.maxp2(self.re2(self.bn2(self.conv2(y_out1))))
        # y_out2 = y2 * self.sa2(y2)
        # y3 = self.maxp3(self.re3(self.bn3(self.conv3(y_out2))))
        # y_out3 = y3 * self.sa3(y3)
        # y4 =self.re4(self.bn4(self.conv4(y_out3)))
        # y_out4 = y4 * self.sa4(y4)
        # output = self.fc(y_out4.view(x.shape[0], -1))

        # #不加注意力机制
        # y1 = self.re1(self.bn1(self.conv1(x)))
        # y2 = self.re2(self.bn2(self.conv2(y1)))
        # y3 = self.re3(self.bn3(self.conv3(y2)))
        # y4 = self.re4(self.bn4(self.conv4(y3)))
        # output = self.fc(y4.view(x.shape[0], -1))
        return output1

# train_set = Data.TensorDataset(train_data,train_lable1,train_lable2)
# val_set=Data.TensorDataset(validata_data,validata_lable1,validata_lable2)
# test_set=Data.TensorDataset(test_data,test_lable1,test_lable2)
train_set = Data.TensorDataset(train_data,train_lable)
val_set=Data.TensorDataset(validata_data,validata_lable)
test_set=Data.TensorDataset(test_data,test_lable)
dataiter_tra = Data.DataLoader(dataset = train_set,
                           batch_size =300,
                           num_workers=0,
                           shuffle = True)
dataiter_val = Data.DataLoader(dataset = val_set,
                           batch_size =300,
                           num_workers=0,
                           shuffle = True)
dataiter_test = Data.DataLoader(dataset = test_set,
                           batch_size =300,
                           num_workers=0,
                           shuffle = True)
model = CNN(ratio=2,kernel_size=3)
# model=torch.load("model/model_attention.pkl")
model=model.to(device)
classes=3
weight=torch.empty(classes).uniform_(0,1)
loss2=nn.CrossEntropyLoss(weight=weight,reduction='none')
# loss_fun = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_fun1 = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01,weight_decay=1E-5)
val_loss=[]
val_loss_total=[]
tra_loss=[]
r_2_total=[]
r_2_1=[]
epochs=1000
batch_size=300

for epoch in range(epochs):
    # model.train() #激活dropout层
    train_loss=0
    val_loss=0
    num_correct=0
    accuracy=0
    total=0
    num=0
    loss2=loss2.to(device)
    for step,(x,y1) in enumerate(dataiter_tra):
        #move to GPU

        x=x.to(device)
        y1=y1.to(device)

        # target = y2.torch.empty(batch_size, dtype=torch.long).random_(classes)
        # clear the gradients of all oprtimized variables
        optimizer.zero_grad()
        output1 = model(x)
        # print("output2:",output2)
        train_loss= loss_fun1(output1, y1)
        # train_loss2=torch.nn.functional.cross_entropy(output2,y2)

        train_loss.backward()
        optimizer.step()

        tra_loss.append(train_loss)
        pass
    print("Train Epoch:{}\t Loss:{:.4f}".format(epoch,np.mean(train_loss.item())))
    model.eval()
    for step,(x1,y1) in enumerate(dataiter_val):
        #move to GPU
        x1=x1.to(device)
        y1=y1.to(device)

        # target = y2.torch.empty(batch_size, dtype=torch.long).random_(classes)
        validata_out1= model(x1)
        validata_loss = loss_fun1(validata_out1, y1)
        # validata_loss2=torch.nn.functional.cross_entropy(validata_out2,y2)

        if epoch%50==0:
            y1=y1.cpu()

            validata_out1=validata_out1.cpu()
            r_2 = r2_score(y1.detach(), validata_out1.detach())

            # val_loss.append(validata_loss.detach())
            r_2_1.append(r_2)

    # val_loss1=np.mean(val_loss)
    # val_loss_total.append(val_loss1)
    r_2=np.mean(r_2_1)
    r_2_total.append(r_2)
    if epoch % 50 == 0:
        print("epoch:{},验证损失:{:.4f},绝对系数:{:.4f}".format(epoch,np.mean(validata_loss.item()), r_2))
torch.save(model,"model-big/model_attention.pkl")


#测试
test_accuracy=0
test_lable1=test_lable.to(device)

test_data=test_data.to(device)
test_out1= model(test_data)
test_lable1=test_lable1.cpu()

test_out1=test_out1.cpu()
r_2_test = r2_score(test_lable1.detach(), test_out1.detach())

print(" r_2_test:{:.4f}".format(r_2_test))

# test_out1=test_out1.detach().numpy()
# test_lable1=mm.inverse_transform(test_lable1.detach().numpy())
# test_out1=mm.inverse_transform(test_out1.detach().numpy())
test_lable=ss.inverse_transform(test_lable1.detach().numpy())
test_out=ss.inverse_transform(test_out1.detach().numpy())
# pred_inc_angle=test_out1[:,0]*10
# pred_azi_angle1=test_out1[:,1]*100
# pred_rad=test_out1[:,2]*100
# pred_azi_angle2=test_out1[:,3]*100
# pred_azi_angle3=test_out1[:,4]*100
# pred_times=test_pred+1
#
# True_inc_angle=test_lable1[:,0]*10
# True_azi_angle1=test_lable1[:,1]*100
# True_rad=test_lable1[:,2]*100
# True_azi_angle2=test_lable1[:,3]*100
# True_azi_angle3=test_lable1[:,4]*100
# True_times=test_lable2+1

# # test_data=test_data.cpu()
# test_out=test_out.cpu()
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

True_times=test_lable1[:,5]
pred_times=test_out1[:,5]
pred_inc_angle=test_out1[:,0]
pred_azi_angle1=test_out1[:,1]
pred_rad=test_out1[:,2]
pred_azi_angle2=test_out1[:,3]
pred_azi_angle3=test_out1[:,4]

True_inc_angle=test_lable1[:,0]
True_azi_angle1=test_lable1[:,1]
True_rad=test_lable1[:,2]
True_azi_angle2=test_lable1[:,3]
True_azi_angle3=test_lable1[:,4]

val_loss1=np.ravel(val_loss)
r_2_total1=np.ravel(r_2_total)
# #对沉积次数四舍五入取整保存 numpy数据
# pred_rad=np.around(pred_rad)
# pred_dep_times=np.around(pred_times)
#
# #对沉积次数四舍五入取整保存
# # pred_rad=torch.round_(pred_rad)
# # pred_dep_times=torch.round_(pred_times)
#保存参数
np.savetxt("F:/zjl/attention-cnn/result-big/loss-3.txt", val_loss1, fmt="%f")
np.savetxt("F:/zjl/attention-cnn/result-big/r_2-3.txt", r_2_total1, fmt="%f")
np.savetxt("F:/zjl/attention-cnn/result-big/pred_inc_angle-3.txt", pred_inc_angle, fmt="%d")
np.savetxt("F:/zjl/attention-cnn/result-big/true_inc_angle-3.txt", True_inc_angle, fmt="%d")
np.savetxt("F:/zjl/attention-cnn/result-big/pred_azi_angle1-3.txt", pred_azi_angle1, fmt="%d")
np.savetxt("F:/zjl/attention-cnn/result-big/true_azi_angle1-3.txt", True_azi_angle1, fmt="%d")
np.savetxt("F:/zjl/attention-cnn/result-big/pred_azi_angle2-3.txt", pred_azi_angle2, fmt="%d")
np.savetxt("F:/zjl/attention-cnn/result-big/true_azi_angle2-3.txt", True_azi_angle2, fmt="%d")
np.savetxt("F:/zjl/attention-cnn/result-big/pred_azi_angle3-3.txt", pred_azi_angle3, fmt="%d")
np.savetxt("F:/zjl/attention-cnn/result-big/true_azi_angle3-3.txt", True_azi_angle3, fmt="%d")
np.savetxt("F:/zjl/attention-cnn/result-big/pred_rad-3.txt", pred_rad, fmt="%d")
np.savetxt("F:/zjl/attention-cnn/result-big/true_rad-3.txt", True_rad, fmt="%d")
np.savetxt("F:/zjl/attention-cnn/result-big/pred_times-3.txt", pred_times, fmt="%d")
np.savetxt("F:/zjl/attention-cnn/result-big/true_times-3.txt", True_times, fmt="%d")

#画图
plt.title("测试")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.xlabel("样本数")
plt.ylabel("结构参数")
plt.subplot(3, 2, 1)
plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_out1[0:100][:, 0], label='入射角预测')
plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_lable1[0:100][:, 0], label='入射角真实')
plt.legend()
plt.subplot(3, 2, 2)
plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_out1[0:100][:, 1], label='方位角1预测')
plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_lable1[0:100][:, 1], label='方位角1真实')
plt.legend()
plt.subplot(3, 2, 3)
plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_out1[0:100][:, 2], label='半径预测')
plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_lable1[0:100][:, 2], label='半径真实')
plt.legend()
plt.subplot(3, 2, 4)
plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), pred_times[0:100], label='次数预测')
plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), True_times[0:100], label='次数真实')
plt.legend()
plt.subplot(3, 2, 5)
plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_out1[0:100][:, 3], label='方位角2预测')
plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_lable1[0:100][:, 3], label='方位角2真实')
plt.legend()
plt.subplot(3, 2, 6)
plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_out1[0:100][:, 4], label='方位角3预测')
plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_lable1[0:100][:,4], label='方位角3真实')
plt.legend()
plt.show()

# for epoch in range(epochs):
#     # model.train() #激活dropout层
#     train_loss=0
#     val_loss=0
#     num_correct=0
#     accuracy=0
#     total=0
#     num=0
#     loss2=loss2.to(device)
#     for step,(x,y1,y2) in enumerate(dataiter_tra):
#         #move to GPU
#
#         x=x.to(device)
#         y1=y1.to(device)
#         y2=y2.to(device)
#
#         # target = y2.torch.empty(batch_size, dtype=torch.long).random_(classes)
#         # clear the gradients of all oprtimized variables
#         optimizer.zero_grad()
#         output1,output2 = model(x)
#         # print("output2:",output2)
#         train_loss1 = loss_fun1(output1, y1)
#         # train_loss2=torch.nn.functional.cross_entropy(output2,y2)
#         train_loss2 = loss2(output2, y2)
#         train_loss2=torch.mean(train_loss2)
#         train_loss_all=train_loss1+train_loss2
#         train_loss_all.backward()
#         optimizer.step()
#         train_loss+=float(train_loss_all.item())
#         tra_loss.append(train_loss)
#         pred=output2.argmax(dim=1)
#         y2=y2.argmax(dim=1)
#         num_correct+=torch.eq(pred,y2).sum().float().item()
#         # print("num:",len(dataiter_tra))
#         num=len(dataiter_tra)*600
#         pass
#     print("Train Epoch:{}\t Loss:{:.4f}\t Acc:{:.4f}\t".format(epoch,train_loss/num,num_correct/num))
#     model.eval()
#     for step,(x1,y1,y2) in enumerate(dataiter_val):
#         #move to GPU
#         x1=x1.to(device)
#         y1=y1.to(device)
#         y2 = y2.to(device)
#         # target = y2.torch.empty(batch_size, dtype=torch.long).random_(classes)
#         validata_out1,validata_out2 = model(x1)
#         validata_loss1 = loss_fun1(validata_out1, y1)
#         # validata_loss2=torch.nn.functional.cross_entropy(validata_out2,y2)
#         validata_loss2 = loss2(validata_out2, y2)
#         validata_loss2=torch.mean(validata_loss2)
#         validata_loss=validata_loss1+validata_loss2
#         val_loss += float(validata_loss.item())
#         if epoch%50==0:
#             y1=y1.cpu()
#             y2=y2.cpu()
#             validata_out1=validata_out1.cpu()
#             validata_out2 = validata_out2.cpu()
#             validata_loss=validata_loss.cpu()
#             r_2 = r2_score(y1.detach(), validata_out1.detach())
#             val_pred = validata_out2.argmax(dim=1)
#             y2=y2.argmax(dim=1)
#             accuracy += torch.eq(val_pred, y2).sum().float().item()
#             # val_loss.append(validata_loss.detach())
#             r_2_1.append(r_2)
#     num1 = len(dataiter_val) * 600
#     # val_loss1=np.mean(val_loss)
#     # val_loss_total.append(val_loss1)
#     r_2=np.mean(r_2_1)
#     r_2_total.append(r_2)
#     if epoch % 50 == 0:
#         print("epoch:{},验证损失:{:.4f},绝对系数:{:.4f},accuracy:{:.4f}".format(epoch,val_loss/num1, r_2,accuracy/num1))
# torch.save(model,"model-big/model_attention.pkl")
#
#
# #测试
# test_accuracy=0
# test_lable1=test_lable1.to(device)
# test_lable2=test_lable2.to(device)
# test_data=test_data.to(device)
# test_out1,test_out2 = model(test_data)
# test_pred=test_out2.argmax(dim=1)
# test_lable1=test_lable1.cpu()
# test_lable2=test_lable2.argmax(dim=1).cpu()
# test_pred=test_pred.cpu()
# test_out1=test_out1.cpu()
# r_2_test = r2_score(test_lable1.detach(), test_out1.detach())
# test_accuracy += torch.eq(test_pred, test_lable2).sum().float().item()
# print("test_accuracy:{:.2f}\t r_2_test:{:.4f}".format(test_accuracy/100,r_2_test))
#
# # test_out1=test_out1.detach().numpy()
# test_pred=test_pred.detach()
# test_lable1=mm.inverse_transform(test_lable1.detach().numpy())
# test_out1=mm.inverse_transform(test_out1.detach().numpy())
# # test_lable=ss.inverse_transform(test_lable.detach().numpy())
# # test_out=ss.inverse_transform(test_out.detach().numpy())
# # pred_inc_angle=test_out1[:,0]*10
# # pred_azi_angle1=test_out1[:,1]*100
# # pred_rad=test_out1[:,2]*100
# # pred_azi_angle2=test_out1[:,3]*100
# # pred_azi_angle3=test_out1[:,4]*100
# # pred_times=test_pred+1
# #
# # True_inc_angle=test_lable1[:,0]*10
# # True_azi_angle1=test_lable1[:,1]*100
# # True_rad=test_lable1[:,2]*100
# # True_azi_angle2=test_lable1[:,3]*100
# # True_azi_angle3=test_lable1[:,4]*100
# # True_times=test_lable2+1
#
# # # test_data=test_data.cpu()
# # test_out=test_out.cpu()
# # pred_inc_angle=test_out[:,0].detach()
# # pred_azi_angle1=test_out[:,1].detach()
# # pred_rad=test_out[:,2].detach()
# # pred_times=test_out[:,3].detach()
# # pred_azi_angle2=test_out[:,4].detach()
# # pred_azi_angle3=test_out[:,5].detach()
# # True_inc_angle=test_lable[:,0].detach()
# # True_azi_angle1=test_lable[:,1].detach()
# # True_rad=test_lable[:,2].detach()
# # True_times=test_lable[:,3].detach()
# # True_azi_angle2=test_lable[:,4].detach()
# # True_azi_angle3=test_lable[:,5].detach()
#
# True_times=test_lable2+1
# pred_times=test_pred+1
# pred_inc_angle=test_out1[:,0]
# pred_azi_angle1=test_out1[:,1]
# pred_rad=test_out1[:,2]
# pred_azi_angle2=test_out1[:,3]
# pred_azi_angle3=test_out1[:,4]
#
# True_inc_angle=test_lable1[:,0]
# True_azi_angle1=test_lable1[:,1]
# True_rad=test_lable1[:,2]
# True_azi_angle2=test_lable1[:,3]
# True_azi_angle3=test_lable1[:,4]
#
# val_loss1=np.ravel(val_loss)
# r_2_total1=np.ravel(r_2_total)
# # #对沉积次数四舍五入取整保存 numpy数据
# # pred_rad=np.around(pred_rad)
# # pred_dep_times=np.around(pred_times)
# #
# # #对沉积次数四舍五入取整保存
# # # pred_rad=torch.round_(pred_rad)
# # # pred_dep_times=torch.round_(pred_times)
# #保存参数
# np.savetxt("F:/zjl/attention-cnn/result1/loss-2.txt", val_loss1, fmt="%f")
# np.savetxt("F:/zjl/attention-cnn/result1/r_2-2.txt", r_2_total1, fmt="%f")
# np.savetxt("F:/zjl/attention-cnn/result1/pred_inc_angle-2.txt", pred_inc_angle, fmt="%d")
# np.savetxt("F:/zjl/attention-cnn/result1/true_inc_angle-2.txt", True_inc_angle, fmt="%d")
# np.savetxt("F:/zjl/attention-cnn/result1/pred_azi_angle1-2.txt", pred_azi_angle1, fmt="%d")
# np.savetxt("F:/zjl/attention-cnn/result1/true_azi_angle1-2.txt", True_azi_angle1, fmt="%d")
# np.savetxt("F:/zjl/attention-cnn/result1/pred_azi_angle2-2.txt", pred_azi_angle2, fmt="%d")
# np.savetxt("F:/zjl/attention-cnn/result1/true_azi_angle2-2.txt", True_azi_angle2, fmt="%d")
# np.savetxt("F:/zjl/attention-cnn/result1/pred_azi_angle3-2.txt", pred_azi_angle3, fmt="%d")
# np.savetxt("F:/zjl/attention-cnn/result1/true_azi_angle3-2.txt", True_azi_angle3, fmt="%d")
# np.savetxt("F:/zjl/attention-cnn/result1/pred_rad-2.txt", pred_rad, fmt="%d")
# np.savetxt("F:/zjl/attention-cnn/result1/true_rad-2.txt", True_rad, fmt="%d")
# np.savetxt("F:/zjl/attention-cnn/result1/pred_times-2.txt", pred_times, fmt="%d")
# np.savetxt("F:/zjl/attention-cnn/result1/true_times-2.txt", True_times, fmt="%d")
#
# #画图
# plt.title("测试")
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.xlabel("样本数")
# plt.ylabel("结构参数")
# plt.subplot(3, 2, 1)
# plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_out1[0:100][:, 0], label='入射角预测')
# plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_lable1[0:100][:, 0], label='入射角真实')
# plt.legend()
# plt.subplot(3, 2, 2)
# plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_out1[0:100][:, 1], label='方位角1预测')
# plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_lable1[0:100][:, 1], label='方位角1真实')
# plt.legend()
# plt.subplot(3, 2, 3)
# plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_out1[0:100][:, 2], label='半径预测')
# plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_lable1[0:100][:, 2], label='半径真实')
# plt.legend()
# plt.subplot(3, 2, 4)
# plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), pred_times[0:100], label='次数预测')
# plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), True_times[0:100], label='次数真实')
# plt.legend()
# plt.subplot(3, 2, 5)
# plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_out1[0:100][:, 3], label='方位角2预测')
# plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_lable1[0:100][:, 3], label='方位角2真实')
# plt.legend()
# plt.subplot(3, 2, 6)
# plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_out1[0:100][:, 4], label='方位角3预测')
# plt.plot(np.array([i for i in range(1, 101)]).reshape(100, 1), test_lable1[0:100][:,4], label='方位角3真实')
# plt.legend()
# plt.show()
