import struct
import numpy as np
import random
import math
import os

from numpy.ma.core import reshape, squeeze


import glob
import matplotlib.pyplot as plt
import datetime

from scipy.signal import butter,lfilter
from scipy import signal
import copy as cp
from scipy.signal import hilbert
#102.4
#定义两个归一化函数，用于将数据缩放到特定范围内。
def norm_sgy(data):
    maxval = max(max(data))
    minval = min(min(data))
    return [[(float(i)-minval)/(float(maxval-minval)+0.01e-100) for i in j] for j in data]
def norm_sgy1(data):
    maxval=max([max([abs(i) for i in j]) for j in data]) #获取数据最大幅值
    return [[float(i)/(maxval+0.01e-100) for i in j] for j in data] #将数据归一化到（-1，1）
#定义事件级别的归一化函数。
def norm_1event(data):
    maxval=[max([max([abs(i[0]),abs(i[1]),abs(i[2])]) for i in j]) for j in data]
    return [[[float(data[j][i][0])/(maxval[j]+0.01e-100),float(data[j][i][1])/(maxval[j]+0.01e-100),float(data[j][i][2])/(maxval[j]+0.01e-100)] for i in range(len(data[j]))] for j in range(len(data))]
def norm_1event1(data):
    maxval=max([max([max([abs(i[0]),abs(i[1]),abs(i[2])]) for i in j]) for j in data])
    return [[[float(i[0])/(maxval+0.01e-100),float(i[1])/(maxval+0.01e-100),float(i[2])/(maxval+0.01e-100)] for i in j] for j in data]
def norm_all_event(data):
    return [norm_1event(i) for i in data];
#定义一个函数来读取 .sgy 文件，并返回样本数量、采样点数量和数据
def read_sgy(sgynam):
    try:
        binsgy = open(sgynam, 'rb')
    except IOError:
        return 0, 0, []
    fhead = binsgy.read(3600)
    nr = struct.unpack(">H", fhead[3212:3214])
    nsmp = struct.unpack(">H", fhead[3220:3222])
    data = []
    for ir in range(0, nr[0]):
        trchead = binsgy.read(240)
        trcdata = binsgy.read(nsmp[0]*4)
        data1 = []
        for i in range(0, nsmp[0]):
          # print(trcdata[i*4:i*4+4])
            data1 = data1+list(struct.unpack(">f", trcdata[i*4:i*4+4]))
        data.append(data1)
    #print("read 1sgy end")
    binsgy.close()
    return nr, nsmp, data;
def read_npz(npzname):
    """
    读取 .npz 文件，返回格式与 read_sgy 一致：
    nr   -> tuple: (nr,)
    nsmp -> tuple: (nsmp,)
    data -> list[list[float]]
    """
    try:
        npz = np.load(npzname)
    except IOError:
        print(f"无法打开文件: {npzname}")
        return 0, 0, []

    data_array = npz["data"]   # shape: (nr, nsmp)
    nr_val = int(npz["nr"])
    nsmp_val = int(npz["nsmp"])

    # 转换为和 read_sgy 一样的 list 格式
    data = data_array.tolist()

    # 保持与你原来 read_sgy 一致的返回类型（tuple）
    nr = (nr_val,)
    nsmp = (nsmp_val,)

    return nr, nsmp, data
#构造位置图像，基于给定的位置和范围参数。

def loca_img_xyz(xr,yr,zr,xyz,r,rtz):
    img = []
    # rtz=(100.0/12.0)**2;
    # rtz=1.0;
    for i in range(0, xr[2]):
        x = xr[0]+xr[1]*i
        tmp1 = []
        for j in range(0, yr[2]):
            y = yr[0]+yr[1]*j
            tmp2 = []
            for k in range(0, zr[2]):
                z = zr[0]+zr[1]*k
                ftmp=(x-xyz[0])*(x-xyz[0])+(y-xyz[1])*(y-xyz[1])+rtz*(z-xyz[2])*(z-xyz[2])
                tmp2 = tmp2+[math.exp(-0.5*ftmp/r)]
            tmp1.append(tmp2)
        img.append(tmp1)
    return img
#xr=[0.20,0.005,72],yr=[-0.15,0.005,96],zr=[3.05,0.005,30], r=(0.03)**2, rtz=(15.0/12.0)**2 ,xyr=[]  r=(0.03)**2, rtz=(15.0/12.0)**2
#r = [[0.190, 0.0035, 108], [-0.220, 0.0034, 128], [3.010, 0.0037, 64]]  xyr=[0.190,0.0035,128]  0.190-0.474
def loca_img_xyz1(x,y,z,xyz,r):
    img1 = []
    img2 = []
    img3 = []
    for j in range(0, x[2]):
        xx = x[0]+x[1]*j
        tmp2 = []
        ftmp2=(xx-xyz[0])*(xx-xyz[0])
        tmp2 = tmp2+[math.exp(-0.5*ftmp2/r)]
        #tmp1.append(tmp1)
        img2.append(tmp2)
    img2 = np.array(img2)
    img2 = img2.reshape(144,1,1)

    for i in range(0, y[2]):
        yy = y[0]+y[1]*i
        tmp1 = []
        ftmp2=(yy-xyz[1])*(yy-xyz[1])
        tmp1 = tmp1+[math.exp(-0.5*ftmp2/r)]
        #tmp1.append(tmp1)
        img1.append(tmp1)
    img1 = np.array(img1)
    img1 = img1.reshape(144,1,1)

    for k in range(0, z[2]):
        zz = z[0]+z[1]*k
        tmp3 = []
        ftmp3=(zz-xyz[2])*(zz-xyz[2])
        tmp3 = tmp3+[math.exp(-0.5*ftmp3/r)]
        #tmp1.append(tmp1)
        img3.append(tmp3)
    img3 = np.array(img3)
    img3 = img3.reshape(144,1,1)
    img = np.concatenate([img2, img1, img3], axis=2)  # 沿最后一个轴（通道轴）拼接

    img = img.tolist()

    return img

#洗牌函数，用于随机打乱数据和标签的顺序
def shuffle_data(data, ydata, seed, shuffle):
    if shuffle == 'false':
        return data, ydata
    else:
        index=[i for i in range(len(ydata))]
        random.seed(seed)
        random.shuffle(index)
        data = [data[i] for i in index]
        ydata = [ydata[i] for i in index]
        return data,ydata

def butter_bandpass(lowcut,highcut,sample_rate,order=4):
    rate = sample_rate * 0.5
    low = lowcut /rate
    high = highcut /rate
    b,a = signal.butter(order,[low,high],btype='bandpass',analog=False)
    return b,a
#函数最终返回处理后的地震道数据（data）和相应的标签或图像（ydata），这些数据现在已准备好用于训练机器学习或深度学习模型。受位置，间隔，数量
def load_sgylist_xyz1(sgylist,sgyr=[0,10,1], x=[0.190,0.0035,144],y=[-0.190,0.0035,144],z=[3.000, 0.0035, 108],  r=(0.037) ** 2,
                      shuffle='false', shiftdata=[list(range(-5, 2)), 0]):
    # nx,ny,stn=read_stn(stnnam)
    with open(sgylist[1], 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()  # readlines() 函数用于读取文件中的所有行
    #lines = lines[0:-1:1]+[lines[-1]]; #列表切片，0开始索引位置，-1终止位置索引，1切片的步长。将所有信息存放在列表
    lines=lines[sgyr[0]:sgyr[1]:sgyr[2]]+[lines[sgyr[1]]]   #前11行
    data = []
    ydata = []
    for i in range(0, len(lines)):
        line1 = lines[i].strip().split() #strip()表示删除掉数据中的换行符，split（‘’）则是数据中遇到‘ ’ 就隔开。
        name = os.path.splitext(line1[0].strip())[0]

        sgynam = sgylist[0]+name+'.npz' #输出路径
        loca = [float(num) for num in line1[1:4]]
        #print(loca)
        img = loca_img_xyz1(xyz=[loca[0], loca[1], loca[2]],x=x,y=y,z=z, r=r) #构造标签


        nr,nsmp,data1 = read_npz(sgynam)
    #    print(len(data1))
        if len(data1)<36:
            continue
        b, a = butter_bandpass(1,200,1000,order=4)#4阶带通滤波器，允许1Hz到200Hz之间的信号通过，假设信号的采样率为1000Hz。

        data1 = signal.filtfilt(b, a, data1)## 现在 data1 包含了滤波后的信号
        data1=np.nan_to_num(np.array(data1)).tolist()  ## 将NaN替换为0，并将结果转换为列表
        if nr!= 1:
            data1 = norm_sgy1(data1)  #归一化
            data1=[[[data1[ir][j],data1[ir+1][j],data1[ir+2][j]] for j in range(nsmp[0])] for ir in range(0,nr[0],3)] #取3x512
            data.append(data1)
            ydata.append(img)
        else:
            print('1 event sgy not found')
    if shiftdata[1]>0:
        data1,ydata1=augment_data2(data=data,ydata=ydata,shiftdata=shiftdata)
        data=data+data1
        ydata=ydata+ydata1
    data,ydata=shuffle_data(data,ydata,1,shuffle)
    data=np.array(data)
    ydata=np.array(ydata)
    print(data.shape)
    print(ydata.shape)
    return data,ydata
#加载 .sgy 文件列表，读取文件内容，并进行一系列处理，包括数据增强、归一化等
def augment_data2(data=[],ydata=[],shiftdata=[list(range(20,50))+list(range(-200,-20)),0]):
   #     data_out,ydata_out = cut_trace(icut=par[0],data=data,ydata=ydata);
        data1=[]
        ydata1=[]
        # print(data[0][0])
        nsmp=len(data[0][0])
        # print(data[0])
        for i in range(len(ydata)):
            random.seed(i)
            its=random.sample(shiftdata[0],shiftdata[1])
            for j in range(0,len(its)):
                if its[j]<0:
                   data_tmp=[ftmp[nsmp+its[j]:]+ftmp[0:nsmp+its[j]] for ftmp in data[i]]
                else:
                   data_tmp=[ftmp[its[j]:]+ftmp[0:+its[j]] for ftmp in data[i]]
                ydata_tmp=ydata[i]
                data1=data1+[data_tmp]
                ydata1=ydata1+[ydata_tmp]
        return data1,ydata1
#数据增强函数，通过平移数据来生成新的样本。
def plot_ntrc(data):
    dmax=data.max()
    #print(dmax)
    for i,data1 in enumerate(data):
        #print(i)
        #print(data1[:])
        data1[:]=data1[:]/np.max(data1[:]) #data1[:]列表全部      归一化
        plt.plot(data1[:]+i*dmax)
def plot_all(data):
    for i in range(len(data)):#
        plt.figure()
        plot_ntrc(data[i][:,:,:])
        name_list=list(range(1,50))
        name_list=[str(i) for i in name_list]
        mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')#datetime.datetime.now()获取当前日期
        figure_save_path = "waveform"
        if not os.path.exists(figure_save_path):  #判断目录是否存在
            os.makedirs(figure_save_path) #创建多级目录
        plt.savefig(os.path.join(figure_save_path ,  name_list[i]+'-'+mkfile_time))  #os.path.join（）拼接路径
def convert_sgy_folder_to_npz(input_folder="train1", output_folder="train_new"):
    """
    遍历 input_folder 下所有 .sgy 文件，
    使用 read_sgy 读取数据，并将 data 保存为 .npz 到 output_folder。
    """
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 找到所有 .sgy 文件
    sgy_files = glob.glob(os.path.join(input_folder, "*.sgy"))

    if not sgy_files:
        print(f"在 {input_folder} 中没有找到 .sgy 文件。")
        return

    for sgy_path in sgy_files:
        print(f"正在处理: {sgy_path}")

        nr, nsmp, data = read_sgy(sgy_path)
        if nr == 0 or nsmp == 0 or len(data) == 0:
            print(f"读取失败，跳过: {sgy_path}")
            continue

        # 转成 numpy 数组 (nr, nsmp)
        data_array = np.array(data, dtype=np.float32)

        # 输出文件名，用原始文件名去掉后缀，加 .npz
        base_name = os.path.basename(sgy_path)
        file_stem = os.path.splitext(base_name)[0]
        out_path = os.path.join(output_folder, file_stem + ".npz")

        # 保存为 .npz 文件
        # 可以保存一些基本信息，方便后面使用
        np.savez_compressed(
            out_path,
            data=data_array,
            nr=nr,
            nsmp=nsmp
        )

        print(f"已保存为: {out_path}")

    print("全部转换完成！")


if __name__ == "__main__":
    # 根据你的实际路径改一下这两个参数也可以，例如：
    # convert_sgy_folder_to_npz(r"D:\data\train1", r"D:\data\train_new")
    convert_sgy_folder_to_npz("train1", "train_new")
# if __name__ == '__main__':
#     import scipy.io as sio

   #  #r = [ [0.190, 0.0035, 108],[3.010, 0.0037, 64]]
   #  r = [[-0.200,0.0037,108]]
   # # xyr = [0.190, 0.0035, 128]
   #  data,ydata=load_sgylist_xyz1(sgylist=['./output/','test1.txt'],sgyr=[0,3,1], x=[0.190,0.0035,144],y=[-0.190,0.0035,144],z=[3.000, 0.0035, 144],  r=(0.035) ** 2,
   #                    shuffle='false', shiftdata=[list(range(-5, 2)), 1])
   #  print(data.shape)
   #  #arr_3d = ydata.reshape(1, 128, 64)
   #  print(ydata.shape)
   #  bq = squeeze(ydata)
   #  print(bq.shape)
   #  for i in range(len(bq)):
   #      plt.plot(bq[i,:,1], label=f"Row {i}")  # 绘制第i行
   #      plt.show()


    #a = ydata((1,:,1,1)
    #sio.savemat('./data.mat',{'data':data})
