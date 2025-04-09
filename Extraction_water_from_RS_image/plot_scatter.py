# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 20:21:00 2024

@author: jmen
"""
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

def plot_scatter(name,x1,y1,x2,y2,stats_dl,stats_sd,band):
    '''
    x: predict
    y: label
    '''
    min_val = np.amin([x1,y1,x2,y2])
    max_val = np.amax([x1,y1,x2,y2])
    
    x0 = np.array([0.8*min_val,1.2*max_val])
    y0 = np.array([0.8*min_val,1.2*max_val])
    
    # ZX = (mat1-np.mean(mat1))/np.std(mat1)
    # ZY = (mat2-np.mean(mat2))/np.std(mat2)
    # r = np.sum(ZX*ZY)/(len(mat1))
    r2_1 = r2_score(x1,y1)
    rpd_1 = np.mean((x1-y1)/y1)
    r2_2 = r2_score(x2,y2)
    rpd_2 = np.mean((x2-y2)/y2)
    bias_1 = np.mean(y1-x1)
    bias_2 = np.mean(y2-x2)
    
    #创建绘图
    fig = plt.figure(num=1, figsize=(3, 3.5))    #figsize单位为英寸
    ax = plt.subplot(111)
    # 设置字体
    plt.rcParams['axes.unicode_minus'] = False          #使用上标小标小一字号
    plt.rcParams['font.sans-serif']=['Times New Roman'] #设置全局字体，‘SimHei’黑体可现实中文
    font1 = {'family': 'Times New Roman', 
         'color': 'green',
         'weight': 'bold', #wight为字体的粗细，可选 ‘normal\bold\light’等
         'size': 13
         }
    font2 = {'family': 'Times New Roman', 
         'color': 'black',
         'weight': 'bold', 
         'size': 13
        }
    font3 = {'family': 'Times New Roman', 
         'color': 'darkorange',
         'weight': 'bold', #wight为字体的粗细，可选 ‘normal\bold\light’等
         'size': 13
         }
    # plt.rc('font', family='Times New Roman', size=7)    
    #设置x,y轴的风格
    ax.tick_params(axis='x', which='major', bottom=True, top=False, labelbottom=True,
                   direction='inout', labelsize=10, length=10,width=1, colors='black')
    ax.tick_params(axis='x', which='minor', bottom=True, top=False, labelbottom=True,
                   direction='in', labelsize=10, length=5,width=1, colors='black')
    ax.tick_params(axis='y', which='major', left=True, right=False,labelbottom=True,
                   direction='inout', labelsize=10, length=10,width=1, colors='black')
    ax.tick_params(axis='y', which='minor', left=True, right=False,labelbottom=True,
                   direction='in', labelsize=10, length=5,width=1, colors='black')
#        f1 = ax.plot(x, y, marker='o', markersize=1.2, color='blue', linewidth=0.0, linestyle='--')
    plt.xlim(0.8*min_val,1.2*max_val)
    plt.ylim(0.8*min_val,1.2*max_val)
    plt.xticks(weight='bold')
    plt.yticks(weight='bold')
    #绘图
    f0 = ax.plot(x0,y0, color='black',linewidth=1)
    f1 = ax.scatter(x1,y1, s=10, color='green',alpha=0.5)    #x轴为insitu，y轴predict
    f2 = ax.scatter(x2,y2,s=10,color='darkorange',alpha=0.5)
    
    # plt.text(0.05,0.92, 'N       '+chr(963)+'       APD    R      RMSE     bias ',fontdict=font2,transform=ax.transAxes)
    # plt.text(0.05,0.83, '        (%)     (%)               (sr$\mathregular{^{-1}}$)     (sr$\mathregular{^{-1}}$)',fontdict=font2,transform=ax.transAxes)
    # plt.text(0.04,0.76, '%d %6.2f %6.2f %6.2f  %8.4f %8.4f'%(len(x1),stats_dl[0]*100,stats_dl[1]*100,stats_dl[2],stats_dl[3],stats_dl[4]),fontdict=font1,transform=ax.transAxes)
    # plt.text(0.04,0.69, '%d %6.2f %6.2f %6.2f  %8.4f %8.4f'%(len(x2),stats_sd[0]*100,stats_sd[1]*100,stats_sd[2],stats_sd[3],stats_dl[4]),fontdict=font3,transform=ax.transAxes)
    plt.text(0.05,0.92, ' N     APD     R     RMSE',fontdict=font2,transform=ax.transAxes)
    plt.text(0.05,0.83, '          (%)              (sr$\mathregular{^{-1}}$)',fontdict=font2,transform=ax.transAxes)
    plt.text(0.04,0.76, ' %d %6.2f %6.2f %8.4f'%(len(x1),stats_dl[1]*100,stats_dl[2],stats_dl[3]),fontdict=font1,transform=ax.transAxes)
    plt.text(0.04,0.69, ' %d %6.2f %6.2f %8.4f'%(len(x2),stats_sd[1]*100,stats_sd[2],stats_sd[3]),fontdict=font3,transform=ax.transAxes)
    
    # plt.title(name+band)
    # ax.tick_params(axis='y', direction='in', length=3, width=1, colors='black', labelrotation=90)
    #设置图例样式
    ax.legend((f1,f2),(r'DLAC',r'NIR'),loc=4,prop={'size': 10,'weight':'bold'}, frameon=True) 

    #设置坐标名
    ax.set_ylabel(r'Satellite '+r'$\mathregular{R_{rs}}$'+r' ($\mathregular{x10^{2}}$ $\mathregular{sr^{-1}}$)', fontdict=font2)
    ax.set_xlabel(r'In-situ '+r'$\mathregular{R_{rs}}$'+r' ($\mathregular{x10^{2}}$ $\mathregular{sr^{-1}}$)', fontdict=font2)
    plt.minorticks_on()     #开启小坐标
    # ax.xaxis.set_label_coords(0.5, -0.11)
    # ax.tick_params(axis='x', direction='in', length=3, width=1, colors='black', labelrotation=0)
    plt.ticklabel_format(axis='both',style='sci')   #sci文章的风格
    plt.tight_layout(rect=(0,0,1,1))#rect=[left,bottom,right,top]   #设置图框与图片边缘的距离
    figname = r'./' + name +str(band)+'_500dpi.png'
    plt.savefig(figname, dpi=600)

    plt.close()