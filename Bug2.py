# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 15:21:24 2021

@author: wyhboos
"""

import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import copy
#读取图片并进行二值化，返回二值化的结果
def Read_map():
    low=10#下边界
    img = cv2.imread('map7.jpg',cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('map',img)
    # cv2.waitKey(1)
    ret, img_th = cv2.threshold(img,low,255, cv2.THRESH_BINARY)
    return img_th


class BUG2():
    def __init__(self,map_binary,start,target):
        self.map=map_binary
        self.start=start
        self.target=target
        
        self.obstacle=[]
        self.radar_info=[]
        self.path=[]
        self.target_theta=None #既定的朝目标的角度
        self.current_pose=self.start
        self.next_theta=None#下一步的角度
    
        self.step_len=1 #模拟步长
        self.state=0 #行走状态，0代表直线走到目标，1代表逆时针环绕障碍物，2代表顺时针环绕障碍物
        self.theta_res=360 #雷达分辨率（360°平分的范围）
        self.radar_range=50 #雷达探测距离
        self.avoid_dis=40 #绕行时离障碍物的距离
        self.follow_p=0.010 #绕行时的比例控制
        self.limit_try=3000 #最大循环次数
    
    def get_path(self):
        self.target_theta=self.compute_theta(self.start,self.target)
        cnt=0
        while True:
            self.compute_obstacle()#更新新区块障碍物信息
            self.get_radar_info()#更新雷达信息
            self.get_next_theta()#获得下一步角度
            next_pose_x=self.current_pose[0]+math.cos(self.next_theta)*self.step_len
            next_pose_y=self.current_pose[1]+math.sin(self.next_theta)*self.step_len
            self.path.append([next_pose_x,next_pose_y])
            self.current_pose=[next_pose_x,next_pose_y]
            cnt+=1
            if cnt>=self.limit_try:
                print('BUG2 failed!')
                return self.path
            if abs(self.current_pose[0]-self.target[0])<5 and abs(self.current_pose[1]-self.target[1])<5:
                print('BUG2 succeeded!')
                return self.path
        
            
    def get_next_theta(self):
        self.target_theta=self.compute_theta(self.start,self.target)
        theta_360=(self.target_theta/math.pi)*180#角度制
        index=int(theta_360*self.theta_res/360)#对应的雷达信息的下标
        dis=self.radar_info[index]#此方向上障碍物的距离
        #小于避障距离，开始环绕
        if dis<self.avoid_dis:
            if self.state==0:
                self.state=1
                #计算开始环绕的距离
                self.obs_follow_dis_start=((self.current_pose[0]-self.target[0])**2+(self.current_pose[1]-self.target[1])**2)**0.5
        
        #判断是否回到原来的状态
        if self.state==1:
            target_theta=self.compute_theta(self.current_pose,self.target)#计算到目标的角度
            target_dis=((self.current_pose[0]-self.target[0])**2+(self.current_pose[1]-self.target[1])**2)**0.5
            if ((target_theta<self.target_theta*1.02 and target_theta>self.target_theta*0.98) and#回到原来的直线方向
                (target_dis<self.obs_follow_dis_start)and
                dis>self.avoid_dis):#距离更近
                self.state=0#回到直线状态
            
        #逆时针环绕，左侧距离保持定值
        # if self.state==1:
        #     theta_left_360=(((self.next_theta+0.5*math.pi)/math.pi)*180)%360
        #     index_left=int(theta_left_360*self.theta_res/360)

        #     dis_left=self.radar_info[index_left]
        #     #使用P控制沿墙
        #     error=dis_left-self.avoid_dis
        #     p_control=self.follow_p*error
        #     self.next_theta=((((self.next_theta+p_control)/math.pi*180)+360)%360)*math.pi/180#限定在0-360°内
        #     return
        
        #针对上面的修改，左侧最近值保持定值（90°张角）
        if self.state==1:
            theta_left_s_360=(((self.next_theta+0.25*math.pi)/math.pi)*180)%360
            theta_left_e_360=(((self.next_theta+0.75*math.pi)/math.pi)*180)%360
            self.index_left_s=int(theta_left_s_360*self.theta_res/360)
            self.index_left_e=int(theta_left_e_360*self.theta_res/360)
            
            if self.index_left_e>=self.index_left_s:
                dis_left=min(self.radar_info[self.index_left_s:self.index_left_e])
            else:
                dis_left=min(self.radar_info[self.index_left_s:]+self.radar_info[:self.index_left_e])
            #使用P控制沿墙
            error=dis_left-self.avoid_dis
            p_control=self.follow_p*error
            
            #转弯受限
            if p_control>0.5*math.pi:
                p_control=math.pi*0.5
            if p_control<-0.5*math.pi:
                p_control=-math.pi*0.5
                
            #减速
            self.next_theta=((((self.next_theta+p_control)/math.pi*180)+360)%360)*math.pi/180#限定在0-360°内
            return
        
        if self.state==0:
            self.next_theta=self.target_theta
            return 
        

    #计算障碍物,以及与当前位置的距离,与当前位置的夹角
    def compute_obstacle(self):
        self.obstacle.clear()
        for x_r in range(-self.radar_range,self.radar_range):
            for y_r in range(-self.radar_range,self.radar_range):
                x=int(self.current_pose[0])+x_r
                y=int(self.current_pose[1])+y_r

                if x>=0 and y>=0 and x<self.map.shape[0] and y<self.map.shape[1]:#保证边界
                    if self.map[x,y]==0:
                        d=math.sqrt((x_r**2+y_r**2))
                        theta=self.compute_theta(start=[0,0],end=[x_r,y_r])
                        if d<=self.radar_range:
                            self.obstacle.append([x,y,d,theta,int(x),int(y),int(d),int(theta*180/math.pi)])
    
    # def anti_clc_follow_theta(self):
    #     theta_left=self.next_theta+0.5*math.pi
    #     theta_left_360=(((self.next_theta+0.5*math.pi)/math.pi)*180)%360
    #     index_left=int(theta_left_360*self.theta_res/360)
    #     dis_left=self.radar_info[index_left]

    #     theta_left_inc=self.next_theta+0.5*math.pi+0.17
    #     theta_left_inc_360=(((self.next_theta+0.5*math.pi+0.17)/math.pi)*180)%360
    #     index_left_inc=int(theta_left_inc_360*self.theta_res/360)
    #     dis_left_inc=self.radar_info[index_left]
        
    #获取雷达信息
    def get_radar_info(self):
        self.radar_info.clear()
        self.radar_info=[9999 for i in range(self.theta_res)] #雷达信息，9999表示没有探测到障碍物
        for obs in self.obstacle:
            theta_360=obs[3]*180/math.pi#转换为角度制
            dis=obs[2]
            index=int(theta_360*self.theta_res/360)
            if dis<self.radar_info[index]:#取最近的作为雷达信息
                self.radar_info[index]=dis
                 
    def compute_theta(self,start,end):#计算角度，输出的是0到360°的弧度制
        dy=end[1]-start[1]
        dx=end[0]-start[0]
        
        if dx==0:
            if dy>=0:
                theta=0.5*math.pi
            if dy<0:
                theta=1.5*math.pi
            return theta
        
        theta=math.atan(dy/dx)
        if dx<0 and dy<0:
            theta=theta-math.pi
        if dx<0 and dy>0:
            theta=theta+math.pi
            
        if theta<0:
            theta=theta+2*math.pi
            
        return theta

    def show_path(self):
        map_img=self.map
        path=self.path
        for p in path:
            map_img[int(p[0]),int(p[1])]=127
        # a=plt.figure(figsize=(15, 15))
        # plt.imshow(map_img)
        return map_img

# img1=Read_map()
# bug2=BUG2(map_binary=img1,start=[0,0],target=[300,700])
# path1=bug2.get_path()
# img1_with_path=bug2.show_path()

# a=plt.figure(figsize=(50, 50))
# plt.imshow(img1_with_path)
# length1=len(path1)
# plt.title("BUG2 with"+str(length1)+'steps')
# print('--------------------------------------------------------------')
# # print(list(np.array(path1).T))