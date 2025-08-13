# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:30:59 2024

@author: 27963
"""

###建立kd树和实现查询功能
import numpy as np
import cv2
import matplotlib.pyplot as plt

class kdTree:
    def __init__(self, parent_node):
        '''
        节点初始化
        '''
        self.nodedata = None   #当前节点的数据值，二维数据
        self.split = None #分割平面的方向轴序号,0代表沿着x轴分割，1代表沿着y轴分割
        self.range = None  #分割临界值
        self.left = None    #左子树节点
        self.right = None   #右子树节点
        self.parent = parent_node  #父节点
        self.leftdata = None  #保留左边节点的所有数据
        self.rightdata = None #保留右边节点的所有数据
        self.isinvted = False #记录当前节点是否被访问过

    def print(self):
        '''
        打印当前节点信息
        '''
        print(self.nodedata, self.split, self.range)

    def getSplitAxis(self, all_data):
        '''
        根据方差决定分割轴
        '''
        var_all_data = np.var(all_data, axis=0)
        if var_all_data[0] > var_all_data[1]:
            return 0
        else:
            return 1
    

    def getRange(self, split_axis, all_data):
        '''
        获取对应分割轴上的中位数据值大小
        '''
        split_all_data = all_data[:, split_axis]
        data_count = split_all_data.shape[0]
        med_index = int(data_count/2)
        sort_split_all_data = np.sort(split_all_data)
        range_data = sort_split_all_data[med_index]
        return range_data


    def getNodeLeftRigthData(self, all_data):
        '''
        将数据划分到左子树，右子树以及得到当前节点
        '''
        data_count = all_data.shape[0]
        ls_leftdata = []
        ls_rightdata = []
        for i in range(data_count):
            now_data = all_data[i]
            if now_data[self.split] < self.range:
                ls_leftdata.append(now_data)
            elif now_data[self.split] == self.range and self.nodedata == None:
                self.nodedata = now_data
            else:
                ls_rightdata.append(now_data)
        self.leftdata = np.array(ls_leftdata)
        self.rightdata = np.array(ls_rightdata)


    def createNextNode(self,all_data):
        '''
        迭代创建节点，生成kd树
        '''
        if all_data.shape[0] == 0:
            print("create kd tree finished!")
            return None
        self.split = self.getSplitAxis(all_data)
        self.range = self.getRange(self.split, all_data)
        self.getNodeLeftRigthData(all_data)
        if self.leftdata.shape[0] != 0:
            self.left = kdTree(self)
            self.left.createNextNode(self.leftdata)
        if self.rightdata.shape[0] != 0:
            self.right = kdTree(self)
            self.right.createNextNode(self.rightdata)
        
    
    def divDataToLeftOrRight(self, find_data):
        '''
        根据传入的数据将其分给左节点(0)或右节点(1)
        '''
        data_value = find_data[self.split]
        if data_value < self.range:
            return 0
        else:
            return 1

    def getSearchPath(self, ls_path, find_data):
        '''
        二叉查找到叶节点上
        '''
        now_node = ls_path[-1]
        if now_node == None:
            return ls_path
        now_split = now_node.divDataToLeftOrRight(find_data)
        if now_split == 0:
            next_node = now_node.left
        else:
            next_node = now_node.right
        while(next_node!=None):
            ls_path.append(next_node)
            next_split = next_node.divDataToLeftOrRight(find_data)
            if next_split == 0:
                next_node = next_node.left
            else:
                next_node = next_node.right
        return ls_path
            
    def getNestNode(self, find_data, min_dist, min_data):
        '''
        回溯查找目标点的最近邻距离
        '''
        ls_path = []
        ls_path.append(self)
        self.getSearchPath(ls_path, find_data)
        now_node = ls_path.pop()
        now_node.isinvted = True
        min_data = now_node.nodedata
        min_dist = np.linalg.norm(find_data-min_data)
        while(len(ls_path)!=0):
            back_node = ls_path.pop()   # 向上回溯一个节点
            if back_node.isinvted == True:
                continue
            else:
                back_node.isinvted = True
            back_dist = np.linalg.norm(find_data-back_node.nodedata)
            if back_dist < min_dist:
                min_data = back_node.nodedata
                min_dist = back_dist
            if np.abs(find_data[back_node.split]-back_node.range) < min_dist:
                ls_path.append(back_node)
                if back_node.left.isinvted == True:
                    if back_node.right == None:
                        continue
                    ls_path.append(back_node.right)
                else:
                    if back_node.left == None:
                        continue
                    ls_path.append(back_node.left)
                ls_path = back_node.getSearchPath(ls_path, find_data)
                now_node = ls_path.pop()
                now_node.isinvted = True
                now_dist = np.linalg.norm(find_data-now_node.nodedata)
                if now_dist < min_dist:
                    min_data = now_node.nodedata
                    min_dist = now_dist
        return min_data

def k_means(obs, k, dist=np.median):
    '''
    :param obs: 待观测点
    :param k: 聚类数k
    :param dist: 表征聚类中心函数
    :guess_central_points 中心点
    :return: current_cluster 分类结果
    '''
    obs_num = obs.shape[0]
    obs_dim = obs.shape[1]
    if k < 1:
        raise ValueError("Asked for %d clusters." % k)
    # 随机取中心点
  
    guess_central_points = obs[np.random.choice(obs_num, size=k, replace=False)]  # 初始化最大距离
    last_cluster = np.zeros((obs_num, ))
    current_cluster = np.zeros((obs_num, ))
    # 迭代
    while True:
        for i in range(0,obs_num):
            my_kd_tree = kdTree(None)
            my_kd_tree.createNextNode(guess_central_points)
            min_dist = 0                                 # 临时变量，存储最短距离
            min_data = np.zeros(3)
            closest_point = my_kd_tree.getNestNode(obs[i,:], min_dist, min_data)
            for j in range(0,k):
                whether = 0
                for h in range(0,obs_dim):
                    if guess_central_points[j][h] != closest_point[h]:
                         whether = whether+1    
                if whether == 0:
                    current_cluster[i] = j
        if (last_cluster == current_cluster).all():
            break
 
        # 计算新的中心
        for i in range(k):
            guess_central_points[i] = dist(obs[current_cluster == i], axis=0)
        last_cluster = current_cluster
        
    return current_cluster

def preprocess_image(image):
    # 将图像转换为浮点型，并进行归一化
    normalized_image = image.astype(np.float32)   
    pixel_values = normalized_image.reshape(-1, 3).astype(np.float32)
    return pixel_values


image_path = r'D:\image1.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image.shape)
data = preprocess_image(image)
km_segmentation = k_means(data, 3, dist=np.median).reshape(image.shape[:2])

plt.figure(figsize=(12, 6))

plt.subplot(131)
plt.imshow(image)
plt.title('Original Image')

plt.subplot(132)
plt.imshow(km_segmentation, cmap='viridis')
plt.title(' Segmentation')






