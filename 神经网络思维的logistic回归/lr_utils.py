import numpy as np
import h5py
import matplotlib.pyplot as plt

    
def load_dataset():
    train_dataset = h5py.File('/Users/kyt/kyt/DL/神经网络思维的logistic回归/datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # 训练集里面的图像数据
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # 保存的是训练机的图像对应的分类值，0表示不是猫，1表示是猫

    test_dataset = h5py.File('/Users/kyt/kyt/DL/神经网络思维的logistic回归/datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
'''
print('==========test lr_utils.py file==========')
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()

plt.imshow(train_set_x_orig[1])
print(train_set_x_orig.shape)
print(train_set_y_orig.shape)
plt.show()
'''