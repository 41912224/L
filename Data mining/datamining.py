# -*- coding:utf-8 -*-
import numpy as np
from PIL import Image, ImageDraw, ImageFont
# 首先，导入图像分割函数要用到的label(连通区域标记)、regionprops(连续区域抽取)函数
from matplotlib import pyplot as plt
from skimage import transform as tf
from skimage.measure import label, regionprops
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state


# 图像分割函数接收图像，返回小图像列表，每张小图像为单词的一个字母
def segment_image(image):
    # 用scikit-image的label函数检测字母的位置，它能找出图像中像素值相同且相邻的像素块。图像连接在一起的区域用不同的值来表示，在这些区域以外的像素用0来表示。
    labeled_image = label(image > 0)
    # 抽取每一张小图像，将它们保存到一个列表中。
    subimages = []
    # regionprops函数提供抽取连续区域，遍历这些区域，分别处理。
    for region in regionprops(labeled_image):
        start_x, start_y, end_x, end_y = region.bbox
        # 用这两组坐标作为索引就能抽取到小图像（image对象为numpy数组，可以直接用索引值），然后，把它保存到subimages列表中。
        subimages.append(image[start_x:end_x, start_y:end_y])
    # 返回找到的小图像，每张小图像包含单词的一个字母区域。没有找到小图像的情况，直接把原图像作为子图返回。
    if len(subimages) == 0:
        return [image, ]
    return subimages


# 生成验证码
def create_captcha(text, shear=0, size=(100, 24), scale=1):
    # 我们使用字母L来生成一张黑白图像，为`ImageDraw`类初始化一个实例。这样，我们就可以用`PIL`绘图
    im = Image.new("L", size, "black")
    draw = ImageDraw.Draw(im)
    # 指定验证码文字所使用的字体。
    font = ImageFont.truetype("C:\\WINDOWS\\Fonts\\arial.TTF", 22)
    draw.text((0, 0), text, fill=1, font=font)
    # 把PIL图像转换为`numpy`数组，以便用`scikit-image`库为图像添加错切变化效果。`scikit-image`大部分计算都使用`numpy`数组格式。
    image1 = np.array(im)
    affine_tf = tf.AffineTransform(shear=shear)
    image1 = tf.warp(image1, affine_tf)
    # 最后一行代码对图像特征进行归一化处理，确保特征值落在0到1之间。归一化处理可在数据预处理、分类或其他阶段进行
    return image1 / image1.max()


# 生成验证码图像并显示它
image = create_captcha("WEFB", shear=0.2)
plt.imshow(image, cmap='Greys')
plt.savefig('E:/test/1.jpg')
plt.show()
subimages = segment_image(image)
f, axes = plt.subplots(1, len(subimages),
                       figsize=(10, 3))
for i in range(len(subimages)):
    axes[i].imshow(subimages[i], cmap="gray")
plt.show()
# 创建训练集
# 使用图像切割函数就能创建字母数据集，其中字母使用了不同的错切效果，然后就可以训练神经网络分类器来识别图像中的字母。
# 首先，指定随机状态值，创建字母列表，指定错切值。
random_state = check_random_state(14)
letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
shear_values = np.arange(0, 0.5, 0.05)


# 创建一个函数（用来生成一条训练数据），从提供的选项中随机选取字母和错切值。
def generate_sample(random_state=None):
    random_state = check_random_state(random_state)
    letter = random_state.choice(letters)
    shear = random_state.choice(shear_values)
    # 返回字母图像及表示图像中字母属于哪个类别的数值。字母A为类别0，B为类别1，C为类别2，以此类推
    return create_captcha(letter, shear=shear, size=(30, 30)), letters.index(letter)


# 在上述函数体的外面，调用该函数，生成一条训练数据，用pyplot显示图像
image, target = generate_sample(random_state)
plt.imshow(image, cmap="Greys")
# plt.show()
print("The target for this image is: {0}".format(letters[target]))
# 调用几千次该函数，就能生成足够的训练数据。把这些数据传入到numpy的数组里，因为数组操作起来比列表更容易。
dataset, targets = zip(*(generate_sample(random_state) for i in range(2000)))
dataset = np.array([tf.resize(segment_image(sample)[0], (30, 30)) for sample in dataset])
dataset = np.array(dataset, dtype='float')
targets = np.array(targets)
# 共有26个类别，每个类别（字母）用从0到25之间的一个整数表示。多个神经元就能有多个输出，每个输出值在0到1之间。如果结果像某字母，使用   近似于1的值；如果不像，就用近似于0的值。
onehot = OneHotEncoder()
y = onehot.fit_transform(targets.reshape(targets.shape[0], 1))
y = y.todense()
# 训练数据集中每条数据都是一个恰好为20×20的字母。验证码图像中抽取的字母大小不一、图像偏离中心或者引入其他问题。理想情况下，训练分类器所使用的数据应该与分类器即将处理的数据尽可能相似。但实际中，经常有所不同，但是应尽量缩小两者之间的差别。
from skimage.transform import resize

# 对每条数据运行`segment_image`函数，将得到的小图像调整为20×20
dataset = np.array([resize(segment_image(sample)[0], (20, 20)) for sample in dataset])
# 创建数据集。`dataset`数组为三维的，因为它里面存储的是二维图像信息。由于分类器接收的是二维数组，因此，需要将最后两维扁平化
X = dataset.reshape((dataset.shape[0], dataset.shape[1] * dataset.shape[2]))
# 使用`scikit-learn`中的`train_test_split`函数，把数据集切分为训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)

# 构造神经网络分类器，接收图像，预测图像中的单个字母是什么。每张图像有400个特征，将把它们作为神经网络的输入。输出结果为26个0到1之间的值。值越大，表示图像中的字母为该值所对应的字母（输出的第一个值对应字母A，第二个对应字母B，以此类推）的可能性越大。创建一个最基础的、具有三层结构的多层感知机，它由输入层、输出层和一层隐含层组成。输入层和输出层的神经元数量是固定的。现在，数据集有400个特征，那么第一层就需要有400个神经元，而26个可能的类别表明我们需要26个用于输出的神经元。
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(100,), random_state=14)
# 训练与分类
# 导入数据，训练模型。
clf.fit(X_train, y_train)
len(clf.coefs_)
clf.coefs_[0].shape
clf.coefs_[1].shape
fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = clf.coefs_[0].min(), clf.coefs_[0].max()
for coef, ax in zip(clf.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(20, 20), cmap=plt.cm.gray, vmin=.5 * vmin,vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())
plt.show()
# 预测数据。
# 使用F1指标来判断结果好坏。分数越高，说明模型预测越好。
dataset = np.array([tf.resize(subimage, (20, 20)) for subimage in subimages])
X_test = dataset.reshape((dataset.shape[0], dataset.shape[1] * dataset.shape[2]))
y_pred = clf.predict(X_test)
predictions = np.argmax(y_pred, axis=1)
assert len(y_pred) == len(X_test)
predicted_word = str.join("", [letters[prediction] for prediction in predictions])
print(predicted_word)
