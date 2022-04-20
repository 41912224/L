import numpy as np
import random
import sys
from PIL import Image
from captcha.image import ImageCaptcha


number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

#生成验证码的随机数字列表，验证码图片中有6个数字
def random_text(charset=number + alphabet + ALPHABET, size=6):
    caplist = []
    for i in range(size):
        captcha_text = random.choice(charset)
        caplist.append(captcha_text)
    return caplist

#生成验证码
def create_captext():
    caplist = random_text()
    caplist = ''.join(caplist)# 将验证码列表转为字符串
    image = ImageCaptcha()
    captcha = image.generate(caplist)# 生成图片
    image.write(caplist, 'E:/test/'+caplist+'.jpg' )

#自动进度条
from tqdm import tqdm
import time
for i in tqdm(range(40000)):
    create_captext()
    time.sleep(0.01)
