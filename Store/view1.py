import tkinter as tk
from tkinter import *
from tkinter.messagebox import *
from MainPage1 import *
import LoginPage
import pymssql
import io
from PIL import Image
import matplotlib.pyplot as plot
def money():
    def save():
        money1 = en.get()
        print(money1)
        top.destroy()
    top = tk.Tk()
    top.title('登陆成功')
    top.geometry('%dx%d' % (300, 100))

    #Label(top, anchor='center', width=10, height=10, text='"欢迎来到无人售货店！"', font=('Arial', 10) ).grid(row=1, stick=W, pady=10)
    label = Label(top, text='请问您带了多少钱呢: ')
    label.grid(row=2, stick=W, pady=10, padx=10)
    en = Entry(top)
    en.grid(row=2, column=1, stick=E)
    # print("欢迎来到无人售货店！")
    bu = Button(top, text='确认', font=('Arial', 9), width=8, height=1,command = save)
    bu.grid(row=3,stick=E, column=1,pady=10,padx=10)
    top.mainloop()
class InputFrame(Frame):  # 继承Frame类
    # 购买提示
    def print_selection(self):
        top = tk.Tk()
        top.title('成功')
        top.geometry('%dx%d' % (200, 100))
        l = tk.Label(top,anchor = 'center', width=10, height=10, text='购买成功', font=('Arial', 10),)
        l.pack()

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.root = master  # 定义内部变量root
        self.itemName = StringVar()
        self.importPrice = StringVar()
        self.sellPrice = StringVar()
        self.deductPrice = StringVar()
        self.createPage()

    def tiao(self):
        top = tk.Tk()
        top.title('调味类')
        top.geometry('%dx%d' % (300, 250))
        li = ['辣椒酱  2021-5', '巧克力酱  2021-6', '食盐  2021-6', '味精  2021-7']
        def CallOn(event):
            root1 = Tk()
            root1.title('成功')
            root1.geometry('%dx%d' % (200, 100))
            Label(root1, text='你购买的是' + lb.get(lb.curselection()),padx=10,pady=15).pack()
            Button(root1, text='退出', command=root1.destroy).pack()
        lb = Listbox(top)
        # 双击命令
        lb.bind('<Double-Button-1>', CallOn)
        for i in li:
            lb.insert(END, i)
        lb.pack()

    def jiu(self):
        top = tk.Tk()
        top.title('酒水类')
        top.geometry('%dx%d' % (300, 250))
        li = ['鸡尾酒  2021-5', '啤酒  2021-5', '白酒  2021-4', '伞兵  2021-5']

        def CallOn(event):
            root1 = Tk()
            root1.title('成功')
            root1.geometry('%dx%d' % (200, 100))
            Label(root1, text='你购买的是' + lb.get(lb.curselection()), padx=10, pady=15).pack()
            Button(root1, text='退出', command=root1.destroy).pack()

        lb = Listbox(top)
        # 双击命令
        lb.bind('<Double-Button-1>', CallOn)
        for i in li:
            lb.insert(END, i)
        lb.pack()
    def drink(self):
        top = tk.Tk()
        top.title('饮料类')
        top.geometry('%dx%d' % (300, 250))
        li = ['可口可乐  2021-4', '百事可乐  2021-4', '牛栏山  2021-6', '白兰地  2011-4']

        def CallOn(event):
            root1 = Tk()
            root1.title('成功')
            root1.geometry('%dx%d' % (200, 100))
            Label(root1, text='你购买的是' + lb.get(lb.curselection()), padx=10, pady=15).pack()
            Button(root1, text='退出', command=root1.destroy).pack()

        lb = Listbox(top)
        # 双击命令
        lb.bind('<Double-Button-1>', CallOn)
        for i in li:
            lb.insert(END, i)
        lb.pack()
    def hong(self):
        top = tk.Tk()
        top.title('烘焙类')
        top.geometry('%dx%d' % (300, 250))
        li = ['饼干  2021-5', '面包  2021-5', '薯片  2021-3', '炸鸡  2021-9']

        def CallOn(event):
            root1 = Tk()
            root1.title('成功')
            root1.geometry('%dx%d' % (200, 100))
            Label(root1, text='你购买的是' + lb.get(lb.curselection()), padx=10, pady=15).pack()
            Button(root1, text='退出', command=root1.destroy).pack()

        lb = Listbox(top)
        # 双击命令
        lb.bind('<Double-Button-1>', CallOn)
        for i in li:
            lb.insert(END, i)
        lb.pack()

    def createPage(self):

        Button(self, text='饮料类', command =self.drink).grid(row=0, column=1, pady=14,padx=14)
        Button(self, text='烘焙类', command =self.hong).grid(row=0, column=2, pady=14,padx=14)
        Button(self, text='调味类', command =self.tiao).grid(row=0, column=3, pady=14,padx=14)
        Button(self, text='酒水类', command =self.jiu).grid(row=0, column=4, pady=14,padx=14)
        Button(self, text='饮料类', command =self.drink).grid(row=1, column=1, pady=14,padx=14)
        Button(self, text='烘焙类', command =self.hong).grid(row=1, column=2, pady=14,padx=14)
        Button(self, text='调味类', command =self.tiao).grid(row=1, column=3, pady=14,padx=14)
        Button(self, text='酒水类', command =self.jiu).grid(row=1, column=4, pady=14,padx=14)
        Button(self, text='饮料类', command =self.drink).grid(row=2, column=1, pady=14,padx=14)
        Button(self, text='烘焙类', command =self.hong).grid(row=2, column=2, pady=14,padx=14)
        Button(self, text='调味类', command =self.tiao).grid(row=2, column=3, pady=14,padx=14)
        Button(self, text='酒水类', command =self.jiu).grid(row=2, column=4, pady=14,padx=14)
        Button(self, text='饮料类', command =self.drink).grid(row=3, column=1, pady=14,padx=14)
        Button(self, text='烘焙类', command =self.hong).grid(row=3, column=2, pady=14,padx=14)
        Button(self, text='调味类', command =self.tiao).grid(row=3, column=3, pady=14,padx=14)
        Button(self, text='酒水类', command =self.jiu).grid(row=3, column=4, pady=14,padx=14)
class QueryFrame(Frame):  # 继承Frame类
    def creat(self):
        b = self.itemName.get()
        d = self.sellPrice.get()
        print(b,d)
        top = tk.Tk()
        top.title('成功')
        top.geometry('%dx%d' % (200, 100))
        l = tk.Label(top, anchor='center', width=10, height=10, text= '具体位置', font=('Arial', 10), )
        l.pack()
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.root = master  # 定义内部变量root
        self.itemName = StringVar()
        self.importPrice = StringVar()
        self.sellPrice = StringVar()
        self.deductPrice = StringVar()
        self.createPage()

    def createPage(self):

        Label(self).grid(row=0, stick=W, pady=10)
        Label(self, text='商品名称: ').grid(row=1, stick=W, pady=10)
        Entry(self, textvariable=self.itemName).grid(row=1, column=1, stick=E)
        Label(self, text='商品品牌: ').grid(row=2, stick=W, pady=10)
        Entry(self, textvariable=self.sellPrice).grid(row=2, column=1, stick=E)
        Button(self, text='查询',command = self.creat).grid(row=6, column=1, stick=E, pady=10)





