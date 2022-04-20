from tkinter import *
from tkinter.messagebox import *
import tkinter as tk
from MainPage2 import *

def request():
    def look():

        top.destroy()
    money1 = StringVar
    top = tk.Tk()
    top.title('订单')
    top.geometry('%dx%d' % (300, 100))
    Label(top, text='你有新消息，是否要查看详情？ ').grid(row=2, pady=10,padx=10)
    Button(top, text='确认', font=('Arial', 9), width=8, height=1,command = look).grid(row=3, column=0,stick=W, pady=10,padx=10)
    Button(top, text='取消', font=('Arial', 9), width=8, height=1,command = top.destroy).grid(row=3, column=1,stick=E, pady=10,padx=10)
    mainloop()


class InputFrame(Frame):  # 继承Frame类
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.root = master  # 定义内部变量root
        self.itemName = StringVar()
        self.importPrice = StringVar()
        self.sellPrice = StringVar()
        self.deductPrice = StringVar()
        self.createPage()

    def createPage(self):
        Label(self, text='新增订单',pady=10,padx=10).pack()
        li = ['辣椒酱  100件', '巧克力酱  50件', '食盐  100件', '味精  123件', '面包 100件', '牛奶 123件' , '牛栏山 50件']
        listb = Listbox(self,width= 150)  # 创建两个列表组件
        for item in li:  # 第一个小部件插入数据
            listb.insert(0, item)
        listb.pack()
        lb = Listbox(self,width=150)
        # 双击命令
        lb.bind('<Double-Button-1>')
        for i in ['python', 'C++', 'C', 'Java', 'Php']:
            lb.insert(END, i)
        lb.pack()


class QueryFrame(Frame):  # 继承Frame类
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.root = master  # 定义内部变量root
        self.itemName = StringVar()
        self.sellPrice = StringVar()
        self.number = StringVar()
        self.sell = StringVar()
        self.createPage()

    def createPage(self):
        Label(self).grid(row=0, stick=W, pady=10)
        Label(self, text='商品名称: ').grid(row=1, stick=W, pady=10)
        Entry(self, textvariable=self.itemName).grid(row=1, column=1, stick=E)
        Label(self, text='商品品牌: ').grid(row=2, stick=W, pady=10)
        Entry(self, textvariable=self.sellPrice).grid(row=2, column=1, stick=E)
        Label(self, text='商品数量: ').grid(row=3, stick=W, pady=10)
        Entry(self, textvariable=self.number).grid(row=3, column=1, stick=E)
        Label(self, text='进货商账号: ').grid(row=4, stick=W, pady=10)
        Entry(self, textvariable=self.sell).grid(row=4, column=1, stick=E)
        Button(self, text='确认').grid(row=6, column=1, stick=E, pady=10)






