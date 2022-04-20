from tkinter import *
from tkinter.messagebox import *
import tkinter
from MainPage3 import *

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
        Label(self, text='新增订单', pady=10, padx=10).pack()
        li = ['辣椒酱  100件', '巧克力酱  50件', '食盐  100件', '味精  123件', '面包 100件', '牛奶 123件', '牛栏山 50件']

        def CallOn(event):
            def shu():
                def jia():
                    rot1.destroy()
                    lb.delete(ACTIVE)
                rot1 = Tk()
                rot1.title('许可')
                rot1.geometry('%dx%d' % (260, 200))
                Label(rot1, text='生产日期: ', pady=10).pack()
                Entry(rot1, textvariable=self.itemName).pack()
                Label(rot1, text='保质期: ',pady=10).pack()
                Entry(rot1, textvariable=self.sellPrice,).pack()
                Button(rot1, text='确认',command= jia).pack()
                root1.destroy()
            def shan():
                lb.delete(ACTIVE)
                root1.destroy()
            root1 = Tk()
            root1.title('许可')
            root1.geometry('%dx%d' % (260,150))
            Label(root1, text='是否同意此订单:' + lb.get(lb.curselection()), padx=10, pady=15).pack()
            Button(root1, text='同意',command=shu).pack()
            Button(root1, text='拒绝', command=shan).pack()
        lb = Listbox(self, width=150)
        # 双击命令
        lb.bind('<Double-Button-1>', CallOn)
        for i in li:
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
        Label(self, text='历史订单',pady=10,padx=10).pack()
        li = ['辣椒酱  100件', '巧克力酱  50件', '食盐  100件', '味精  123件', '面包 100件', '牛奶 123件' , '牛栏山 50件']
        listb = Listbox(self,width= 150)  # 创建两个列表组件
        for item in li:  # 第一个小部件插入数据
            listb.insert(0, item)
        listb.pack()







