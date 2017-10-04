# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 15:43:16 2016

@author: Administrator
"""

#1. 编写Python自定义函数，求解输入的年份是否是闰年 
def tmp1(year):
    if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
        print(str(year)+u"是润年")
    else:
        print(str(year)+u"不是润年")

print tmp1(2016)
print tmp1(2000)
print tmp1(1900)

#2. 编写Python自定义函数，输入三个整数x,y,z，请把这三个数由小到大输出。 
def tmp2(x,y,z):
    a=[x,y,z]
    a.sort()
    print a
    
tmp2(3,4,1)

#3.求解1！+2！+...+10! 的结果。（其中！表示阶乘运算，x!=1*2*...*x） 
a=range(11)[1:]
sum=0
for i in a:
    tmp=1
    for j in range(i+1)[1:]:
        tmp=tmp*j
    sum=sum+tmp
print sum