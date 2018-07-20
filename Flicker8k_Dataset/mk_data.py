#coding=utf-8



import os
import jieba
import shutil


dirs = []
for d in os.listdir(os.getcwd()):
    if os.path.isfile(d):
        continue
    dirs.append(d)
imgs = []
with open("data.txt",'w') as fw , open("imgs.txt","w") as fi:
    for d in dirs:
        for f in os.listdir(os.path.join(d)):
            fi.write("%s\n" % f)
            shutil.move(os.path.join(d,f),f)
            fw.write("%s#0\t%s\n" % (f," ".join(jieba.cut(d))) )
