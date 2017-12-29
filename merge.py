#-*-coding:utf8-*-
#########################################################################
#   Copyright (C) 2017 All rights reserved.
# 
#   FileName:merge.py
#   Creator: yuliu1@microsoft.com
#   Time:11/15/2017
#   Description:
#
#   Updates:
#
#########################################################################
#!/usr/bin/python
# please add your code here!
import sys;
import re;
def Process(ifile,ofile):
    fout = open(ofile,"w");
    with open(ifile,"r") as f:
        newline="";
        str1="";
        str2="";
        str3="";
        str4="";
        for line in f:
            line = line.strip();
            col = line.split("\t");
            if(len(col)==2):
                subcol = col[0].split("#");
                subcol=subcol+[col[1]];
                if (subcol[2]=="S"):
                    if (str1!=""):
                        str1+="#";
                    if (str2 != ""):
                        str2+="#";
                    if (str3 != ""):
                        str3 += "#";
                    if (str4 != ""):
                        str4 += "#"
                    str1+=subcol[0];
                    str2+=subcol[1];
                    str3+=subcol[2];
                    str4+=subcol[3];
                elif(subcol[2]=="B"):
                    if (str1!=""):
                        str1+="#";
                    if (str2 != ""):
                        str2+="#";
                    if (str3 != ""):
                        str3 += "#";
                    if (str4 != ""):
                        str4 += "#";
                    str1+=subcol[0];
                    str2+=subcol[1];
                    str3+=subcol[2];
                    str4+=subcol[3];
                else:
                    str1+=",";
                    str2+=",";
                    str3+=",";
                    str4+=",";
                    str1+=subcol[0];
                    str2+=subcol[1];
                    str3+=subcol[2];
                    str4+=subcol[3];
            else:
                newline="%s\t%s\t%s\t%s"%(str1,str2,str3,str4);
                fout.write("%s\n"%newline);
                newline="";
                str1="";
                str2="";
                str3="";
                str4="";
    fout.close();
if __name__=="__main__":
    if (len(sys.argv)!=3):
        sys.stderr.write("not enough params");
        sys.exit(1);
    Process(sys.argv[1],sys.argv[2]);

