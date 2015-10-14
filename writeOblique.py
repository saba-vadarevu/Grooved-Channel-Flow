
import numpy as np
import glob
from myUtils import *
theta = np.arange(15,90,15)
print(theta)

fileStr = "runCases_dragE*.m"
for T in theta:
    for mFile in glob.glob(fileStr):
        fileID = mFile[-6:-2]  # This is the part that reads E0G4, etc..
        fOld = mFile
        fNew = "runCases_dragT"+str(T)+fileID+".m"
        # Changing name of matlab function:
        # Line to be changed: "function [] = runCases_dragE0G1(fileind)"
        functionNameStr = "function [] = runCases_drag"  +"T"+str(T)+ fileID +   "(fileind)"
        changeLine(fOld=fOld, fNew=fNew, strOld="function []", strNew=functionNameStr)
        
        # Changing name of output .mat files:
        # Line to be changed: "fname = ['data_dragE0G1_',num2str(fileind),'.mat'];"
        matFileStr = "fname = ['data_drag" +"T"+str(T)+ fileID+ "_',num2str(fileind),'.mat'];"
        changeLine(fOld=fNew, fNew=fNew, strOld="fname = ", strNew=matFileStr, deleteOld=True)
        
        # Adding a line to allow for non-zero beta:
        # Need to append it after line '        a = g/eps;'
        extraLine = "        a = g/eps;"+"\n"+ \
                    "        b = tan("  +str(T)+  "/180*pi)*a;"
        changeLine(fOld=fNew, fNew=fNew, strOld="        a = g/eps;", strNew=extraLine)
    