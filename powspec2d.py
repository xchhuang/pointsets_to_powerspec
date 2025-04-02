"""
Pytorch implementation of 2D fourier power spectrum
The code is translated from the original numpy implementation: https://github.com/sinbag/EmpiricalErrorAnalysis/blob/master/src/analyzer/FourierAnalyzer.cpp
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

# def continuous_fourier_spectrum_2d_np(pts, res=64, freqStep=1):
#     """
#     function:
#         2D fourier power spectrum for a 2D point set
#     input:
#         pts: numpy array, [N, 2]
#     output:
#         out: numpy array, [res, res]
#     """
#     X = pts
#     resolution = res
#     xfreqStep = yfreqStep = freqStep
#     numPoints = X.shape[0]
#     xlow = -resolution * 0.5 * xfreqStep
#     xhigh = resolution * 0.5 * xfreqStep
#     ylow = -resolution * 0.5 * yfreqStep
#     yhigh = resolution * 0.5 * yfreqStep
#     u = np.arange(xlow, xhigh, xfreqStep)
#     v = np.arange(ylow, yhigh, yfreqStep)
#     uu, vv = np.meshgrid(u, v)
#     # Compute fourier transform
#     dotXU = np.tensordot(X, ([uu, vv]), 1)
#     angle = 2.0 * np.pi * dotXU
#     realCoeff = np.sum(np.cos(angle), 0)
#     imagCoeff = np.sum(np.sin(angle), 0)
#     out = (realCoeff ** 2 + imagCoeff ** 2) / numPoints
#     half_res = int(res // 2)
#     out[half_res, half_res] = 0     # cancel DC component
#     return out



def continuous_fourier_2d_torch(device, pts, res, freqStep=1):
    X = pts
    resolution = res
    xfreqStep = yfreqStep = freqStep
    numPoints = X.shape[0]
    xlow = -resolution * 0.5 * xfreqStep
    xhigh = resolution * 0.5 * xfreqStep
    ylow = -resolution * 0.5 * yfreqStep
    yhigh = resolution * 0.5 * yfreqStep
    u = torch.arange(xlow, xhigh, xfreqStep)
    v = torch.arange(ylow, yhigh, yfreqStep)
    
    uu, vv = torch.meshgrid(u, v)
    # print('uu:', uu)
    # uu, vv = np.meshgrid(np.linspace(xlow, xhigh, res), np.linspace(ylow, yhigh, res))
    # print('uu:', uu)
    # print(uu.shape, vv.shape)

    # Compute fourier transform
    # dotXU = torch.tensordot(X, ([uu, vv]), 1)
    batchGrid = torch.cat((vv.unsqueeze(-1).to(device), uu.unsqueeze(-1).to(device)), -1)
    # print(X.shape, batchGrid.shape)
    dotXU = torch.tensordot(X, batchGrid, dims=[[1], [2]])
    # print(dotXU.shape)
    angle = 2.0 * np.pi * dotXU
    realCoeff = torch.sum(torch.cos(angle), 0)
    imagCoeff = torch.sum(torch.sin(angle), 0)
    power = (realCoeff ** 2 + imagCoeff ** 2) / numPoints

    dcPos = int(res / 2.)
    dcComp = torch.ones_like(power)
    # print(dcComp.shape)
    dcComp[dcPos, dcPos] = 0
    power = power * dcComp

    return power
