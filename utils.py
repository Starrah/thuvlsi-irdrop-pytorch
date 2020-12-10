from typing import Union, List, Tuple

import torch
import numpy as np
from matplotlib import pyplot as PLT
from matplotlib import cm as CM

global_device = torch.device("cuda:0")


def strToNum(s: str) -> Union[int, float]:
    try:
        return int(s)
    except:
        return float(s)


def average(arr: List[float]) -> float:
    sum = 0
    for v in arr:
        sum += v
    return sum / len(arr)


def calculateDiagonal(A: torch.Tensor, sizeA: List[int]) -> torch.Tensor:
    assert sizeA[0] == sizeA[1]
    dia = torch.zeros(sizeA[0], dtype=torch.float32, device=global_device)
    boolDiagonal = A[:, 0] == A[:, 1]
    indexA = A.to(torch.int64)
    dia[indexA[boolDiagonal, 0]] = A[boolDiagonal, 2]  # 对角元
    return dia


def diagonalPreconditioned(A: torch.Tensor, b: torch.tensor, sizeA: List[int]) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    :param A:
    :param b:
    :param sizeA:
    :return: 预条件化后的A、b、对角线
    """
    dia = calculateDiagonal(A, sizeA)
    A[:, 2] = A[:, 2] / dia[A[:, 0].to(torch.int64)]
    b = b / dia
    return A, b, dia


def sparseMatmul(A: torch.Tensor, x: torch.Tensor, sizeA: Union[List[int], Tuple[int, int]],
                 indexA=None) -> torch.Tensor:
    """
    :param A: <Tensor t, 3> m*n的矩阵的稀疏表示
    :param x: <Tensor n>
    :param sizeA: A的大小，即[m,n]
    :param indexA: 可选，是A[:, 0:2].to(torch.int64)
    :return: A*x（矩阵乘法）的结果 <Tensor m>
    """
    indexA = indexA if indexA is not None else A[:, 0:2].to(torch.int64)
    item1 = A[:, 2] * x[indexA[:, 1]]
    return torch.scatter_add(torch.zeros(sizeA[0], dtype=x.dtype, device=x.device), 0, indexA[:, 0], item1)


def RMSE(a: torch.Tensor, b: torch.Tensor) -> float:
    d = a - b
    return torch.sqrt(torch.mul(d, d).sum() / d.numel()).item()

def drawHeatmap(inputDepthMap, outputFile=None, title=None):
    fig: PLT.Figure = PLT.figure(facecolor='w')
    ax1: PLT.Axes = fig.add_subplot(1, 1, 1)

    # cmap = CM.get_cmap('RdYlBu_r', 1000)
    cmap = CM.get_cmap('rainbow', 1000)

    ax1.axis("off")
    map = ax1.imshow(inputDepthMap, interpolation="bilinear", cmap=cmap, aspect='auto')
    if title:
        PLT.title(title)

    # plot it
    if outputFile:
        PLT.savefig(outputFile)
    else:
        PLT.show()
    PLT.close(fig)

if __name__ == '__main__':
    a = np.random.randn(10,10)
    drawHeatmap(a)
    b = 1
