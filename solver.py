import time

import torch

from utils import global_device, sparseMatmul, RMSE


class AbstracrtSolver:
    """
    解线性方程Ax=b
    """
    def solve(self, A: torch.Tensor, b: torch.Tensor, sizeA: tuple, initialX: torch.Tensor = None):
        """
        :param A: <Tensor t, 3>矩阵的稀疏表示，(行、列、值)
        :param b: <Tensor n>向量
        :param initialX: 可选，迭代法中初始化的向量
        :param sizeA 长为2的tuple A矩阵的实际尺寸(n, m)
        :return: <Tensor m>解
        """
        raise NotImplementedError()

    def solveVddGnd(self, vdd, gnd, vddValue):
        initialCVdd = torch.randn(len(vdd[1]), dtype=torch.float32, device=global_device) * (vddValue / 9) + (vddValue * 5 / 6)
        initialCGnd = torch.randn(len(gnd[1]), dtype=torch.float32, device=global_device) * (vddValue / 9) + (vddValue * 1 / 6)
        startTime = time.time()
        resvdd = self.solve(vdd[0], vdd[1], (len(vdd[1]), len(vdd[1])), initialCVdd)
        resgnd = self.solve(gnd[0], gnd[1], (len(gnd[1]), len(gnd[1])), initialCGnd)
        endTime = time.time()
        print("time: " + str(endTime - startTime))
        return resvdd, resgnd


class PesudoInvSolver(AbstracrtSolver):
    def solve(self, A: torch.Tensor, b: torch.Tensor, sizeA: tuple, initialX: torch.Tensor = None):
        fullA = torch.zeros(sizeA, dtype=torch.float32, device=global_device)
        fullA[A[:, 0].to(torch.int64), A[:, 1].to(torch.int64)] = A[:, 2]
        return torch.matmul(torch.pinverse(fullA), b)

class JacobiSolver(AbstracrtSolver):
    def __init__(self):
        self.maxiter = 50000
        self.tolerance = 1e-6

    def solve(self, A: torch.Tensor, b: torch.Tensor, sizeA: tuple, initialX: torch.Tensor = None):
        x = initialX if initialX is not None else torch.randn(sizeA[0], dtype=torch.float32, device=global_device)
        assert sizeA[0] == sizeA[1]
        dia = torch.zeros(sizeA[0], dtype=torch.float32, device=global_device)
        boolDiagonalInA = A[:,0]==A[:,1]
        dia[A[boolDiagonalInA, 0].to(torch.int64)] = A[boolDiagonalInA, 2] # 对角元
        A = A[~boolDiagonalInA] # A现在仅含有非对角元元素的稀疏表示
        diff: float = 0.0
        iter = 0

        for iter in range(self.maxiter):
            AxNoDiag = sparseMatmul(A, x, sizeA)
            newX = (b - AxNoDiag) / dia
            diff = RMSE(x, newX)
            x = newX
            if diff <= self.tolerance:
                break
        print(iter+1, diff)
        return x

class ConjugateGradientSolver(AbstracrtSolver):
    def __init__(self):
        self.maxiter = 50000
        self.tolerance = 1e-6

    def solve(self, A: torch.Tensor, b: torch.Tensor, sizeA: tuple, initialX: torch.Tensor = None):
        x = initialX if initialX is not None else torch.randn(sizeA[0], dtype=torch.float32, device=global_device)
        assert sizeA[0] == sizeA[1]
        diff: float = 0.0
        iter = 0
        r = b - sparseMatmul(A, x, sizeA)
        p = r.clone()
        lastRR = torch.mul(r, r).sum()

        for iter in range(self.maxiter):
            Ap = sparseMatmul(A, p, sizeA)
            a = lastRR / torch.mul(p, Ap).sum()
            newX = x + a * p
            newR = r - a * Ap
            newRR = torch.mul(newR, newR).sum()
            b = newRR / lastRR
            p = newR + b * p
            r = newR
            lastRR = newRR
            diff = RMSE(x, newX)
            x = newX
            if diff <= self.tolerance:
                break
        print(iter+1, diff)
        return x

class NewtownSolver(AbstracrtSolver):
    def __init__(self):
        self.maxiter = 20000
        self.tolerance = 1e-10
        self.lr = 1e-1

    def solve(self, A: torch.Tensor, b: torch.Tensor, sizeA: tuple, initialX: torch.Tensor = None):
        assert sizeA[0] == sizeA[1]
        diff: float = 0.0
        iter = 0

        with torch.enable_grad():
            x = initialX if initialX is not None else torch.randn(sizeA[0], dtype=torch.float32, device=global_device)

            for iter in range(self.maxiter):
                x.requires_grad = True
                Ax = sparseMatmul(A, x, sizeA)
                loss = torch.norm(Ax - b)
                loss.backward()
                newX = (x - self.lr * x.grad).detach()
                diff = RMSE(x, newX)
                x = newX
                if diff <= self.tolerance:
                    break
        print(iter + 1, diff)
        return x