import re
from typing import Dict, Tuple, List, Union

import numpy as np
import torch

from utils import global_device, strToNum, average, diagonalPreconditioned, RMSE, drawHeatmap

# 用在Netlist.nodeLabel中，表示属于VDD还是GND哪一部分。
L_VDD = 1
L_GND = 0


class Netlist:
    def __init__(self):
        self.SHORT_THRESHOLD = 1e-5

        self.nodenameDict: Dict[str, int] = {}
        self.edges: List[List[Union[float, int]]] = []  # 数组元素必定是长为3的list，前两个表示节点的编号，第三个是电阻
        self.nodeCurrents: List[float] = []  # 节点上电流源的总电流
        self.nodeVoltages: List[float] = []  # 节点上附着的对地电压源电压（PS:本程序仅处理，所有电压源均是对地的VDD电压的情况。）
        self.connectToVSource: List[bool] = []  # 节点的电压是否被电压源固定。
        self.nodeLabel: List[int] = []  # 结点的分类标签。目前仅有VDD和GND两类。
        self.shortedEdges: List[List[int]] = []
        self.quickReferenceFromNodeToEdge: Dict[int, List[List[Union[float, int]]]] = {}

        self.L_VDDNodes: List[int] = []
        self.L_GNDNodes: List[int] = []
        self.vddValue: float = 0.0
        self.groundTruth: Dict[str, float] = {}
        # 以下是debug用的变量，与直接逻辑无关
        self.vddMapping: List[int] = []
        self.gndMapping: List[int] = []
        self.vddPlaneSize: int = 0
        self.gndPlaneSize: int = 0

        self.getNodeIndex("0")

    def load_from_spice_file(self, filename: str):
        with open(filename, "r") as f:
            for line in f.readlines():
                words = line.split()
                if len(words) != 4:
                    continue
                type = words[0]
                typechar = type[0]
                if typechar == "V" or typechar == "v" or typechar == "I" or typechar == "i" or typechar == "R" or typechar == "r":
                    nameA = words[1]
                    nameB = words[2]
                    lineValue = strToNum(words[3])
                    a = self.getNodeIndex(nameA)
                    b = self.getNodeIndex(nameB)
                else:
                    continue

                if typechar == "V" or typechar == "v":
                    if lineValue != 0:
                        # 是真正的电压源，登记。
                        self.nodeVoltages[a] = lineValue
                        self.connectToVSource[a] = True
                        self.connectToVSource[b] = True
                        self.nodeLabel[a] = L_VDD
                        if lineValue > self.vddValue:
                            self.vddValue = lineValue
                    else:
                        # 等同于短路
                        self.shortedEdges.append([a, b])
                elif typechar == "I" or typechar == "i":
                    # 根据IBM Benchmark的描述，电流源必定是成对的，一个是x 0，一个是0 y。
                    if nameB == "0":
                        self.nodeCurrents[a] += lineValue
                    elif nameA == "0":
                        self.nodeCurrents[b] -= lineValue
                elif typechar == "R" or typechar == "r":
                    if lineValue > self.SHORT_THRESHOLD:
                        # 是真正的电阻。
                        edge = [a, b, lineValue]
                        self.edges.append(edge)
                        self.quickReferenceFromNodeToEdge[a].append(edge)
                        self.quickReferenceFromNodeToEdge[b].append(edge)
                    else:
                        # 等同于短路
                        self.shortedEdges.append([a, b])

    def process_after_loadfile(self):
        """
        :return:
        """
        # 合并所有短路节点。
        # 总共分两步：1、根据短路边定义，重排edge序号等；2、删除短路节点，产生新的结点。
        # 第一步
        redirect_list = []  # 重定向表
        for i in range(len(self.nodenameDict)):
            redirect_list.append(i)

        def mergeTwo(into: int, fromm: int):
            into = redirect_list[into]
            redirect_list[fromm] = into
            self.connectToVSource[into] = self.connectToVSource[into] or self.connectToVSource[fromm]
            self.nodeVoltages[into] = max(self.nodeVoltages[into], self.nodeVoltages[fromm])
            self.nodeCurrents[into] += self.nodeCurrents[fromm]
            self.nodeLabel[into] = max(self.nodeLabel[into], self.nodeLabel[fromm])

        for shortEdge in self.shortedEdges:
            a = min(shortEdge[0], shortEdge[1])
            b = max(shortEdge[0], shortEdge[1])
            mergeTwo(a, b)

        # 第二步
        newNodesCount = 0
        newNodeCurrents: List[float] = []  # 节点上电流源的总电流
        newNodeVoltages: List[float] = []  # 节点上附着的对地电压源电压（PS:本程序仅处理，所有电压源均是对地的VDD电压的情况。）
        newConnectToVSource: List[bool] = []  # 节点的电压是否被电压源固定。
        newNodeLabel: List[int] = []  # 结点的分类标签。目前仅有VDD和GND两类。
        newQuickReferenceFromNodeToEdge: Dict[int, List[List[Union[float, int]]]] = {}

        for i in range(len(redirect_list)):
            if redirect_list[i] == i:
                # 是未被合并的节点
                redirect_list[i] = newNodesCount
                newNodeCurrents.append(self.nodeCurrents[i])
                newNodeVoltages.append(self.nodeVoltages[i])
                newConnectToVSource.append(self.connectToVSource[i])
                newNodeLabel.append(self.nodeLabel[i])
                newNodesCount += 1
            else:
                redirect_list[i] = redirect_list[redirect_list[i]]

        for nodename in self.nodenameDict:
            self.nodenameDict[nodename] = redirect_list[self.nodenameDict[nodename]]
        for i in range(newNodesCount):
            newQuickReferenceFromNodeToEdge[i] = []

        newEdges = []
        for edge in self.edges:
            edge[0] = redirect_list[edge[0]]
            edge[1] = redirect_list[edge[1]]
            if edge[0] != edge[1]:
                newEdges.append(edge)
                newQuickReferenceFromNodeToEdge[edge[0]].append(edge)
                newQuickReferenceFromNodeToEdge[edge[1]].append(edge)

        self.edges = newEdges
        self.nodeCurrents = newNodeCurrents
        self.nodeVoltages = newNodeVoltages
        self.connectToVSource = newConnectToVSource
        self.nodeLabel = newNodeLabel
        self.quickReferenceFromNodeToEdge = newQuickReferenceFromNodeToEdge

        # 按照广度优先搜索算法计算L_VDD集合。
        queue = []
        visited = []
        for label, i in zip(self.nodeLabel, range(len(self.nodeLabel))):
            visited.append(False)
            if label == L_VDD:
                visited[i] = True
                queue.append(i)
        while len(queue) > 0:
            cur = queue.pop(0)
            for edge in self.quickReferenceFromNodeToEdge[cur]:
                if edge[0] == cur:
                    other = edge[1]
                elif edge[1] == cur:
                    other = edge[0]
                else:
                    assert False
                if not visited[other]:
                    self.nodeLabel[other] = L_VDD
                    visited[other] = True
                    queue.append(other)

    def generateMatrix(self, preconditioned=True) -> Tuple[Tuple[torch.Tensor, torch.Tensor, List[int]],
                                                           Tuple[torch.Tensor, torch.Tensor, List[int]]]:
        """
        最终要求解的线性方程为 AGA^T·u=AGU-I，记为B·u=c，本函数即是计算B和c并返回。
        B是三元组格式<Tensor t,3>，c是<Tensor n>。t是B矩阵中非零元素稀疏表示。
        mapping: List[int]是从全局的index到子集（例如VDD点集合、GND点集合）的index的映射）
        :return: ((B,c,mapping),(B,c,mapping))前面是VDD，后面是GND。
        """
        vddToSolve: List[int] = []
        gndToSolve: List[int] = []
        for i in range(len(self.nodeLabel)):
            if not self.connectToVSource[i]:
                if self.nodeLabel[i] == L_VDD:
                    vddToSolve.append(i)
                elif self.nodeLabel[i] == L_GND:
                    gndToSolve.append(i)
        vddBCM = self._generateBAndC(vddToSolve)
        gndBCM = self._generateBAndC(gndToSolve)
        self.vddPlaneSize = len(vddBCM[1])
        self.gndPlaneSize = len(gndBCM[1])
        self.vddMapping = vddBCM[2]
        self.gndMapping = gndBCM[2]
        if preconditioned:
            vddConditioned = diagonalPreconditioned(vddBCM[0], vddBCM[1], [self.vddPlaneSize, self.vddPlaneSize])
            vddBCM = vddConditioned[0:2] + vddBCM[2:]
            gndConditioned = diagonalPreconditioned(gndBCM[0], gndBCM[1], [self.gndPlaneSize, self.gndPlaneSize])
            gndBCM = gndConditioned[0:2] + gndBCM[2:]
        return vddBCM, gndBCM

    def _generateBAndC(self, nodes: List[int]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        :param nodes: VDD或GND的节点集合，大小记为n
        :return: （B: <Tensor t, 3>, C: <Tensor n>, mapping: List[int]是从全局index到vdd index的映射）
        """
        mapping = [-1] * len(self.nodeLabel)
        curI = 0
        for i in nodes:
            mapping[i] = curI
            curI += 1
        B: Dict[int, Dict[int, float]] = {}  # 矩阵稀疏表示
        C: List[float] = [0.0] * len(nodes)  # 向量

        def addIntoB(a, b, value):
            a = mapping[a]
            b = mapping[b]
            if b == -1:
                return
            if a not in B:
                B[a] = {}
            if b not in B[a]:
                B[a][b] = 0.0
            B[a][b] += value

        for a in nodes:
            for edge in self.quickReferenceFromNodeToEdge[a]:
                b = edge[1] if edge[0] == a else edge[0]
                addIntoB(a, a, 1 / edge[2])
                if self.connectToVSource[b]:
                    if self.nodeLabel[b] == L_VDD:
                        C[mapping[a]] += self.nodeVoltages[b] / edge[2]
                else:
                    addIntoB(a, b, -1 / edge[2])
            C[mapping[a]] -= self.nodeCurrents[a]

        resB = []
        for a in B:
            for b in B[a]:
                resB.append([a, b, B[a][b]])

        dd = {}
        for cc in C:
            if cc not in dd:
                dd[cc] = 1
            else:
                dd[cc] += 1

        return torch.tensor(resB, dtype=torch.float32, device=global_device), \
               torch.tensor(C, dtype=torch.float32, device=global_device), mapping

    def getNodeIndex(self, nodename: str) -> int:
        r = self.nodenameDict.get(nodename, None)
        if r is None:
            r = len(self.nodenameDict)
            self.nodenameDict[nodename] = r
            self.nodeCurrents.append(0.0)
            self.nodeVoltages.append(0.0)
            self.connectToVSource.append(False)
            self.nodeLabel.append(L_GND)
            self.quickReferenceFromNodeToEdge[r] = []
        return r

    def load_groundtruth(self, filename: str):
        with open(filename, "r") as f:
            for line in f.readlines():
                words = line.split()
                if len(words) != 2:
                    continue
                nodename = words[0]
                lineValue = strToNum(words[1])
                self.groundTruth[nodename] = lineValue
        self.groundTruth["0"] = 0.0

    def generateFinalResult(self, vddRes: torch.Tensor, gndRes: torch.Tensor) -> Tuple[
        Dict[str, float], Union[float, None]]:
        """
        把Tensor格式的结果数组转化为结点的字典。
        :param vddRes: <Tensor n_vdd> n_vdd是vdd平面待定结点个数
        :param gndRes: <Tensor n_gnd> n_gnd是gnd平面待定结点个数
        :return: (计算结果-node名称到算出的电压值对应的字典， 均方误差开根号（RMSE）)
        """
        result = {}
        for nodename in self.nodenameDict:
            idx = self.nodenameDict[nodename]
            if self.connectToVSource[idx]:
                result[nodename] = self.nodeVoltages[idx]
            else:
                if self.nodeLabel[idx] == L_VDD:
                    result[nodename] = vddRes[self.vddMapping[idx]].item()
                elif self.nodeLabel[idx] == L_GND:
                    result[nodename] = gndRes[self.gndMapping[idx]].item()
        rmse = None
        if len(self.groundTruth) > 0:
            # 计算RMSE
            gtArr = []
            resArr = []
            for nodename in result:
                resArr.append(result[nodename])
                gtArr.append(self.groundTruth[nodename])
            rmse = RMSE(torch.tensor(gtArr, device=global_device), torch.tensor(resArr, device=global_device))
        return result, rmse

    def drawResult(self, valueDict: Dict[str, float], title=""):
        pts = {"x": [], "y": [], "v": []}
        for nodename in valueDict:
            matchObj = re.match(r'^n(\d+)_(\d+)_(\d+)$', nodename, re.I)
            if not matchObj:
                continue
            layer = int(matchObj.group(1))
            x = int(matchObj.group(2))
            y = int(matchObj.group(3))
            value = valueDict[nodename]
            pts["x"].append(x)
            pts["y"].append(y)
            pts["v"].append(value)

        x = np.array(pts["x"])
        y = np.array(pts["y"])
        v = np.array(pts["v"])
        resolution = 100
        x = np.round((x - x.min()) / (x.max() - x.min()) * resolution).astype(np.int32)
        y = np.round((y - y.min()) / (y.max() - y.min()) * resolution).astype(np.int32)
        heatmap = np.zeros((resolution + 1, resolution + 1))
        heatmap[x, y] = v
        drawHeatmap(heatmap, title=title)

    def outputFile(self, dict, filename):
        with open(filename, "w") as f:
            lines = map(lambda k: k + " " + str(dict[k]) + "\n", dict)
            f.writelines(lines)

    def groundtruthToResultTensor(self, type: str) -> torch.Tensor:
        """
        :param type: "vdd"或"gnd"
        :return: <Tensor n>
        """
        if type == "vdd":
            result = []
            for i in range(self.vddPlaneSize):
                result.append([])
            for nodename in self.nodenameDict:
                idx = self.nodenameDict[nodename]
                if not self.connectToVSource[idx] and self.nodeLabel[idx] == L_VDD:
                    result[self.vddMapping[idx]].append(self.groundTruth[nodename])
            result = list(map(lambda arr: average(arr), result))
            return torch.tensor(result, dtype=torch.float32, device=global_device)
        elif type == "gnd":
            result = [[]] * self.gndPlaneSize
            for nodename in self.nodenameDict:
                idx = self.nodenameDict[nodename]
                if not self.connectToVSource[idx] and self.nodeLabel[idx] == L_GND:
                    result[self.gndMapping[idx]].append(self.groundTruth[nodename])
            result = list(map(lambda arr: average(arr), result))
            return torch.tensor(result, dtype=torch.float32, device=global_device)
