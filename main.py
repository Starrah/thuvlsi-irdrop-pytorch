import torch

from netlist_parser import Netlist
from solver import PesudoInvSolver, JacobiSolver, ConjugateGradientSolver
from utils import global_device


def testOneSpice(spiceName):
    with torch.no_grad():
        n = Netlist()
        n.load_from_spice_file(spiceName + ".spice")
        n.load_groundtruth(spiceName + ".solution")
        n.process_after_loadfile()
        vdd, gnd = n.generateMatrix()
        print(spiceName + ".spice load finish, count: " + str(len(n.nodeLabel)))

        solvers = [
            # PesudoInvSolver(),
            JacobiSolver(),
            ConjugateGradientSolver()
        ]

        for solver in solvers:
            print(solver.__class__.__name__)
            resvdd, resgnd = solver.solveVddGnd(vdd, gnd, n.vddValue)
            dict, rmse = n.generateFinalResult(resvdd, resgnd)
            print("RMSE: " + str(rmse))
            n.outputFile(dict, spiceName + "_" + solver.__class__.__name__ + ".out")
            print()

        n.drawResult(dict, spiceName + "-" + solvers[-1].__class__.__name__)



if __name__ == '__main__':
    testOneSpice("ibmpg1")
    testOneSpice("ibmpg2")
    testOneSpice("ibmpg4")
    testOneSpice("ibmpg5")
    testOneSpice("ibmpg6")
    a = 1


