import os
from math import inf
from typing import List

from RangeAnalyser import readSsaFile, RangeAnalyser, Function, ValueRange


def printSsaInfo(ssaFile: str, analyser: RangeAnalyser) -> None:
    print('file name:', ssaFile)
    print('function information:')
    try:
        analyser.drawControlFlowGraph(file = '{}_CFG.png'.format(os.path.splitext(ssaFile)[0]))
        analyser.drawSimpleControlFlowGraph(file = '{}_SCFG.png'.format(os.path.splitext(ssaFile)[0]))
        analyser.drawConstraintGraph(file = '{}_CG.png'.format(os.path.splitext(ssaFile)[0]))
    except TypeError:
        pass
    for func in analyser.functions.values():
        print('>>> function:', func.declaration)
        print('|   identifiers:', '({})'.format(', '.join('{} {}'.format(dtype, id) for id, dtype in func.variables.items())))
        print('|   variables:', '({})'.format(', '.join(sorted(func.GEN, key = Function.idCompareKey))))
        print('|   block labels:', func.blockLabels)
        print('|   control flow graph:', func.controlFlow)
        print('|   data flow:', func.dataFlow)
        print('|   constraints:', func.constraints)
        print('|   def statement of variables:', func.defOfVariable)
        print('|   use statements of variables:', func.useOfVariable)
    print()


def main() -> None:
    print('-' * 200)
    while True:
        ssaFile: str = input('Input the name of the SSA form file (type \"quit\" to exit): ')
        if ssaFile == 'quit':
            break
        code: str = readSsaFile(file = ssaFile)
        analyser: RangeAnalyser = RangeAnalyser(code = code)
        printSsaInfo(ssaFile = ssaFile, analyser = analyser)
        print('#' * 200)
        if len(analyser.functionNames) > 1:
            funcName: str = input('Choose function to benchmark ({}): '.format(' / '.join(analyser.functionNames)))
        else:
            funcName: str = analyser.functionNames[0]
            print('Only one function named {} in the SSA file.'.format(funcName))
        func: Function = analyser.functions[funcName]
        print('function declaration:', func.declaration)
        print()
        args: List[ValueRange] = []
        if len(func.args) > 0:
            for arg, dtype in func.args.items():
                argRangeStr: str = input('Input the range of argument {}: '.format(arg))
                lower, upper = eval(argRangeStr)
                args.append(ValueRange(lower = lower, upper = upper, dtype = dtype))
        else:
            print('function {} has no arguments'.format(func.name))
        print()
        analyser.analyse(func = func.name, args = args)
        print('#' * 200)
        print('-' * 200)


def benchmark() -> None:
    testArgs: List[List[ValueRange]] = [
        [],
        [ValueRange(lower = 200, upper = 300, dtype = int)],
        [ValueRange(lower = 0, upper = 10, dtype = int), ValueRange(20, upper = 50, dtype = int)],
        [ValueRange(lower = -inf, upper = +inf, dtype = int)],
        [],
        [ValueRange(lower = -inf, upper = +inf, dtype = int)],
        [ValueRange(lower = -10, upper = 10, dtype = int)],
        [ValueRange(lower = 1, upper = 100, dtype = int), ValueRange(lower = -2, upper = 2, dtype = int)],
        [],
        [ValueRange(lower = 30, upper = 50, dtype = int), ValueRange(lower = 90, upper = 100, dtype = int)]
    ]
    refRanges: List[str] = ['[100, 100]',
                            '[200, 300]',
                            '[20, 50]',
                            '[0, +inf)',
                            '[210, 210]',
                            '[-9, 10]',
                            '[16, 30]',
                            '[-3.2192308, 5.94230769]',
                            '[9791, 9791]',
                            '[-10, 40]']
    print('-' * 200)
    for i, (testArg, refRange) in enumerate(zip(testArgs, refRanges), start = 1):
        ssaFile = 'benchmark/t{}.ssa'.format(i)
        code: str = readSsaFile(file = ssaFile)
        analyser: RangeAnalyser = RangeAnalyser(code = code)
        printSsaInfo(ssaFile = ssaFile, analyser = analyser)
        print('#' * 200)
        analyser.analyse(func = 'foo', args = testArg)
        print('reference range: {}'.format(refRange))
        print('#' * 200)
        print('-' * 200)


if __name__ == '__main__':
    useBenchmark: bool = True
    if useBenchmark:
        benchmark()
    else:
        main()
