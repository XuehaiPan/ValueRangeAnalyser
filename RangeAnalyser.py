import re
import os
from typing import Type, Union, Optional, List, Tuple, Sequence, Set, Dict, Deque, Pattern, Match
from collections import OrderedDict, deque
import pygraphviz as pgv
from math import inf, isinf, nan, isnan
from Function import *
from ValueRange import ValueRange, EmptySet, IntegerNumberSet, RealNumberSet, dtypeFromString


def formatCode(statements: List[str]) -> List[str]:
    def doPreprocessing(stmt: str) -> str:
        stmt: str = re.sub(r'\s+', repl = ' ', string = stmt)
        stmt: str = re.sub(r'\s*;', repl = ';', string = stmt)
        stmt: str = re.sub(r';;.*$', repl = '', string = stmt)
        stmt: str = re.sub(r'(int|float)\s+D\.\d*\s*;', repl = '', string = stmt)
        stmt: str = re.sub(r'(?P<number>[+\-]?\d+(\.\d+)?([Ee][+\-]?\d+))', repl = lambda m: m.group('number').upper(), string = stmt)
        stmt: str = re.sub(r'^\s*goto\s+(?P<label1><[\w\s]+>)\s*\((?P<label2><[\w\s]+>)\)\s*;\s*$',
                           repl = lambda m: 'goto {};'.format(m.group('label2')), string = stmt)
        return stmt.strip()
    
    statements = list(filter(None, map(doPreprocessing, statements)))
    for i, stmt in enumerate(statements):
        if stmt.startswith('if') or stmt.startswith('else'):
            statements[i + 1] = '\t{}'.format(statements[i + 1])
    return statements


def readSsaFile(file: str) -> str:
    with open(file = file, mode = 'r', encoding = 'UTF-8') as ssaFile:
        return '\n'.join(formatCode(statements = ssaFile.readlines()))


class RangeAnalyser(object):
    def __init__(self, code: str) -> None:
        self.__functions: Dict[str, Function] = OrderedDict()
        for matcher in functionImplementation.finditer(string = code):
            self.__functions[matcher.group('name')] = Function(matcher.group())
        self.__functionNames: List[str] = list(self.functions.keys())
    
    @property
    def functions(self) -> Dict[str, Function]:
        return self.__functions
    
    @property
    def functionNames(self) -> List[str]:
        return self.__functionNames
    
    def analyse(self, func: Union[str, Function], args: Sequence[ValueRange], depth: int = 0) -> ValueRange:
        def execute(stmt: str) -> ValueRange:
            constraint: Dict[str, Union[str, List[str]]] = func.constraints[stmt]
            resRange: ValueRange = EmptySet
            if constraint['type'] == 'funcCall':
                print('>' * (4 * depth + 3), 'call', self.functions[constraint['op']].declaration)
                resRange: ValueRange = self.analyse(func = constraint['op'],
                                                    args = list(attributes[arg]['range'] for arg in constraint['args']),
                                                    depth = depth + 1)
            elif constraint['type'] != 'condition':
                arg1Range: ValueRange = attributes[constraint['arg1']]['range']
                if constraint['type'] == 'assignment':
                    resRange: ValueRange = arg1Range
                elif constraint['type'] == 'monocular':
                    if constraint['op'] == 'minus':
                        resRange: ValueRange = -arg1Range
                    elif constraint['op'] == '(int)':
                        resRange: ValueRange = arg1Range.asDtype(dtype = int)
                    elif constraint['op'] == '(float)':
                        resRange: ValueRange = arg1Range.asDtype(dtype = float)
                else:
                    arg2Range: ValueRange = attributes[constraint['arg2']]['range']
                    if constraint['type'] == 'PHI':
                        resRange: ValueRange = arg1Range.union(other = arg2Range)
                    elif constraint['type'] == 'arithmetic':
                        if constraint['op'] == '+':
                            resRange: ValueRange = arg1Range + arg2Range
                        elif constraint['op'] == '-':
                            resRange: ValueRange = arg1Range - arg2Range
                        elif constraint['op'] == '*':
                            resRange: ValueRange = arg1Range * arg2Range
                        elif constraint['op'] == '/':
                            resRange: ValueRange = arg1Range / arg2Range
            attributes[constraint['res']]['range']: ValueRange = resRange
            return resRange
        
        def doWidening() -> None:
            for constraint in func.constraints.values():
                try:
                    attributes[constraint['res']]: Union[str, ValueRange] = {'type': 'var',
                                                                             'dtype': None,
                                                                             'range': EmptySet}
                except KeyError:
                    pass
                for arg in constraint['args']:
                    if number.fullmatch(string = arg):
                        num: Union[int, float] = (int(arg) if integer.fullmatch(string = arg) else float(arg))
                        attributes[arg]: Union[str, ValueRange] = {'type': 'num',
                                                                   'dtype': type(num).__name__,
                                                                   'range': ValueRange.asValueRange(value = num)}
                    else:
                        attributes[arg]: Union[str, ValueRange] = {'type': 'var',
                                                                   'dtype': None,
                                                                   'range': EmptySet}
            for id, dtype in func.variables.items():
                for var in attributes.keys():
                    if var.startswith(id):
                        attributes[var]['dtype']: str = dtype
            for i, arg in enumerate(func.args.keys()):
                for var in func.varsFromArg:
                    if var.startswith(arg):
                        attributes[var]['type']: str = 'arg'
                        attributes[var]['dtype']: str = func.args[arg]
                        attributes[var]['range']: ValueRange = args[i].asDtype(dtype = attributes[var]['dtype'])
            definitions: Deque[str] = deque(filter(None, func.defOfVariable.values()))
            while len(definitions) > 0:
                stmt: str = definitions.popleft()
                try:
                    var: str = func.constraints[stmt]['res']
                except KeyError:
                    continue
                if attributes[var]['type'] != 'var':
                    continue
                oldRange: ValueRange = attributes[var]['range']
                newRange: ValueRange = execute(stmt = func.defOfVariable[var])
                if newRange != oldRange:
                    if newRange.isEmptySet():
                        for arg in func.constraints[stmt]['args']:
                            try:
                                definitions.extend(func.defOfVariable[arg])
                            except KeyError:
                                pass
                        definitions.append(stmt)
                    elif not oldRange.isEmptySet():
                        if newRange.lower < oldRange.lower and newRange.upper > oldRange.upper:
                            newRange: ValueRange = IntegerNumberSet.asDtype(dtype = attributes[var]['dtype'])
                        elif newRange.lower < oldRange.lower:
                            newRange: ValueRange = ValueRange(lower = -inf, upper = oldRange.upper,
                                                              dtype = attributes[var]['dtype'])
                        elif newRange.upper > oldRange.upper:
                            newRange: ValueRange = ValueRange(lower = oldRange.lower, upper = +inf,
                                                              dtype = attributes[var]['dtype'])
                        else:
                            newRange: ValueRange = oldRange
                    attributes[var]['range']: ValueRange = newRange
                    definitions.extend(func.useOfVariable[var])
        
        def doFutureResolution() -> None:
            pass
        
        def doNarrowing() -> None:
            pass
        
        if isinstance(func, Function):
            func: str = func.name
        func: Function = self.functions[func]
        print('{}analyse {}'.format('|   ' * depth, func.declaration), end = '')
        if len(func.args) > 0:
            print(' for {', ', '.join('{} in {}'.format(arg, args[i]) for i, arg in enumerate(func.args.keys())), '}')
        else:
            print()
        
        if func.ret is None:
            print('no return')
            return EmptySet
        attributes: Dict[str, Dict[str, Union[str, ValueRange]]] = dict()
        doWidening()
        doFutureResolution()
        doNarrowing()
        for var in sorted(filter(lambda var: attributes[var]['type'] != 'num', attributes.keys()), key = Function.idCompareKey):
            print('{}{} {}: {}'.format('|   ' * (depth + 1), attributes[var]['dtype'], var, attributes[var]['range']))
        print('{}{} returns {}'.format('|   ' * depth, func.declaration, attributes[func.ret]['range']))
        # print(attributes)
        return attributes[func.ret]['range']
    
    def drawControlFlowGraph(self, file: str = None) -> pgv.AGraph:
        graph: pgv.AGraph = pgv.AGraph(directed = True, strict = True, overlap = False, compound = True, layout = 'dot')
        graph.node_attr['fontname'] = graph.edge_attr['fontname'] = 'Menlo'
        for func in self.functions.values():
            namespace: str = '{}::{{}}'.format(func.name)
            codeSplit: Dict[str, List[str]] = {label: ['{}:'.format(label)] + block.codeSplit
                                               for label, block in func.blocks.items()}
            for label, block in func.blocks.items():
                codeSplit[label].append('transferCondition: {}'.format(block.transferCondition))
                codeSplit[label].append('trueList:  {{{}}}'.format(', '.join(block.trueList)))
                codeSplit[label].append('falseList: {{{}}}'.format(', '.join(block.falseList)))
                codeSplit[label].append('nextList:  {{{}}}'.format(', '.join(block.nextList)))
                codeSplit[label].append('GEN:  {{{}}}'.format(', '.join(sorted(block.GEN, key = Function.idCompareKey))))
                codeSplit[label].append('KILL: {{{}}}'.format(', '.join(sorted(block.KILL, key = Function.idCompareKey))))
                codeSplit[label].append('USE:  {{{}}}'.format(', '.join(sorted(block.USE, key = Function.idCompareKey))))
                codeSplit[label].append('IN:   {{{}}}'.format(', '.join(sorted(block.IN, key = Function.idCompareKey))))
                codeSplit[label].append('OUT:  {{{}}}'.format(', '.join(sorted(block.OUT, key = Function.idCompareKey))))
                codeSplit[label].insert(-9, '-' * max(map(len, codeSplit[label])))
            nodeLabels: Dict[str, str] = {label: r'{}\l'.format(r'\l'.join(codeSplit[label]))
                                          for label, block in func.blocks.items()}
            graph.add_node(namespace.format('entry'), label = 'entry', style = 'bold', shape = 'ellipse')
            graph.add_node(namespace.format('exit'), label = 'exit', style = 'bold', shape = 'ellipse')
            for block in func.blocks.values():
                graph.add_node(namespace.format(block.label), label = nodeLabels[block.label], shape = 'box')
                if re.search('return\s*\w*\s*;', string = block.code) is not None:
                    graph.add_edge(namespace.format(block.label), namespace.format('exit'))
            graph.add_edge(namespace.format('entry'), namespace.format('<entry>'))
            for block, neighbors in func.controlFlow.items():
                for successor in neighbors['successor']:
                    graph.add_edge(namespace.format(block), namespace.format(successor))
            nbunch: List[str] = [namespace.format('entry'), namespace.format('exit')]
            nbunch.extend(namespace.format(label) for label in func.blockLabels)
            graph.add_subgraph(nbunch = nbunch, name = 'cluster_{}'.format(func.name), label = func.declaration,
                               style = 'dashed', fontname = 'Menlo bold')
        if file is not None:
            graph.draw(path = file, prog = 'dot')
            # from matplotlib import pyplot as plt
            # plt.imshow(plt.imread(fname = file))
            # plt.xticks([])
            # plt.yticks([])
            # plt.show()
        return graph
    
    def drawSimpleControlFlowGraph(self, file: str = None) -> pgv.AGraph:
        graph: pgv.AGraph = pgv.AGraph(directed = True, strict = False, overlap = False, compound = True, layout = 'dot')
        graph.node_attr['fontname'] = graph.edge_attr['fontname'] = 'Menlo'
        for func in self.functions.values():
            for prefix in ('', 'dominant::'):
                namespace: str = '{}{}::{{}}'.format(prefix, func.name)
                graph.add_node(namespace.format('entry'), label = 'entry', style = 'bold', shape = 'ellipse')
                graph.add_node(namespace.format('exit'), label = 'exit', style = 'bold', shape = 'ellipse')
                for block in func.blocks.values():
                    graph.add_node(namespace.format(block.label), label = '{}\l'.format(block.label), shape = 'box')
                    if re.search('return\s*\w*\s*;', string = block.code) is not None:
                        graph.add_edge(namespace.format(block.label), namespace.format('exit'))
                graph.add_edge(namespace.format('entry'), namespace.format('<entry>'))
                for label, neighborLabels in func.controlFlow.items():
                    for successorLabel in neighborLabels['successor']:
                        graph.add_edge(namespace.format(label), namespace.format(successorLabel),
                                       color = (None if prefix != '' else 'black'))
                for label in func.blockLabels:
                    for dominantBlockLabel in func.dominantBlockLabelsOf(block = label):
                        graph.add_edge(namespace.format(dominantBlockLabel), namespace.format(label),
                                       color = ('black' if prefix != '' else None), style = 'dashed')
                nbunch: List[str] = [namespace.format('entry'), namespace.format('exit')]
                nbunch.extend(namespace.format(label) for label in func.blockLabels)
                graph.add_subgraph(nbunch = nbunch,
                                   name = 'cluster_{}{}'.format(prefix, func.name),
                                   label = '{}{}'.format(('dominant relations of ' if prefix != '' else ''), func.prototype),
                                   style = 'dashed', fontname = 'Menlo bold')
        if file is not None:
            graph.draw(path = file, prog = 'dot')
            # from matplotlib import pyplot as plt
            # plt.imshow(plt.imread(fname = file))
            # plt.xticks([])
            # plt.yticks([])
            # plt.show()
        return graph
    
    def drawConstraintGraph(self, file: str = None) -> pgv.AGraph:
        graph: pgv.AGraph = pgv.AGraph(directed = True, strict = False, overlap = True, layout = 'dot')
        graph.node_attr['fontname'] = graph.edge_attr['fontname'] = 'Menlo'
        for func in self.functions.values():
            namespace: str = '{}::{{}}'.format(func.name)
            nbunch: List[str] = list()
            for stmt, constraint in func.constraints.items():
                try:
                    graph.add_node(namespace.format(constraint['res']), label = constraint['res'])
                    nbunch.append(namespace.format(constraint['res']))
                except KeyError:
                    pass
                if constraint['type'] == 'assignment':
                    arg: str = constraint['arg1']
                    if number.fullmatch(string = arg):
                        graph.add_node(namespace.format('{}::{}'.format(constraint['blockLabel'], arg)), label = arg)
                        arg: str = '{}::{}'.format(constraint['blockLabel'], arg)
                        nbunch.append(namespace.format(arg))
                    graph.add_edge(namespace.format(arg), namespace.format(constraint['res']))
                else:
                    nbunch.append(namespace.format(stmt))
                    if constraint['type'] == 'condition':
                        graph.add_node(namespace.format(stmt), label = stmt, style = 'bold', color = 'orange')
                    else:
                        color: str = 'crimson'
                        if constraint['type'] == 'funcCall':
                            color: str = 'brown'
                        elif constraint['op'] == 'PHI':
                            color: str = 'purple'
                        graph.add_node(namespace.format(stmt), label = constraint['op'], style = 'bold', color = color)
                        graph.add_edge(namespace.format(stmt), namespace.format(constraint['res']))
                    for arg in constraint['args']:
                        if number.fullmatch(string = arg):
                            graph.add_node(namespace.format('{}::{}'.format(constraint['blockLabel'], arg)), label = arg)
                            arg: str = '{}::{}'.format(constraint['blockLabel'], arg)
                            nbunch.append(namespace.format(arg))
                        graph.add_edge(namespace.format(arg), namespace.format(stmt))
            for stmt, constraint in func.constraints.items():
                if constraint['type'] == 'condition':
                    for flag in ('true', 'false'):
                        for successor in map(func.blocks.get, constraint['{}List'.format(flag)]):
                            for arg in constraint['args']:
                                if number.fullmatch(string = arg) is not None:
                                    continue
                                elif arg in successor.USE:
                                    for cons in successor.constraints.values():
                                        if cons['stmt'] != stmt and arg in cons['args']:
                                            node: str = cons['stmt']
                                            if cons['type'] == 'assignment':
                                                node: str = cons['res']
                                            try:
                                                graph.remove_edge(namespace.format(arg), namespace.format(node))
                                            except KeyError:
                                                pass
                                            newNode: str = namespace.format('({})::{}'.format(cons['stmt'], arg))
                                            try:
                                                graph.get_node(newNode)
                                                while True:
                                                    try:
                                                        graph.remove_edge(newNode, namespace.format(node))
                                                    except KeyError:
                                                        break
                                            except KeyError:
                                                graph.add_node(newNode, label = None, shape = 'point', width = 0.0)
                                                graph.add_edge(namespace.format(stmt), newNode, label = flag[0].upper(),
                                                               dir = 'none', style = 'dashed',
                                                               fontname = 'Menlo bold', fontcolor = 'deeppink')
                                                graph.add_edge(namespace.format(arg), newNode, dir = 'none')
                                                nbunch.append(newNode)
                                            for i in range(cons['args'].count(arg)):
                                                graph.add_edge(newNode, namespace.format(node))
            if func.ret is not None:
                graph.add_node(namespace.format(func.ret), label = 'ret: {}'.format(func.ret), style = 'bold', color = 'dodgerblue')
                nbunch.append(namespace.format(func.ret))
            for arg in func.varsFromArg:
                graph.add_node(namespace.format(arg), label = 'arg: {}'.format(arg), style = 'bold', color = 'forestgreen')
                nbunch.append(namespace.format(arg))
            graph.add_subgraph(nbunch = nbunch, name = 'cluster_{}'.format(func.name), label = func.declaration,
                               style = 'dashed', fontname = 'Menlo bold')
        if file is not None:
            graph.draw(path = file, prog = 'dot')
            # from matplotlib import pyplot as plt
            # plt.imshow(plt.imread(fname = file))
            # plt.xticks([])
            # plt.yticks([])
            # plt.show()
        return graph


def main() -> None:
    # ssaFile: str = input('Input the name of the SSA form file: ')
    for i in range(1, 11):
        # i = 6
        ssaFile = 'benchmark/t%d.ssa' % i
        code: str = readSsaFile(file = ssaFile)
        analyser: RangeAnalyser = RangeAnalyser(code = code)
        print('file name:', ssaFile)
        for func in analyser.functions.values():
            print('function:', func.declaration)
            print('identifiers:', '({})'.format(', '.join('{} {}'.format(dtype, id)
                                                          for id, dtype in func.localVariables.items())))
            print('block labels:', func.blockLabels)
            print('control flow graph:', func.controlFlow)
            print('data flow:', func.dataFlow)
            print('constraints:', func.constraints)
            print('def of variables:', func.defOfVariable)
            print('use of variables:', func.useOfVariable)
            print()
            analyser.analyse(func = func, args = [ValueRange(0, 10, int) for arg in func.args.keys()])
            print()
        print()
        analyser.drawControlFlowGraph(file = '{}_CFG.png'.format(os.path.splitext(ssaFile)[0]))
        analyser.drawSimpleControlFlowGraph(file = '{}_SCFG.png'.format(os.path.splitext(ssaFile)[0]))
        analyser.drawConstraintGraph(file = '{}_CG.png'.format(os.path.splitext(ssaFile)[0]))
        # break


if __name__ == '__main__':
    main()
