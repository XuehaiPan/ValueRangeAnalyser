import re
from collections import OrderedDict, deque
from math import inf, isinf, isfinite
from typing import Union, List, Sequence, Dict, Deque, Callable

from Function import Function, functionImplementation, number, integer
from ValueRange import ValueRange, EmptySet, IntegerNumberSet


__all__: List[str] = ['readSsaFile', 'RangeAnalyser', 'Function', 'ValueRange']


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
        def execute(stmt: str, env: str) -> ValueRange:
            constraint: Dict[str, Union[str, List[str]]] = func.constraints[stmt]
            resRange: ValueRange = EmptySet
            if constraint['type'] == 'funcCall':
                print('>' * (4 * depth + 3), 'call', self.functions[constraint['op']].declaration)
                resRange: ValueRange = self.analyse(func = constraint['op'],
                                                    args = list(attributes[arg]['range'][env] for arg in constraint['args']),
                                                    depth = depth + 1)
            elif constraint['type'] != 'condition':
                arg1Range: ValueRange = attributes[constraint['arg1']]['range'][env]
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
                    arg2Range: ValueRange = attributes[constraint['arg2']]['range'][env]
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
            attributes[constraint['res']]['range'][env]: ValueRange = resRange
            return resRange
        
        def applyFutureResolution() -> None:
            for stmt in resolutions.keys():
                constraint: Dict[str, Union[str, List[str]]] = func.constraints[stmt]
                env: str = constraint['blockLabel']
                for var, resolution in resolutions[stmt].items():
                    varRange: ValueRange = attributes[var]['range'][env]
                    if attributes[var]['type'] != 'num':
                        for flag in ('true', 'false'):
                            for blockLabel in constraint['{}List'.format(flag)]:
                                attributes[var]['range'][blockLabel]: ValueRange = varRange.intersection(other = resolution[flag])
        
        def doWidening() -> None:
            for constraint in func.constraints.values():
                try:
                    attributes[constraint['res']]: Union[str, Dict[str, ValueRange]] = {'type': 'var',
                                                                                        'dtype': None,
                                                                                        'range': {'temp': EmptySet}}
                except KeyError:
                    pass
                for arg in constraint['args']:
                    if number.fullmatch(string = arg):
                        num: Union[int, float] = (int(arg) if integer.fullmatch(string = arg) else float(arg))
                        attributes[arg]: Union[str, Dict[str, ValueRange]] = {'type': 'num',
                                                                              'dtype': type(num).__name__,
                                                                              'range': {'temp': ValueRange.asValueRange(value = num)}}
                    else:
                        attributes[arg]: Union[str, Dict[str, ValueRange]] = {'type': 'var',
                                                                              'dtype': None,
                                                                              'range': {'temp': EmptySet}}
            for id, dtype in func.variables.items():
                for var in attributes.keys():
                    if var.startswith(id):
                        attributes[var]['dtype']: str = dtype
            for i, arg in enumerate(func.args.keys()):
                for var in func.varsFromArg:
                    if var.startswith(arg):
                        attributes[var]['type']: str = 'arg'
                        attributes[var]['dtype']: str = func.args[arg]
                        attributes[var]['range']['temp']: ValueRange = args[i].asDtype(dtype = attributes[var]['dtype'])
            definitions: Deque[str] = deque(filter(None, func.defOfVariable.values()))
            while len(definitions) > 0:
                stmt: str = definitions.popleft()
                try:
                    var: str = func.constraints[stmt]['res']
                except KeyError:
                    continue
                if attributes[var]['type'] != 'var':
                    continue
                oldRange: ValueRange = attributes[var]['range']['temp']
                newRange: ValueRange = execute(stmt = func.defOfVariable[var], env = 'temp')
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
                    attributes[var]['range']['temp']: ValueRange = newRange
                    definitions.extend(func.useOfVariable[var])
            for var in attributes.keys():
                for blockLabel in func.blockLabels:
                    attributes[var]['range'][blockLabel]: ValueRange = attributes[var]['range']['temp']
        
        def determineFutureResolution() -> None:
            for stmt in filter(lambda stmt: func.constraints[stmt]['type'] == 'condition', func.constraints.keys()):
                constraint: Dict[str, Union[str, List[str]]] = func.constraints[stmt]
                op: str = constraint['op']
                arg1: str = constraint['arg1']
                arg2: str = constraint['arg2']
                env: str = constraint['blockLabel']
                arg1Range: ValueRange = attributes[arg1]['range'][env]
                arg2Range: ValueRange = attributes[arg2]['range'][env]
                arg1Resolution: Dict[str, ValueRange] = dict()
                arg2Resolution: Dict[str, ValueRange] = dict()
                if op == '<':
                    arg1Resolution['true']: ValueRange = RangeAnalyser.resolutionLT(varRange = arg2Range)
                    arg1Resolution['false']: ValueRange = RangeAnalyser.resolutionGE(varRange = arg2Range)
                    arg2Resolution['true']: ValueRange = RangeAnalyser.resolutionGT(varRange = arg1Range)
                    arg2Resolution['false']: ValueRange = RangeAnalyser.resolutionLE(varRange = arg1Range)
                elif op == '<=':
                    arg1Resolution['true']: ValueRange = RangeAnalyser.resolutionLE(varRange = arg2Range)
                    arg1Resolution['false']: ValueRange = RangeAnalyser.resolutionGT(varRange = arg2Range)
                    arg2Resolution['true']: ValueRange = RangeAnalyser.resolutionGE(varRange = arg1Range)
                    arg2Resolution['false']: ValueRange = RangeAnalyser.resolutionLT(varRange = arg1Range)
                if op == '>':
                    arg1Resolution['true']: ValueRange = RangeAnalyser.resolutionGT(varRange = arg2Range)
                    arg1Resolution['false']: ValueRange = RangeAnalyser.resolutionLE(varRange = arg2Range)
                    arg2Resolution['true']: ValueRange = RangeAnalyser.resolutionLT(varRange = arg1Range)
                    arg2Resolution['false']: ValueRange = RangeAnalyser.resolutionGE(varRange = arg1Range)
                elif op == '>=':
                    arg1Resolution['true']: ValueRange = RangeAnalyser.resolutionGE(varRange = arg2Range)
                    arg1Resolution['false']: ValueRange = RangeAnalyser.resolutionLT(varRange = arg2Range)
                    arg2Resolution['true']: ValueRange = RangeAnalyser.resolutionLE(varRange = arg1Range)
                    arg2Resolution['false']: ValueRange = RangeAnalyser.resolutionGT(varRange = arg1Range)
                elif op == '==':
                    arg1Resolution['true']: ValueRange = RangeAnalyser.resolutionEQ(varRange = arg2Range)
                    arg1Resolution['false']: ValueRange = RangeAnalyser.resolutionNE(varRange = arg2Range)
                    arg2Resolution['true']: ValueRange = RangeAnalyser.resolutionEQ(varRange = arg1Range)
                    arg2Resolution['false']: ValueRange = RangeAnalyser.resolutionNE(varRange = arg1Range)
                elif op == '!=':
                    arg1Resolution['true']: ValueRange = RangeAnalyser.resolutionNE(varRange = arg2Range)
                    arg1Resolution['false']: ValueRange = RangeAnalyser.resolutionEQ(varRange = arg2Range)
                    arg2Resolution['true']: ValueRange = RangeAnalyser.resolutionNE(varRange = arg1Range)
                    arg2Resolution['false']: ValueRange = RangeAnalyser.resolutionEQ(varRange = arg1Range)
                resolutions[stmt] = {arg1: arg1Resolution, arg2: arg2Resolution}
        
        def doNarrowing() -> None:
            changedBlockLabels: Deque[str] = deque(func.blocks.keys())
            while len(changedBlockLabels) > 0:
                env: str = changedBlockLabels.popleft()
                for stmt, constraint in func.blocks[env].constraints.items():
                    if constraint['type'] != 'condition':
                        applyFutureResolution()
                        res: str = constraint['res']
                        oldRange: ValueRange = attributes[res]['range'][env]
                        newRange: ValueRange = execute(stmt = stmt, env = env)
                        if newRange != oldRange:
                            try:
                                if isfinite(newRange.lower) and (isinf(oldRange.lower) or newRange.lower < oldRange.lower):
                                    if isfinite(oldRange.upper):
                                        newRange: ValueRange = newRange.union(ValueRange(lower = newRange.upper, upper = oldRange.upper,
                                                                                         dtype = newRange.dtype))
                                
                                if isfinite(newRange.upper) and (isinf(oldRange.upper) or newRange.upper > oldRange.upper):
                                    if isfinite(oldRange.lower):
                                        newRange: ValueRange = newRange.union(ValueRange(lower = oldRange.lower, upper = newRange.lower,
                                                                                         dtype = newRange.dtype))
                            except TypeError:
                                pass
                            attributes[res]['range'][env]: ValueRange = newRange
                            applyFutureResolution()
                            for successorLabel in func.successorLabelsWithoutKilling(block = env, var = res):
                                if attributes[res]['range'][successorLabel] != attributes[res]['range'][env]:
                                    attributes[res]['range'][successorLabel]: ValueRange = attributes[res]['range'][env]
                                    changedBlockLabels.append(successorLabel)
            applyFutureResolution()
        
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
        attributes: Dict[str, Dict[str, Union[str, Dict[str, ValueRange]]]] = dict()
        resolutions: Dict[str, Dict[str, Dict[str, ValueRange]]] = dict()
        doWidening()
        determineFutureResolution()
        doNarrowing()
        for block in func.blocks.values():
            env: str = block.label
            print('{}{}:'.format('|   ' * (depth + 1), env))
            if block.label != func.returnBlockLabel:
                for var in sorted(filter(lambda var: attributes[var]['type'] != 'num', block.GEN), key = Function.idCompareKey):
                    print('{}{} {}: {}'.format('|   ' * (depth + 2), attributes[var]['dtype'], var, attributes[var]['range'][env]))
            elif func.ret is not None:
                var: str = func.ret
                print('{}{} {}: {}'.format('|   ' * (depth + 2), attributes[var]['dtype'], var, attributes[var]['range'][env]))
        print('{}{} returns {}'.format('|   ' * depth, func.prototype, attributes[func.ret]['range'][func.returnBlockLabel]))
        return attributes[func.ret]['range'][func.returnBlockLabel]
    
    drawControlFlowGraph = None
    drawSimpleControlFlowGraph = None
    drawConstraintGraph = None
    
    @staticmethod
    def resolutionLT(varRange: ValueRange) -> ValueRange:
        return ValueRange(lower = -inf,
                          upper = varRange.upper - (1 if varRange.dtype == int else 0),
                          dtype = varRange.dtype)
    
    @staticmethod
    def resolutionLE(varRange: ValueRange) -> ValueRange:
        return ValueRange(lower = -inf, upper = varRange.upper, dtype = varRange.dtype)
    
    @staticmethod
    def resolutionGT(varRange: ValueRange) -> ValueRange:
        return ValueRange(lower = varRange.lower + (1 if varRange.dtype == int else 0),
                          upper = +inf,
                          dtype = varRange.dtype)
    
    @staticmethod
    def resolutionGE(varRange: ValueRange) -> ValueRange:
        return ValueRange(lower = varRange.lower, upper = +inf, dtype = varRange.dtype)
    
    @staticmethod
    def resolutionEQ(varRange: ValueRange) -> ValueRange:
        return varRange.copy()
    
    @staticmethod
    def resolutionNE(varRange: ValueRange) -> ValueRange:
        if varRange.lower == varRange.upper:
            return IntegerNumberSet.difference(other = varRange)
        return IntegerNumberSet.asDtype(dtype = varRange.dtype)


try:
    from pygraphviz import AGraph
    
    
    def drawControlFlowGraph(self: RangeAnalyser, file: str = None) -> AGraph:
        graph: AGraph = AGraph(directed = True, strict = True, overlap = False, compound = True, layout = 'dot')
        graph.node_attr['fontname'] = graph.edge_attr['fontname'] = 'Menlo'
        for func in self.functions.values():
            namespace: str = '{}::{{}}'.format(func.name)
            codeSplit: Dict[str, List[str]] = {label: ['{}:'.format(label)] + block.codeSplit for label, block in func.blocks.items()}
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
            nodeLabels: Dict[str, str] = {label: r'{}\l'.format(r'\l'.join(codeSplit[label])) for label, block in func.blocks.items()}
            graph.add_node(namespace.format('entry'), label = 'entry', style = 'bold', shape = 'ellipse')
            graph.add_node(namespace.format('exit'), label = 'exit', style = 'bold', shape = 'ellipse')
            for block in func.blocks.values():
                graph.add_node(namespace.format(block.label), label = nodeLabels[block.label], shape = 'box')
            graph.add_edge(namespace.format('entry'), namespace.format('<entry>'))
            graph.add_edge(namespace.format(func.returnBlockLabel), namespace.format('exit'))
            for block, neighbors in func.controlFlow.items():
                for successor in neighbors['successor']:
                    graph.add_edge(namespace.format(block), namespace.format(successor))
            nbunch: List[str] = [namespace.format('entry'), namespace.format('exit')]
            nbunch.extend(namespace.format(label) for label in func.blockLabels)
            graph.add_subgraph(nbunch = nbunch, name = 'cluster_{}'.format(func.name), label = func.declaration,
                               style = 'dashed', fontname = 'Menlo bold')
        if file is not None:
            graph.draw(path = file, prog = 'dot')
        return graph
    
    
    def drawSimpleControlFlowGraph(self: RangeAnalyser, file: str = None) -> AGraph:
        graph: AGraph = AGraph(directed = True, strict = False, overlap = False, compound = True, layout = 'dot')
        graph.node_attr['fontname'] = graph.edge_attr['fontname'] = 'Menlo'
        for func in self.functions.values():
            for prefix in ('', 'dominant::'):
                namespace: str = '{}{}::{{}}'.format(prefix, func.name)
                graph.add_node(namespace.format('entry'), label = 'entry', style = 'bold', shape = 'ellipse')
                graph.add_node(namespace.format('exit'), label = 'exit', style = 'bold', shape = 'ellipse')
                for block in func.blocks.values():
                    graph.add_node(namespace.format(block.label), label = '{}\l'.format(block.label), shape = 'box')
                graph.add_edge(namespace.format('entry'), namespace.format('<entry>'))
                graph.add_edge(namespace.format(func.returnBlockLabel), namespace.format('exit'))
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
        return graph
    
    
    def drawConstraintGraph(self: RangeAnalyser, file: str = None) -> AGraph:
        graph: AGraph = AGraph(directed = True, strict = False, overlap = True, layout = 'dot')
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
                    for flag in ('trueList', 'falseList'):
                        for successor in map(func.blocks.get, constraint[flag]):
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
        return graph
    
    
    RangeAnalyser.drawControlFlowGraph: Callable[[RangeAnalyser, str], AGraph] = drawControlFlowGraph
    RangeAnalyser.drawSimpleControlFlowGraph: Callable[[RangeAnalyser, str], AGraph] = drawSimpleControlFlowGraph
    RangeAnalyser.drawConstraintGraph: Callable[[RangeAnalyser, str], AGraph] = drawConstraintGraph
except ImportError:
    RangeAnalyser.drawControlFlowGraph = None
    RangeAnalyser.drawSimpleControlFlowGraph = None
    RangeAnalyser.drawConstraintGraph = None
