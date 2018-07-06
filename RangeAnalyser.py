import re
import os
from typing import Type, Union, Optional, List, Tuple, Sequence, Set, Dict, Deque, Pattern, Match
from collections import OrderedDict, deque
import pygraphviz as pgv
from math import inf, isinf, nan, isnan
from ValueRange import ValueRange, EmptySet, IntegerNumberSet, RealNumberSet, dtypeFromString


keywords: Set[str] = {'int', 'float', 'if', 'else', 'goto'}
types: Dict[str, Type] = {'int': int, 'float': float}

monocularOperators: Set[str] = {'minus', '(int)', '(float)'}
binocularOperators: Set[str] = {'+', '-', '*', '/', '==', '!=', '<', '>', '<=', '>='}
relationalOperators: Set[str] = {'==', '!=', '<', '>', '<=', '>='}
arithmeticOperators: Set[str] = {'+', '-', '*', '/'}
wordSplitters: str = r'[\s,.?;:\'\"\\|~!@#$%^&+\-*/=<>{}\[\]()]'

functionImplementation: Pattern = re.compile(r'(?P<name>\w+)\s*\((?P<args>[\w\s,]*)\)\s*{\s*(?P<body>[^}]*)\s*}')
variableDeclaration: Pattern = re.compile(r'(?P<type>int|float)\s+(?P<id>\w+)\s*' + wordSplitters)
blockLabel: Pattern = re.compile(r'^\s*(?P<label><[\w\s]+>)\s*:\s*$')
ifStatement: Pattern = re.compile(r'^\s*if\s*\((?P<cond>[\w\s=<>!+\-]+)\)\s*$')
gotoStatement: Pattern = re.compile(r'^\s*goto\s+(?P<label><[\w\s]+>)\s*;\s*$')
returnStatement: Pattern = re.compile(r'^\s*return\s*(?P<id>\w*)\s*;\s*$')

variableAssignment: Pattern = re.compile('^\s*(?P<res>(?P<id>\w*)_(?P<num>\d+))\s*=(?P<expr>[^=;]*);\s*$')
phiStatement: Pattern = re.compile(r'^\s*#\s+(?P<res>(?P<id>\w*)_(?P<num>\d+))\s*=\s*PHI\s*<\s*'
                                   r'(?P<arg1>(?P<id1>\w*)_(?P<num1>\d+))\s*,\s*'
                                   r'(?P<arg2>(?P<id2>\w*)_(?P<num2>\d+))'
                                   r'\s*>\s*$')
functionCall: Pattern = re.compile(r'^\s*(?P<name>\w+)\s*\((?P<args>[\w\s+\-.,]*)\)\s*$')

numPattern: str = r'(?P<number>[+\-]?\d+(\.\d+)?([Ee][+\-]?\d+)?)'
varPattern: str = r'(?P<id>\w*)_(?P<num>\d+)'
factorPattern: str = r'(\w*_\d+)|([+\-]?\d+(\.\d+)?([Ee][+\-]?\d+)?)'
number: Pattern = re.compile(r'^\s*{}\s*$'.format(numPattern))
integer: Pattern = re.compile(r'^\s*(?P<integer>[+\-]?\d+)\s*$')
variable: Pattern = re.compile(r'^\s*{}\s*$'.format(varPattern))
factor: Pattern = re.compile(r'^\s*{}\s*$'.format(factorPattern))

operations: Dict[str, Pattern] = dict()
var1Pattern: str = r'(?P<id1>\w*)_(?P<num1>\d+)'
var2Pattern: str = r'(?P<id2>\w*)_(?P<num2>\d+)'
num1Pattern: str = r'(?P<number1>[+-]?\d+(\.\d+)?([Ee][+-]?\d+)?)'
num2Pattern: str = r'(?P<number2>[+-]?\d+(\.\d+)?([Ee][+-]?\d+)?)'
operations['minus']: Pattern = re.compile(r'^\s*-\s*(?P<arg1>{})\s*$'.format(factorPattern))
operations['(int)']: Pattern = re.compile(r'^\s*(?P<op>\(int\))\s*(?P<arg1>{})\s*$'.format(factorPattern))
operations['(float)']: Pattern = re.compile(r'^\s*(?P<op>\(float\))\s*(?P<arg1>{})\s*$'.format(factorPattern))
for op in binocularOperators:
    opPattern: str = op
    if op in '+*':
        opPattern: str = r'\{}'.format(op)
    operations[op]: Pattern = re.compile(r'^\s*(?P<arg1>{})\s+(?P<op>{})\s+(?P<arg2>{})\s*$'.format(factorPattern,
                                                                                                    opPattern,
                                                                                                    factorPattern))
    del opPattern

del varPattern, numPattern, factorPattern, var1Pattern, var2Pattern, num1Pattern, num2Pattern


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


Block = type('Block', (object,), dict())


class ThreeAddressCode(object):
    def __init__(self) -> None:
        pass


class Function(object):
    def __init__(self, code: str) -> None:
        matcher: Match = functionImplementation.search(string = code)
        if matcher is not None:
            varFromArg: Set[str] = set()
            code: str = re.sub(r'(?P<postfix>\w*_\d+)\s*\(D\)',
                               repl = lambda m: (varFromArg.add(m.group('postfix')), m.group('postfix'))[1],
                               string = code)
            code: str = re.sub(r'(?P<postfix>\w*_\d+)\s*\(\d+\)', repl = lambda m: m.group('postfix'), string = code)
            matcher: Match = functionImplementation.search(string = code)
            self.__varsFromArg: Set[str] = varFromArg
            self.__code: str = code
            self.__codeSplit: List[str] = code.splitlines()
            self.__body: str = matcher.group('body').strip()
            self.__bodySplit: List[str] = self.body.splitlines()
            self.__name: str = matcher.group('name').strip()
            self.__args: Dict[str, str] = Function.parseVariableDeclaration(statement = '{},'.format(matcher.group('args')))
            try:
                self.__ret: str = re.search('return\s+(?P<ret>\w+)\s*;', string = self.body).group('ret')
            except AttributeError:
                self.__ret: str = None
            self.__declaration: str = '{}({})'.format(self.name, ', '.join('{} {}'.format(dtype, var)
                                                                           for var, dtype in self.args.items()))
            self.__prototype: str = '{}({})'.format(self.name, ', '.join(self.args.values()))
            self.__variables: Dict[str, str] = Function.parseVariableDeclaration(statement = code)
            self.__localVariables: Dict[str, str] = Function.parseVariableDeclaration(statement = self.body)
            self.__GEN: List[str] = list(self.varsFromArg)
            self.__GEN.extend(Function.parseVariableAssignment(statement = self.body))
            self.__blockLabels: List[str] = None
            self.__blocks: Dict[str, Block] = None
            self.__controlFlow: Dict[str, Dict[str, Set[str]]] = None
            self.__dataFlow: Dict[str, Dict[str, Set[str]]] = None
            self.__constraints: Dict[str, Dict[str, Union[str, List[str]]]] = None
            self.__dominantBlockLabelsOf: Dict[str, Set[str]] = None
            self.__dominantBlockLabelsBy: Dict[str, Set[str]] = None
            self.__defOfVariable: Dict[str, str] = None
            self.__useOfVariable: Dict[str, List[str]] = None
        else:
            raise ValueError
    
    @property
    def code(self) -> str:
        return self.__code
    
    @property
    def codeSplit(self) -> List[str]:
        return self.__codeSplit
    
    @property
    def body(self) -> str:
        return self.__body
    
    @property
    def bodySplit(self) -> List[str]:
        return self.__bodySplit
    
    @property
    def name(self) -> str:
        return self.__name
    
    @property
    def declaration(self) -> str:
        return self.__declaration
    
    @property
    def prototype(self) -> str:
        return self.__prototype
    
    @property
    def args(self) -> Dict[str, str]:
        return self.__args
    
    @property
    def ret(self) -> str:
        return self.__ret
    
    @property
    def varsFromArg(self) -> Set[str]:
        return self.__varsFromArg
    
    @property
    def variables(self) -> Dict[str, str]:
        return self.__variables
    
    @property
    def localVariables(self) -> Dict[str, str]:
        return self.__localVariables
    
    @property
    def GEN(self) -> List[str]:
        return self.__GEN
    
    @property
    def defOfVariable(self) -> Dict[str, str]:
        if self.__defOfVariable is None:
            _ = self.useOfVariable
        return self.__defOfVariable
    
    @property
    def useOfVariable(self) -> Dict[str, List[str]]:
        if self.__useOfVariable is None:
            self.__useOfVariable: Dict[str, List[str]] = {var: list() for var in self.GEN}
            self.__defOfVariable: Dict[str, str] = {var: None for var in self.GEN}
            for stmt, constraint in self.constraints.items():
                try:
                    self.__defOfVariable[constraint['res']]: str = stmt
                except KeyError:
                    pass
                for arg in constraint['args']:
                    try:
                        self.__useOfVariable[arg].append(stmt)
                    except KeyError:
                        pass
        return self.__useOfVariable
    
    @property
    def blockLabels(self) -> List[str]:
        if self.__blockLabels is None:
            self.__blockLabels: List[str] = list(self.blocks.keys())
        return self.__blockLabels
    
    @property
    def blocks(self) -> Dict[str, Block]:
        if self.__blocks is None:
            self.__blocks: Dict[str, Block] = OrderedDict()
            label: str = '<entry>'
            codeSplit: List[str] = ['{} {};'.format(dtype, var) for var, dtype in self.args.items()]
            for stmt in self.bodySplit:
                matcher: Match = blockLabel.fullmatch(string = stmt)
                if matcher is None:
                    codeSplit.append(stmt)
                else:
                    self.__blocks[label] = Block(func = self, label = label, codeSplit = codeSplit)
                    label: str = matcher.group('label')
                    codeSplit: List[str] = list()
            else:
                self.__blocks[label]: Block = Block(func = self, label = label, codeSplit = codeSplit)
            self.__blocks['<entry>'].GEN.update(self.varsFromArg)
        return self.__blocks
    
    @property
    def controlFlow(self) -> Dict[str, Dict[str, Set[str]]]:
        if self.__controlFlow is None:
            self.__controlFlow: Dict[str, Dict[str, Set[str]]] = {block.label: {'predecessor': set(), 'successor': set()}
                                                                  for block in self.blocks.values()}
            for block in self.blocks.values():
                gotoList: Set[str] = set(block.trueList + block.falseList + block.nextList)
                self.__controlFlow[block.label]['successor'].update(gotoList)
                for label in gotoList:
                    self.__controlFlow[label]['predecessor'].add(block.label)
        return self.__controlFlow
    
    @property
    def dataFlow(self) -> Dict[str, Dict[str, Set[str]]]:
        if self.__dataFlow is None:
            self.__dataFlow: Dict[str, Dict[str, Set[str]]] = {block.label: {'IN': set(), 'OUT': set(), 'USE': block.USE,
                                                                             'GEN': block.GEN, 'KILL': block.KILL}
                                                               for block in self.blocks.values()}
            changed: Set[Block] = set(self.blocks.values())
            while len(changed) > 0:
                block: Block = changed.pop()
                IN: Set[str] = set()
                for predecessorLabel in block.predecessorLabels:
                    IN.update(self.__dataFlow[predecessorLabel]['OUT'])
                self.__dataFlow[block.label]['IN'] = IN
                OUT: Set[str] = block.GEN.union(IN.difference(block.KILL))
                if OUT != self.__dataFlow[block.label]['OUT']:
                    changed.update(block.successors)
                    self.__dataFlow[block.label]['OUT'] = OUT
        return self.__dataFlow
    
    @property
    def constraints(self) -> Dict[str, Dict[str, Union[str, List[str]]]]:
        if self.__constraints is None:
            self.__constraints: Dict[str, Dict[str, Union[str, List[str]]]] = OrderedDict()
            for block in self.blocks.values():
                self.__constraints.update(block.constraints)
        return self.__constraints
    
    def successorLabelsOf(self, block: Union[str, Block]) -> Set[str]:
        if isinstance(block, Block):
            block: str = block.label
        return self.controlFlow[block]['successor']
    
    def successorsOf(self, block: Union[str, Block]) -> Set[Block]:
        return set(map(self.blocks.get, self.successorLabelsOf(block = block)))
    
    def predecessorLabelsOf(self, block: Union[str, Block]) -> Set[str]:
        if isinstance(block, Block):
            block: str = block.label
        return self.controlFlow[block]['predecessor']
    
    def predecessorsOf(self, block: Union[str, Block]) -> Set[Block]:
        return set(map(self.blocks.get, self.predecessorLabelsOf(block = block)))
    
    def dominantBlockLabelsOf(self, block: Union[str, Block]) -> Set[str]:
        if isinstance(block, Block):
            block: str = block.label
        if self.__dominantBlockLabelsOf is None:
            self.__dominantBlockLabelsOf: Dict[str, Set[str]] = {label: set(self.blockLabels) for label in self.blockLabels}
            self.__dominantBlockLabelsOf['<entry>']: Set[str] = set()
            changed: bool = True
            while changed:
                changed: bool = False
                for label in self.blockLabels:
                    if label == '<entry>':
                        continue
                    dominantBlockLabels: Set[str] = set(self.blockLabels)
                    predecessorLabels: Set[str] = self.predecessorLabelsOf(block = label)
                    for predecessorLabel in predecessorLabels:
                        dominantBlockLabels.intersection_update(self.__dominantBlockLabelsOf[predecessorLabel])
                    if len(predecessorLabels) == 1:
                        dominantBlockLabels.update(predecessorLabels)
                    if dominantBlockLabels != self.__dominantBlockLabelsOf[label]:
                        self.__dominantBlockLabelsOf[label]: Set[str] = dominantBlockLabels
                        changed: bool = True
        return self.__dominantBlockLabelsOf[block]
    
    def dominantBlockLabelsBy(self, block: Union[str, Block]) -> Set[str]:
        if isinstance(block, Block):
            block: str = block.label
        if self.__dominantBlockLabelsBy is None:
            self.__dominantBlockLabelsBy: Dict[str, Set[str]] = {label: set() for label in self.blockLabels}
            for label1 in self.blockLabels:
                for label2 in self.dominantBlockLabelsOf(block = label1):
                    self.__dominantBlockLabelsBy[label2].add(label1)
        return self.__dominantBlockLabelsBy[block]
    
    def IN(self, block: Union[str, Block]) -> Set[str]:
        if isinstance(block, Block):
            block: str = block.label
        return self.dataFlow[block]['IN']
    
    def OUT(self, block: Union[str, Block]) -> Set[str]:
        if isinstance(block, Block):
            block: str = block.label
        return self.dataFlow[block]['OUT']
    
    @staticmethod
    def parseVariableDeclaration(statement: str) -> Dict[str, str]:
        identifiers: Dict[str, str] = OrderedDict()
        for matcher in variableDeclaration.finditer(string = statement):
            identifiers[matcher.group('id')] = matcher.group('type')
        return identifiers
    
    @staticmethod
    def parseVariableAssignment(statement: str) -> List[str]:
        return list(map(lambda m: m.group('id'), re.finditer(r'(?P<id>\w*_\d+)\s*=[^=]', string = statement)))
    
    @staticmethod
    def parseVariableAssignmentStartWith(id: str, statement: str) -> List[str]:
        return list(map(lambda m: m.group('id'), re.finditer(r'(?P<id>{}(_\d+)?)\s*=[^=]'.format(id), string = statement)))
    
    @staticmethod
    def idCompareKey(id_n: str) -> Tuple[str, int]:
        index = id_n.rfind('_')
        return id_n[:index], int(id_n[index + 1:])
    
    def __str__(self) -> str:
        return self.declaration
    
    __repr__ = __str__


class Block(object):
    def __init__(self, func: Function, label: str, codeSplit: List[str]) -> None:
        self.__func: Function = func
        self.__label: str = label
        self.__codeSplit: List[str] = list(codeSplit)
        self.__code: str = '\n'.join(self.codeSplit)
        self.__GEN: Set[str] = set(Function.parseVariableAssignment(statement = self.code))
        self.__KILL: Set[str] = set()
        for id in self.function.variables.keys():
            if len(Function.parseVariableAssignmentStartWith(id = id, statement = self.code)) > 0:
                self.__KILL.update(filter(lambda id_n: id_n.startswith(id), self.function.GEN))
        self.__KILL.difference_update(self.GEN)
        self.__USE: Set[str] = None
        self.__constraints: Dict[str, Dict[str, Union[str, List[str]]]] = None
        self.__transferCondition: str = None
        self.__trueList: List[str] = list()
        self.__falseList: List[str] = list()
        self.__nextList: List[str] = None
        try:
            if ifStatement.fullmatch(string = self.codeSplit[-4]) is not None:
                transferConstraint: Dict[str, str] = list(self.constraints.values())[-1]
                self.__transferCondition: str = transferConstraint['stmt']
                self.trueList.append(transferConstraint['true'])
                self.falseList.append(transferConstraint['false'])
        except IndexError:
            pass
    
    @property
    def function(self) -> Function:
        return self.__func
    
    @property
    def label(self) -> str:
        return self.__label
    
    @property
    def code(self) -> str:
        return self.__code
    
    @property
    def codeSplit(self) -> List[str]:
        return self.__codeSplit
    
    @property
    def GEN(self) -> Set[str]:
        return self.__GEN
    
    @property
    def KILL(self) -> Set[str]:
        return self.__KILL
    
    @property
    def USE(self) -> Set[str]:
        if self.__USE is None:
            self.__USE: Set[str] = set()
            GEN: Set[str] = set()
            for constraint in self.constraints.values():
                for arg in constraint['args']:
                    if variable.fullmatch(string = arg) and arg not in GEN:
                        self.__USE.add(arg)
                try:
                    GEN.add(constraint['res'])
                except KeyError:
                    pass
            matcher: Match = re.search('return\s*(?P<ret>\w+)\s*;', string = self.code)
            if matcher is not None:
                self.__USE.add(matcher.group('ret'))
            self.__USE.intersection_update(self.IN)
        return self.__USE
    
    @property
    def IN(self) -> Set[str]:
        return self.function.IN(block = self)
    
    @property
    def OUT(self) -> Set[str]:
        return self.function.OUT(block = self)
    
    @property
    def constraints(self) -> Dict[str, Dict[str, Union[str, List[str]]]]:
        if self.__constraints is None:
            constraints: Dict[str, Dict[str, Union[str, List[str]]]] = OrderedDict()
            for i, stmt in enumerate(self.codeSplit):
                if ifStatement.fullmatch(string = stmt) is not None:
                    matcher: Match = ifStatement.fullmatch(string = stmt)
                    cond: str = matcher.group('cond').strip()
                    for op in ('==', '!=', '<', '>', '<=', '>='):
                        for matcher in operations[op].finditer(string = cond):
                            stmt: str = '{} {} {}'.format(matcher.group('arg1'), op, matcher.group('arg2'))
                            constraints[stmt] = {'stmt': stmt,
                                                 'type': 'condition',
                                                 'op': op,
                                                 'arg1': matcher.group('arg1'),
                                                 'arg2': matcher.group('arg2'),
                                                 'args': [matcher.group('arg1'), matcher.group('arg2')],
                                                 'true': gotoStatement.fullmatch(string = self.codeSplit[i + 1]).group('label'),
                                                 'false': gotoStatement.fullmatch(string = self.codeSplit[i + 3]).group('label')}
                    continue
                elif phiStatement.fullmatch(string = stmt) is not None:
                    matcher: Match = phiStatement.fullmatch(string = stmt)
                    stmt: str = '# {} = PHI <{}, {}>'.format(matcher.group('res'), matcher.group('arg1'), matcher.group('arg2'))
                    constraints[stmt] = {'stmt': stmt,
                                         'type': 'PHI',
                                         'op': 'PHI',
                                         'res': matcher.group('res'),
                                         'arg1': matcher.group('arg1'),
                                         'arg2': matcher.group('arg2'),
                                         'args': [matcher.group('arg1'), matcher.group('arg2')], }
                elif variableAssignment.fullmatch(string = stmt) is not None:
                    matcher: Match = variableAssignment.fullmatch(string = stmt)
                    res: str = matcher.group('res')
                    expr: str = matcher.group('expr').strip()
                    if factor.fullmatch(string = expr) is not None:
                        stmt: str = '{} = {}'.format(res, expr)
                        constraints[stmt] = {'stmt': stmt,
                                             'type': 'assignment',
                                             'op': 'assign',
                                             'res': res,
                                             'arg1': expr,
                                             'args': [expr]}
                    elif functionCall.fullmatch(string = expr) is not None:
                        matcher: Match = functionCall.fullmatch(string = expr)
                        args: List[str] = list(map(str.strip, matcher.group('args').split(',')))
                        stmt: str = '{} = {}({})'.format(res, matcher.group('name'), ', '.join(args))
                        constraints[stmt] = {'stmt': stmt,
                                             'type': 'funcCall',
                                             'op': matcher.group('name'),
                                             'res': res,
                                             'args': args}
                    else:
                        for op in arithmeticOperators:
                            matcher: Match = operations[op].fullmatch(string = expr)
                            if matcher is not None:
                                stmt: str = '{} = {} {} {}'.format(res, matcher.group('arg1'), op,
                                                                   matcher.group('arg2'))
                                constraints[stmt] = {'stmt': stmt,
                                                     'type': 'arithmetic',
                                                     'op': op,
                                                     'res': res,
                                                     'arg1': matcher.group('arg1'),
                                                     'arg2': matcher.group('arg2'),
                                                     'args': [matcher.group('arg1'), matcher.group('arg2')]}
                        for op in monocularOperators:
                            matcher: Match = operations[op].fullmatch(string = expr)
                            if matcher is not None:
                                stmt: str = '{} = {} {}'.format(res, op, matcher.group('arg1'))
                                constraints[stmt] = {'stmt': stmt,
                                                     'type': 'monocular',
                                                     'op': op,
                                                     'res': res,
                                                     'arg1': matcher.group('arg1'),
                                                     'args': [matcher.group('arg1')]}
            for constraint in constraints.values():
                constraint['blockLabel']: str = self.label
            self.__constraints: Dict[str, Dict[str, Union[str, List[str]]]] = constraints
        return self.__constraints
    
    @property
    def transferCondition(self) -> Optional[str]:
        return self.__transferCondition
    
    @property
    def trueList(self) -> List[str]:
        return self.__trueList
    
    @property
    def falseList(self) -> List[str]:
        return self.__falseList
    
    @property
    def nextList(self) -> List[str]:
        if self.__nextList is None:
            self.__nextList: List[str] = list()
            if self.transferCondition is None:
                try:
                    self.nextList.append(gotoStatement.fullmatch(string = self.codeSplit[-1]).group('label'))
                except AttributeError:
                    try:
                        self.nextList.append(self.function.blockLabels[self.function.blockLabels.index(self.label) + 1])
                    except IndexError:
                        pass
        return self.__nextList
    
    @property
    def successorLabels(self) -> Set[str]:
        return self.function.successorLabelsOf(block = self)
    
    @property
    def successors(self) -> Set[Block]:
        return self.function.successorsOf(block = self)
    
    @property
    def predecessorLabels(self) -> Set[str]:
        return self.function.predecessorLabelsOf(block = self)
    
    @property
    def predecessors(self) -> Set[Block]:
        return self.function.predecessorsOf(block = self)
    
    def __str__(self) -> str:
        return self.label
    
    __repr__ = __str__


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
                        traveledBlocks: Set[Block] = set()
                        successors: Set[Block] = {func.blocks[constraint[flag]]}
                        while len(successors) > 0:
                            successor: Block = successors.pop()
                            if successor in traveledBlocks:
                                continue
                            traveledBlocks.add(successor)
                            for arg in constraint['args']:
                                if number.fullmatch(string = arg) is not None:
                                    continue
                                else:
                                    if arg in successor.USE:
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
                                    if arg not in successor.KILL and arg not in func.varsFromArg:
                                        successors.update(successor.successors)
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
