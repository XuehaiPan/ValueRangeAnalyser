import re
from collections import OrderedDict
from typing import Type, Union, Optional, List, Tuple, Set, Dict, Pattern, Match, Callable


__all__: List[str] = ['Function', 'Block',
                      'keywords', 'dtypes', 'monocularOperators', 'binocularOperators', 'relationalOperators', 'arithmeticOperators',
                      'wordSplitters', 'functionImplementation', 'variableDeclaration', 'blockLabel',
                      'number', 'integer', 'variable', 'factor',
                      'ifStatement', 'gotoStatement', 'returnStatement', 'variableAssignment', 'functionCall', 'phiStatement', 'operations']

keywords: Set[str] = {'int', 'float', 'if', 'else', 'goto'}
dtypes: Dict[str, Type] = {'int': int, 'float': float}

monocularOperators: Set[str] = {'minus', '(int)', '(float)'}
binocularOperators: Set[str] = {'+', '-', '*', '/', '==', '!=', '<', '>', '<=', '>='}
relationalOperators: Set[str] = {'==', '!=', '<', '>', '<=', '>='}
arithmeticOperators: Set[str] = {'+', '-', '*', '/'}
wordSplitters: str = r'[\s,.?;:\'\"\\|~!@#$%^&+\-*/=<>{}\[\]()]'

functionImplementation: Pattern = re.compile(r'(?P<name>\w+)\s*\((?P<args>[\w\s,]*)\)\s*{\s*(?P<body>[^}]*)\s*}')
variableDeclaration: Pattern = re.compile(r'(?P<type>int|float)\s+(?P<id>\w+)\s*{}'.format(wordSplitters))
blockLabel: Pattern = re.compile(r'^\s*(?P<label><[\w\s]+>)\s*:\s*$')
ifStatement: Pattern = re.compile(r'^\s*if\s*\((?P<cond>[\w\s=<>!+\-]+)\)\s*$')
gotoStatement: Pattern = re.compile(r'^\s*goto\s+(?P<label><[\w\s]+>)\s*;\s*$')
returnStatement: Pattern = re.compile(r'^\s*return\s*(?P<id>\w*)\s*;\s*$')

numPattern: str = r'(?P<number>[+\-]?\d+(\.\d+)?([Ee][+\-]?\d+)?)'
varPattern: str = r'(?P<id>\w*)_(?P<num>\d+)'
factorPattern: str = r'(\w*_\d+)|([+\-]?\d+(\.\d+)?([Ee][+\-]?\d+)?)'
var1Pattern: str = r'(?P<id1>\w*)_(?P<num1>\d+)'
var2Pattern: str = r'(?P<id2>\w*)_(?P<num2>\d+)'
num1Pattern: str = r'(?P<number1>[+-]?\d+(\.\d+)?([Ee][+-]?\d+)?)'
num2Pattern: str = r'(?P<number2>[+-]?\d+(\.\d+)?([Ee][+-]?\d+)?)'

number: Pattern = re.compile(r'^\s*{}\s*$'.format(numPattern))
integer: Pattern = re.compile(r'^\s*(?P<integer>[+\-]?\d+)\s*$')
variable: Pattern = re.compile(r'^\s*{}\s*$'.format(varPattern))
factor: Pattern = re.compile(r'^\s*{}\s*$'.format(factorPattern))

variableAssignment: Pattern = re.compile('^\s*(?P<res>{})\s*=(?P<expr>[^=;]*);\s*$'.format(varPattern))
functionCall: Pattern = re.compile(r'^\s*(?P<name>\w+)\s*\((?P<args>[\w\s+\-.,]*)\)\s*$')
phiStatement: Pattern = re.compile(r'^\s*#\s+(?P<res>{})\s*=\s*PHI\s*<\s*(?P<arg1>{})\s*,\s*(?P<arg2>{})\s*>\s*$'.format(varPattern,
                                                                                                                         var1Pattern,
                                                                                                                         var2Pattern))

operations: Dict[str, Pattern] = dict()
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
    del op, opPattern

del varPattern, numPattern, factorPattern, var1Pattern, var2Pattern, num1Pattern, num2Pattern

Block: Type = type('Block', (object,), dict())
Function: Type = type('Function', (object,), dict())


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
            self.__declaration: str = '{}({})'.format(self.name, ', '.join('{} {}'.format(dtype, var)
                                                                           for var, dtype in self.args.items()))
            self.__prototype: str = '{}({})'.format(self.name, ', '.join(self.args.values()))
            self.__variables: Dict[str, str] = Function.parseVariableDeclaration(statement = code)
            self.__localVariables: Dict[str, str] = Function.parseVariableDeclaration(statement = self.body)
            try:
                self.__ret: str = re.search('return\s*(?P<ret>\w*)\s*;', string = self.body).group('ret')
                if self.__ret == '':
                    self.__ret: str = None
            except AttributeError:
                raise ValueError
            self.__returnBlockLabel: str = None
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
    def varsFromArg(self) -> Set[str]:
        return self.__varsFromArg
    
    @property
    def ret(self) -> str:
        return self.__ret
    
    @property
    def returnBlockLabel(self) -> str:
        if self.__returnBlockLabel is None:
            for block in self.blocks.values():
                if re.search('return\s*(?P<ret>\w*)\s*;', string = block.code) is not None:
                    self.__returnBlockLabel: str = block.label
                    break
        return self.__returnBlockLabel
    
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
    
    def successorLabelsWithoutKilling(self, block: Union[str, Block], var: str) -> Set[str]:
        if isinstance(block, Block):
            block: str = block.label
        traveledBlockLabels: Set[str] = set()
        successorLabels: Set[str] = set(self.successorLabelsOf(block = block))
        while len(successorLabels) > 0:
            successorLabel: str = successorLabels.pop()
            if successorLabel in traveledBlockLabels:
                continue
            traveledBlockLabels.add(successorLabel)
            if var not in self.blocks[successorLabel].KILL:
                successorLabels.update(self.successorLabelsOf(block = successorLabel))
        return traveledBlockLabels
    
    def successorsWithoutKilling(self, block: Union[str, Block], var: str) -> Set[Block]:
        return set(map(self.blocks.get, self.successorLabelsWithoutKilling(block = block, var = var)))
    
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
    
    __repr__: Callable[[Function], str] = __str__


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
            matcher: Match = ifStatement.fullmatch(string = self.codeSplit[-4])
            if matcher is not None:
                cond: str = matcher.group('cond').strip()
                for op in ('==', '!=', '<', '>', '<=', '>='):
                    matcher: Match = operations[op].fullmatch(string = cond)
                    if matcher is not None:
                        self.__transferCondition: str = '{} {} {}'.format(matcher.group('arg1'), op, matcher.group('arg2'))
                        self.trueList.append(gotoStatement.fullmatch(string = self.codeSplit[-3]).group('label'))
                        self.falseList.append(gotoStatement.fullmatch(string = self.codeSplit[-1]).group('label'))
                        break
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
            if self.function.returnBlockLabel == self.label and self.function.ret is not None:
                self.__USE.add(self.function.ret)
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
                        matcher: Match = operations[op].fullmatch(string = cond)
                        if matcher is not None:
                            stmt: str = '{} {} {}'.format(matcher.group('arg1'), op, matcher.group('arg2'))
                            constraints[stmt] = {'stmt': stmt,
                                                 'type': 'condition',
                                                 'op': op,
                                                 'arg1': matcher.group('arg1'),
                                                 'arg2': matcher.group('arg2'),
                                                 'args': [matcher.group('arg1'), matcher.group('arg2')],
                                                 'true': gotoStatement.fullmatch(string = self.codeSplit[i + 1]).group('label'),
                                                 'false': gotoStatement.fullmatch(string = self.codeSplit[i + 3]).group('label')}
                            trueList: List[str] = [constraints[stmt]['true']]
                            falseList: List[str] = [constraints[stmt]['false']]
                            for arg in constraints[stmt]['args']:
                                if number.fullmatch(string = arg) is None:
                                    trueList.extend(self.function.successorLabelsWithoutKilling(block = trueList[0], var = arg))
                                    falseList.extend(self.function.successorLabelsWithoutKilling(block = falseList[0], var = arg))
                            constraints[stmt]['trueList']: List[str] = trueList
                            constraints[stmt]['falseList']: List[str] = falseList
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
    
    def successorLabelsWithoutKilling(self, var: str) -> Set[str]:
        return self.function.successorLabelsWithoutKilling(block = self, var = var)
    
    def successorsWithoutKilling(self, var: str) -> Set[Block]:
        return self.function.successorsWithoutKilling(block = self, var = var)
    
    def __str__(self) -> str:
        return self.label
    
    __repr__: Callable[[Block], str] = __str__
