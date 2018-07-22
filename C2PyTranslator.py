import re
from collections import deque
from typing import Type, List, Deque, Dict, Pattern, Match


types: Dict[str, Type] = {'int': int, 'float': float}
wordSplitters: str = r'[\s,.?;:\'\"\\|~!@#$%^&+\-*/=<>{}\[\]()]'
functionDeclaration: Pattern = re.compile(r'(?P<type>int|float)\s+(?P<name>\w+)\s*\((?P<args>[\w\s,]*)\)')
variableDeclaration: Pattern = re.compile(r'(?P<type>int|float)\s+(?P<id>\w+)\s*' + wordSplitters)
variableAssignment: Pattern = re.compile(r'^(?P<id>\w+(_\d+)?)\s*=[^=]')
ifStatement: Pattern = re.compile(r'if\s*\((?P<condition>[\w\s<>=!]+)\)')
elseStatement: Pattern = re.compile(r'else')
whileStatement: Pattern = re.compile(r'while\s*\((?P<condition>[\w\s=<>!+\-*/]+)\)')
doStatement: Pattern = re.compile(r'do')
forStatement: Pattern = re.compile(r'for\s*\((?P<init>[\w\s=+\-]+);(?P<condition>[\w\s=<>!+\-*/]+);(?P<update>[\w\s=+\-]+)\)')


def formatCode(cCode: str) -> List[str]:
    cCode: str = re.sub(r'/\*.*\*/', repl = '', string = cCode, flags = re.DOTALL)
    cCode: str = re.sub(r'{', repl = '\n{\n', string = cCode)
    cCode: str = re.sub(r'}', repl = '\n}\n', string = cCode)
    cCode: str = re.sub(r',\s*(?P<assignment>(?P<id>\w+)\s+=[^=])', repl = lambda m: ';\n{}'.format(m.group('assignment')), string = cCode)
    cCode: str = re.sub(r';\s*\n', repl = '\n', string = cCode)
    cCode: str = re.sub(r'(int|float)\s+(?P<assignment>(?P<id>\w+)\s+=[^=])', repl = lambda m: m.group('assignment'), string = cCode)
    cCode: str = re.sub(r'\+\+\s*(?P<id>\w+)', repl = lambda m: '{} += 1'.format(m.group('id')), string = cCode)
    cCode: str = re.sub(r'(?P<id>\w+)\s*\+\+', repl = lambda m: '{} += 1'.format(m.group('id')), string = cCode)
    cCode: str = re.sub(r'--\s*(?P<id>\w+)', repl = lambda m: '{} -= 1'.format(m.group('id')), string = cCode)
    cCode: str = re.sub(r'(?P<id>\w+)\s*--', repl = lambda m: '{} -= 1'.format(m.group('id')), string = cCode)
    return list(filter(None, map(str.strip, cCode.splitlines())))


def readCFile(file: str) -> List[str]:
    with open(file = file, mode = 'r', encoding = 'UTF-8') as cFile:
        return list(map(str.rstrip, cFile))


def translateBlock(depth: int, cCodeSplit: Deque[str], pyCodeSplit: List[str]) -> None:
    cCode: str = cCodeSplit.popleft()
    indentation: str = '    ' * depth
    if cCode == '{':
        while cCodeSplit[0] != '}':
            translateBlock(depth = depth, cCodeSplit = cCodeSplit, pyCodeSplit = pyCodeSplit)
        cCodeSplit.popleft()
        return
    elif ifStatement.search(string = cCode) is not None:
        pyCodeSplit.append('{}if {}:'.format(indentation, ifStatement.search(string = cCode).group('condition').strip()))
        translateBlock(depth = depth + 1, cCodeSplit = cCodeSplit, pyCodeSplit = pyCodeSplit)
        return
    elif elseStatement.search(string = cCode) is not None:
        pyCodeSplit.append('{}else:'.format(indentation))
        translateBlock(depth = depth + 1, cCodeSplit = cCodeSplit, pyCodeSplit = pyCodeSplit)
        return
    elif whileStatement.search(string = cCode) is not None:
        condition: str = whileStatement.search(string = cCode).group('condition').strip()
        pyCodeSplit.append('{}while {}:'.format(indentation, condition))
        translateBlock(depth = depth + 1, cCodeSplit = cCodeSplit, pyCodeSplit = pyCodeSplit)
        return
    elif doStatement.search(string = cCode) is not None:
        pyCodeSplit.append('{}while True:'.format(indentation))
        translateBlock(depth = depth + 1, cCodeSplit = cCodeSplit, pyCodeSplit = pyCodeSplit)
        condition: str = whileStatement.search(string = cCodeSplit.popleft()).group('condition').strip()
        pyCodeSplit.append('{}    if {}:'.format(indentation, condition))
        pyCodeSplit.append('{}        continue'.format(indentation))
        pyCodeSplit.append('{}    else:'.format(indentation))
        pyCodeSplit.append('{}        break'.format(indentation))
        return
    elif forStatement.search(string = cCode) is not None:
        matcher: Match = forStatement.search(string = cCode)
        pyCodeSplit.append('{}{}'.format(indentation, matcher.group('init').strip()))
        pyCodeSplit.append('{}while {}:'.format(indentation, matcher.group('condition').strip()))
        translateBlock(depth = depth + 1, cCodeSplit = cCodeSplit, pyCodeSplit = pyCodeSplit)
        pyCodeSplit.append('{}    {}'.format(indentation, matcher.group('update').strip()))
        return
    elif functionDeclaration.search(string = cCode) is not None:
        matcher: Match = functionDeclaration.search(string = cCode)
        args: str = ', '.join('{}: {}'.format(m.group('id'), m.group('type'))
                              for m in variableDeclaration.finditer(string = matcher.group('args') + ','))
        pyCodeSplit.append('{}def {}({}) -> {}:'.format(indentation, matcher.group('name'), args, matcher.group('type')))
        translateBlock(depth = depth + 1, cCodeSplit = cCodeSplit, pyCodeSplit = pyCodeSplit)
        return
    else:
        pyCodeSplit.append('{}{}'.format('    ' * depth, cCode))
        return


def translate(cCodeSplit: List[str]) -> List[str]:
    pyCodeSplit: List[str] = list()
    cCodeSplit: Deque[str] = deque(cCodeSplit)
    flag: bool = False
    while len(cCodeSplit) > 0:
        if flag:
            pyCodeSplit.append('')
        else:
            flag: bool = True
        translateBlock(depth = 0, cCodeSplit = cCodeSplit, pyCodeSplit = pyCodeSplit)
    return pyCodeSplit


if __name__ == '__main__':
    cFile: str = 'benchmark/t{}.c'.format(9)
    cCode: str = readCFile(file = cFile)
    print('file name: {}'.format(cFile))
    print()
    print('C code:')
    print('#' * 100)
    print(cCode)
    print('#' * 100)
    print()
    
    cCodeSplit: List[str] = formatCode(cCode = cCode)
    pyCodeSplit: List[str] = translate(cCodeSplit = cCodeSplit)
    pyCode: str = '\n'.join(pyCodeSplit)
    print('Python code:')
    print('#' * 100)
    print(pyCode)
    print('#' * 100)
    print()
    
    ret: list = []
    exec('global ret\n{}\nret.append(foo())\n'.format(pyCode))
    ret.sort()
    print('ret = {}'.format(ret))
    print('bound(ret) = [{}, {}]'.format(min(ret), max(ret)))
