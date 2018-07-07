import re
from typing import *
from typing import Pattern, Match


keywords: Set[str] = {'int', 'float', 'if', 'else', 'goto'}
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

index = 0
doStmt = False


def formatCode(statements: List[str]) -> List[str]:
    def doPreprocessing(stmt: str):
        stmt: str = re.sub(r'\s+', repl = ' ', string = stmt)
        stmt: str = re.sub(r'\s*;', repl = ';', string = stmt)
        stmt: str = re.sub(r';;.*$', repl = '', string = stmt)
        stmt: str = re.sub(r'(int|float)\s+D\.\d*\s*;', repl = '', string = stmt)
        stmt: str = re.sub(r'(?P<postfix>_\d+\s*)\(D\)', repl = lambda m: m.group('postfix'), string = stmt)
        stmt: str = stmt.strip()
        if stmt != 'else':
            return stmt
        else:
            return ''
    
    statements.insert(0, '{')
    statements.insert(-1, '}')
    code: str = '\n'.join(statements)
    code: str = re.sub(r'/\*.*\*/', repl = '', string = code, flags = re.DOTALL)
    code: str = re.sub(r'{', repl = '\n{\n', string = code)
    code: str = re.sub(r'}', repl = '\n}\n', string = code)
    code: str = re.sub(r',\s*(?P<assignment>(?P<id>\w+)\s+=[^=])', repl = lambda m: ';\n' + m.group('assignment'), string = code)
    code: str = re.sub(r'; *\n', repl = '\n', string = code)
    code: str = re.sub(r'(int|float)\s+(?P<assignment>(?P<id>\w+)\s+=[^=])', repl = lambda m: m.group('assignment'), string = code)
    code: str = re.sub(r'\+\+\s*(?P<id>\w+)', repl = lambda m: m.group('id') + ' += 1', string = code)
    code: str = re.sub(r'(?P<id>\w+)\s*\+\+', repl = lambda m: m.group('id') + ' += 1', string = code)
    code: str = re.sub(r'--\s*(?P<id>\w+)', repl = lambda m: m.group('id') + ' -= 1', string = code)
    code: str = re.sub(r'(?P<id>\w+)\s*--', repl = lambda m: m.group('id') + ' -= 1', string = code)
    statements.clear()
    statements.extend(list(filter(None, map(str.strip, code.splitlines()))))
    return statements


def readCFile(file: str) -> List[str]:
    with open(file = file, mode = 'r', encoding = 'UTF-8') as cFile:
        return list(map(str.rstrip, cFile.readlines()))


def translate(cCodeSplit: List[str]) -> List[str]:
    pyCodeSplit: List[str] = list()
    global index, doStmt
    index = 0
    doStmt = False
    translateBlock(depth = 0, cCodeSplit = cCodeSplit, pyCodeSplit = pyCodeSplit)
    return pyCodeSplit


def translateBlock(depth: int, cCodeSplit: List[str], pyCodeSplit: List[str]) -> None:
    global index, doStmt
    try:
        cCode: str = cCodeSplit[index]
    except IndexError:
        return
    if cCode == '{':
        index += 1
        while cCodeSplit[index] != '}':
            cCode: str = cCodeSplit[index]
            if ifStatement.search(string = cCode) is not None:
                matcher: Match = ifStatement.search(string = cCode)
                pyCodeSplit.append('    ' * depth + 'if ' + matcher.group('condition') + ':')
                index += 1
                translateBlock(depth + 1, cCodeSplit, pyCodeSplit)
            elif elseStatement.search(string = cCode) is not None:
                pyCodeSplit.append('    ' * depth + cCode)
                index += 1
                translateBlock(depth + 1, cCodeSplit, pyCodeSplit)
            elif whileStatement.search(string = cCode) is not None:
                matcher: Match = whileStatement.search(string = cCode)
                index += 1
                if not doStmt:
                    pyCodeSplit.append('    ' * depth + 'while ' + matcher.group('condition').strip() + ':')
                    translateBlock(depth + 1, cCodeSplit, pyCodeSplit)
                else:
                    doStmt = False
                    pyCodeSplit.append('    ' * (depth + 1) + 'if ' + matcher.group('condition').strip() + ':')
                    pyCodeSplit.append('    ' * (depth + 2) + 'continue')
                    pyCodeSplit.append('    ' * (depth + 1) + 'else:')
                    pyCodeSplit.append('    ' * (depth + 2) + 'break')
            elif doStatement.search(string = cCode) is not None:
                pyCodeSplit.append('    ' * depth + 'while True:')
                index += 1
                doStmt = True
                translateBlock(depth + 1, cCodeSplit, pyCodeSplit)
            elif forStatement.search(string = cCode) is not None:
                matcher: Match = forStatement.search(string = cCode)
                pyCodeSplit.append('    ' * depth + matcher.group('init').strip())
                pyCodeSplit.append('    ' * depth + 'while ' + matcher.group('condition').strip() + ':')
                index += 1
                translateBlock(depth + 1, cCodeSplit, pyCodeSplit)
                pyCodeSplit.append('    ' * (depth + 1) + matcher.group('update').strip())
            elif functionDeclaration.search(string = cCode) is not None:
                matcher: Match = functionDeclaration.search(string = cCode)
                identifiers: List[Tuple[str, str]] = list()
                for m in variableDeclaration.finditer(string = matcher.group('args') + ','):
                    identifiers.append(([m.group('id')], m.group('type')))
                args = ', '.join('{}: {}'.format(id, type) for id, type in identifiers)
                fn = matcher.group('name')
                ft = matcher.group('type')
                pyCodeSplit.append('    ' * depth + 'def {}({}) -> {}:'.format(fn, args, ft))
                index += 1
                translateBlock(depth + 1, cCodeSplit, pyCodeSplit)
            else:
                pyCodeSplit.append('    ' * depth + cCode)
                index += 1
        index += 1
    elif ifStatement.search(string = cCode) is not None:
        matcher: Match = ifStatement.search(string = cCode)
        pyCodeSplit.append('    ' * depth + 'if ' + matcher.group('condition') + ':')
        index += 1
        translateBlock(depth + 1, cCodeSplit, pyCodeSplit)
        return
    elif elseStatement.search(string = cCode) is not None:
        pyCodeSplit.append('    ' * depth + cCode)
        index += 1
        translateBlock(depth + 1, cCodeSplit, pyCodeSplit)
        return
    elif whileStatement.search(string = cCode) is not None:
        matcher: Match = whileStatement.search(string = cCode)
        index += 1
        if not doStmt:
            pyCodeSplit.append('    ' * depth + 'while ' + matcher.group('condition').strip() + ':')
            translateBlock(depth + 1, cCodeSplit, pyCodeSplit)
        else:
            doStmt = False
            pyCodeSplit.append('    ' * (depth + 1) + 'if ' + matcher.group('condition').strip() + ':')
            pyCodeSplit.append('    ' * (depth + 2) + 'continue')
            pyCodeSplit.append('    ' * (depth + 1) + 'else:')
            pyCodeSplit.append('    ' * (depth + 2) + 'break')
        return
    elif doStatement.search(string = cCode) is not None:
        pyCodeSplit.append('    ' * depth + cCode)
        index += 1
        doStmt = True
        translateBlock(depth + 1, cCodeSplit, pyCodeSplit)
        return
    elif forStatement.search(string = cCode) is not None:
        matcher: Match = forStatement.search(string = cCode)
        pyCodeSplit.append('    ' * depth + matcher.group('init').strip())
        pyCodeSplit.append('    ' * depth + 'while ' + matcher.group('condition').strip() + ':')
        index += 1
        translateBlock(depth + 1, cCodeSplit, pyCodeSplit)
        pyCodeSplit.append('    ' * (depth + 1) + matcher.group('update').strip())
        return
    elif functionDeclaration.search(string = cCode) is not None:
        pyCodeSplit.append('    ' * depth + cCode)
        index += 1
        translateBlock(depth + 1, cCodeSplit, pyCodeSplit)
        return
    else:
        pyCodeSplit.append('    ' * depth + cCode)
        index += 1
        return


if __name__ == '__main__':
    cFile: str = 'benchmark/t%d.c'.format(9)
    cCodeSplit: List[str] = readCFile(file = cFile)
    cCodeSplit: List[str] = list(filter(str.strip, cCodeSplit))
    print('file name: {}'.format(cFile))
    print()
    print('C code:')
    print('#' * 100)
    print('\n'.join(cCodeSplit))
    print('#' * 100)
    print()
    
    formatCode(statements = cCodeSplit)
    pyCodeSplit: List[str] = translate(cCodeSplit = cCodeSplit)
    pyCode: str = '\n'.join(pyCodeSplit)
    print('Python code:')
    print('#' * 100)
    print(pyCode)
    print('#' * 100)
    print()
    
    ret = []
    exec('global ret\n{}\nret.append(foo())\n'.format(pyCode))
    ret.sort()
    print('ret = {}'.format(ret))
    print('bound(ret) = [{}, {}]'.format(min(ret), max(ret)))
