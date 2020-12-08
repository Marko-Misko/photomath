from abc import ABC, abstractmethod


class BaseSolver(ABC):

    @abstractmethod
    def solve(self, expression: str) -> float:
        pass


class StackSolver(BaseSolver):

    def __init__(self):
        self.__values = []
        self.__operators = []

    def solve(self, expression: str) -> int:
        check_if_only_digits_and_operators(expression)
        check_if_parentheses_are_valid(expression)

        i = 0
        while i < len(expression):
            token = expression[i]
            if token.isdigit():
                number = token
                if i + 1 < len(expression) and expression[i + 1].isdigit():
                    number += expression[i + 1]
                    i += 1
                self.__values.append(int(number))
            elif token == '(':
                self.__operators.append(token)
            elif token == ')':
                while self.__operators[-1] != '(':
                    self.calculate_currently_on_stacks()
                self.__operators.pop()
            elif token in '+-*/':
                while self.__operators and has_precedence(token, self.__operators[-1]):
                    self.calculate_currently_on_stacks()
                self.__operators.append(token)
            else:
                pass

            i += 1

        while self.__operators:
            self.calculate_currently_on_stacks()

        return self.__values.pop()

    def calculate_currently_on_stacks(self):
        op = self.__operators.pop()
        b = self.__values.pop()
        a = self.__values.pop()
        value = apply_operator(op, a, b)
        self.__values.append(value)


def apply_operator(op: str, a: int, b: int):
    operators = {
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        '*': lambda x, y: x * y,
        '/': lambda x, y: x // y,
    }
    try:
        result = operators[op](a, b)
    except ZeroDivisionError:
        raise ValueError('Expression is not mathematically correct. There is division with zero.')
    return result


def check_if_only_digits_and_operators(expression: str):
    allowed_symbols = "1234567890()+-*/ "
    for token in expression:
        if token not in allowed_symbols:
            raise ValueError('There are symbols which are not allowed.')


def check_if_parentheses_are_valid(expression: str):
    validity_counter = 0
    for token in expression:
        if token == '(':
            validity_counter += 1
        if token == ')':
            validity_counter -= 1
        if validity_counter < 0:
            raise ValueError('Invalid parentheses. It looks there are opening parentheses missing.')

    if validity_counter != 0:
        raise ValueError('Invalid parentheses. Number of parentheses is not equal')


def has_precedence(op1, op2):
    if op2 == '(' or op2 == ')':
        return False
    elif (op1 == '*' or op1 == '/') and (op2 == '+' or op2 == '-'):
        return False
    else:
        return True
