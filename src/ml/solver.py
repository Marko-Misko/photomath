class StackSolver:
    """
    Solves mathematical expressions containing digits and 4 basic operators and
    parentheses. Operations are executed in natural numbers.
    """

    def __init__(self):
        self._values = []
        self._operators = []

    def solve(self, expression: str) -> int:
        """
        Solves mathematical expression given as `expression`.

        :param expression: mathematical expression
        :return: result of evaluation expression
        """
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
                self._values.append(int(number))
            elif token == '(':
                self._operators.append(token)
            elif token == ')':
                while self._operators[-1] != '(':
                    self.calculate_currently_on_stacks()
                self._operators.pop()
            elif token in '+-*/':
                while self._operators and has_precedence(token, self._operators[-1]):
                    self.calculate_currently_on_stacks()
                self._operators.append(token)
            else:
                # It's a whitespace
                pass

            i += 1

        while self._operators:
            self.calculate_currently_on_stacks()

        return self._values.pop()

    def calculate_currently_on_stacks(self):
        """
        Calculates the result of next operation, implicitly given by the states
        of internal stack structures.
        """
        try:
            op = self._operators.pop()
            b = self._values.pop()
            a = self._values.pop()
        except IndexError as err:
            raise StackSolverException('Expression is not valid')
        value = apply_operator(op, a, b)
        self._values.append(value)


class StackSolverException(Exception):
    """
    Raise when solver can't solve mathematical expression.
    """


def apply_operator(op: str, a: int, b: int) -> int:
    """
    Applies mathematical operation denoted as `op` to the arguments
    `a` and `b`.

    :param op: operator
    :param a: first operand
    :param b: second operand
    :return: result of operation

    :raises InvalidArgumentError: if operation is division by zero
    """
    operators = {
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        '*': lambda x, y: x * y,
        '/': lambda x, y: x // y,
    }
    try:
        result = operators[op](a, b)
    except ZeroDivisionError:
        raise StackSolverException('Expression is not mathematically correct. There is division with zero.')
    return result


def has_precedence(op1, op2):
    """
    Determines which operator has higher priority: parentheses first,
    then multiplication and division, addition and subtraction last.

    :param op1: first operator
    :param op2: second operator
    :return: True if `op1` has higher precedence than `op2`
    """
    if op2 == '(' or op2 == ')':
        return False
    elif (op1 == '*' or op1 == '/') and (op2 == '+' or op2 == '-'):
        return False
    else:
        return True


def check_if_only_digits_and_operators(expression: str):
    """
    Raises ValueError if `expression` is not made only from allowed symbols.

    :param expression:
    :raises InvalidArgumentError: if there are illegal symbols in expression
    """
    allowed_symbols = "1234567890()+-*/ "
    for token in expression:
        if token not in allowed_symbols:
            raise StackSolverException('There are symbols which are not allowed.')


def check_if_parentheses_are_valid(expression: str):
    """
    Validates if parentheses are in valid order, which means there must be
    equal number of opening and closing operators and that every opening parentheses
    must have its closing parentheses.

    :param expression: 
    :raises: StackSolverException: if parentheses are not valid
    """
    validity_counter = 0
    for token in expression:
        if token == '(':
            validity_counter += 1
        if token == ')':
            validity_counter -= 1
        if validity_counter < 0:
            raise StackSolverException('Invalid parentheses. It looks there are opening parentheses missing.')

    if validity_counter != 0:
        raise StackSolverException('Invalid parentheses. Number of parentheses is not equal')
