'''
Exact polynomial arithmetic over the rationals.

Coefficients are stored from lowest to highest degree.

>>> p = Poly([1, 2, 3])
>>> print(p)
3x**2 + 2x + 1

Polynomials are just functions with special structure,
and they can be called like regular functions.

>>> p(3)
34

Because of this special structure, we can do a lot of operations that can't be done on ordinary functions.

We can compute derivatives:

>>> print(p.derivative())
6x + 2

We can check for equality:

>>> q = Poly([0, 3, 2, 1])
>>> print(q)
x**3 + 2x**2 + 3x
>>> p == q
False

We can do addition, subtraction, multiplication, and scalar exponentiation:

>>> print (p + q)
x**3 + 5x**2 + 5x + 1
>>> print (p - q)
-x**3 + x**2 - x + 1
>>> print (p * q)
3x**5 + 8x**4 + 14x**3 + 8x**2 + 3x
>>> print(p ** 2)
9x**4 + 12x**3 + 10x**2 + 4x + 1

Technically, polynomials form a *Ring*
(a type of structure studied in MATH171 Abstract Algebra),
and so they do not support division.
(If they did, they would be called a *Field*.)
'''


class Poly:
    '''
    A dense polynomial with rational coefficients.

    The field .coeffs is a list of length at least 1, from lowest to
    highest degree, with no trailing zeros.  Coeffs[-1] is the leading
    coefficient.

    >>> p = Poly([1, 2, 3])
    >>> p.coeffs
    [1, 2, 3]
    >>> Poly([1, 2, 0, 0]).coeffs
    [1, 2]
    >>> Poly([]).coeffs
    [0]

    The `==` operator compares polynomials by value, and polynomials
    that represent the same function are equal regardless of how their
    coefficient lists were passed in:

    >>> Poly([1, 2, 3]) == Poly([1, 2, 3, 0, 0])
    True
    >>> Poly([1, 2]) == 1
    False

    `Poly` is a mutable value type, so it is deliberately unhashable
    and cannot be used as a dict key or a set element:

    >>> hash(Poly([1, 2, 3]))
    Traceback (most recent call last):
        ...
    TypeError: unhashable type: 'Poly'

    The `repr` is the constructor call, so `eval(repr(p)) == p` for
    every Poly.  The `str` is the usual algebraic notation, with terms
    in descending degree, unit coefficients suppressed, and zero
    coefficients skipped.  The zero polynomial prints as `0`:

    >>> eval(repr(Poly([1, 2, 3]))) == Poly([1, 2, 3])
    True
    >>> print(Poly([1, 2, 3]))
    3x**2 + 2x + 1
    >>> print(Poly([0]))
    0
    >>> print(Poly([5]))
    5
    >>> print(Poly([-5]))
    -5
    >>> print(Poly([1, 1]))
    x + 1
    >>> print(Poly([0, 1]))
    x
    >>> print(Poly([0, 0, -1]))
    -x**2
    >>> print(Poly([1, -1, 0, 1]))
    x**3 - x + 1
    >>> print(Poly([1, 0, 2]))
    2x**2 + 1
    '''
    def __init__(self, coeffs):
        self.coeffs = list(coeffs)
        while len(self.coeffs) > 1 and self.coeffs[-1] == 0:
            self.coeffs.pop()
        if len(self.coeffs) == 0:
            self.coeffs = [0]

    def __eq__(self, other):
        if not isinstance(other, Poly):
            return NotImplemented
        return self.coeffs == other.coeffs

    def __repr__(self):
        return f'Poly({self.coeffs})'

    def __call__(self, x):
        return self.eval_horner(x)

    def __str__(self):
        terms = []
        for i in range(len(self.coeffs) - 1, -1, -1):
            c = self.coeffs[i]
            if c == 0:
                continue
            if i == 0:
                var = ''
            elif i == 1:
                var = 'x'
            else:
                var = f'x**{i}'
            if abs(c) == 1 and var != '':
                num = ''
            else:
                num = str(abs(c))
            if c < 0:
                sign = '-'
            else:
                sign = '+'
            terms.append((sign, num + var))
        if len(terms) == 0:
            return '0'
        sign, text = terms[0]
        if sign == '-':
            out = '-' + text
        else:
            out = text
        for sign, text in terms[1:]:
            out += f' {sign} {text}'
        return out

    def __add__(self, other):
        n = max(len(self.coeffs), len(other.coeffs))
        out = []
        for i in range(n):
            a = self.coeffs[i] if i < len(self.coeffs) else 0
            b = other.coeffs[i] if i < len(other.coeffs) else 0
            out.append(a + b)
        return Poly(out)

    def __neg__(self):
        return Poly([-c for c in self.coeffs])

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        return self.mul_naive(other)

    def derivative(self):
        return Poly([i * c for i, c in enumerate(self.coeffs)][1:])

    def __pow__(self, n):
        if n < 0:
            raise ValueError('negative exponent')
        result = Poly([1])
        for _ in range(n):
            result = result * self
        return result

    def __iadd__(self, other):
        self.coeffs = (self + other).coeffs
        return self

    def __isub__(self, other):
        self.coeffs = (self - other).coeffs
        return self

    def __imul__(self, other):
        self.coeffs = (self * other).coeffs
        return self

    def __ipow__(self, n):
        self.coeffs = (self ** n).coeffs
        return self

    def mul_karatsuba(self, other):
        return Poly(_karatsuba(self.coeffs, other.coeffs))

    def eval_naive(self, x):
        '''
        Evaluate by summing a_i * x**i one term at a time.  The
        per-term `x ** i` recomputes a power each iteration, so the
        runtime is Theta(n) power lifts, each Theta(i) for the
        exponentiation, giving Theta(n**2) overall.

        >>> Poly([1, 2, 3]).eval_naive(2)
        17
        '''
        total = 0
        for i, c in enumerate(self.coeffs):
            total += c * x ** i
        return total

    def eval_horner(self, x):
        '''
        Evaluate with Horner's rule:

            (((a_n x + a_{n-1}) x + a_{n-2}) x + ...) x + a_0

        This is Theta(n) and uses only one multiply and one add per
        coefficient.  It is also numerically more stable than the
        naive evaluation when consecutive coefficients have opposite
        signs, because no large intermediate x**i is computed.

        >>> Poly([1, 2, 3]).eval_horner(2)
        17
        '''
        total = 0
        for c in reversed(self.coeffs):
            total = total * x + c
        return total

    def mul_naive(self, other):
        '''
        Naive multiplication uses Theta(n**2) multiplications.

        >>> Poly([1, 2]) .mul_naive(Poly([3, 4]))
        Poly([3, 10, 8])
        '''
        a, b = self.coeffs, other.coeffs
        if not a or not b:
            return Poly([0])
        out = [0] * (len(a) + len(b) - 1)
        for i, ai in enumerate(a):
            for j, bj in enumerate(b):
                out[i + j] += ai * bj
        return Poly(out)


def _karatsuba(a, b):
    '''
    Product of two coefficient lists by Karatsuba's algorithm.

    The inputs are coefficient lists, lowest degree first.
    The return value is the coefficient list of the product.
    The recursion splits both operands at the midpoint and
    uses only three half-size products.
    That gives the recurrence

        T(n) = 3 T(n/2) + Theta(n)
             = Theta(n^log2 3)          # by master theorem
             = O(n^1.586)               # by defn of O()

    See <https://www.geeksforgeeks.org/dsa/multiply-two-polynomials-2/> for a basic tutorial/implementation you can adapt for your code.

    >>> _karatsuba([1, 2], [3, 4])
    [3, 10, 8]
    >>> _karatsuba([1], [1])
    [1]
    >>> _karatsuba([0], [1, 2, 3])
    [0, 0, 0]
    '''

    if len(a) == 0 or len(b) == 0:
        return []
    if len(a) == 1 or len(b) == 1:
        out = [0] * (len(a) + len(b) - 1)
        for i, ai in enumerate(a):
            for j, bj in enumerate(b):
                out[i + j] += ai * bj
        return out

    m = max(len(a), len(b)) // 2
    a_low, a_high = a[:m], a[m:]
    b_low, b_high = b[:m], b[m:]

    z0 = _karatsuba(a_low, b_low)
    z2 = _karatsuba(a_high, b_high)
    z1 = _karatsuba(_add_lists(a_low, a_high), _add_lists(b_low, b_high))
    z1 = _sub_lists(_sub_lists(z1, z0), z2)

    out = [0] * (len(a) + len(b) + 2 * m)
    for i, c in enumerate(z0):
        out[i] += c
    for i, c in enumerate(z1):
        out[i + m] += c
    for i, c in enumerate(z2):
        out[i + 2 * m] += c
    return out[:len(a) + len(b) - 1]


def _add_lists(a, b):
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)
            for i in range(n)]


def _sub_lists(a, b):
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0)
            for i in range(n)]
