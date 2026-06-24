"""Tiny inline-math compiler: a LaTeX-ish source string -> OMML inner XML.

Used so that variables in running text (``$x_e$``, ``$\\hat\\iota_k$``,
``$r\\dot\\alpha_k(1-s_k)$``) render as genuine Word math (math-italic letters,
proper accents and subscripts) instead of fragile Unicode combining glyphs.
Supports exactly the constructs the paper needs:

    letters/digits/operators   a  3  + - = / , . | ( ) [ ]
    Greek                       \\alpha \\theta \\omega \\gamma \\lambda \\iota ...
    relations / symbols         \\to \\approx \\in \\ge \\le \\gg \\ll \\times \\cdot
    subscript / superscript     x_e   s_{k}   10^{-4}   x^2
    accents                     \\dot{\\alpha}  \\hat{\\iota}  \\hat{s}_L
    grouping                    { ... }
"""
from paper import omml

_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "theta": "θ",
    "lambda": "λ", "mu": "μ", "iota": "ι", "omega": "ω", "tau": "τ",
    "pi": "π", "sigma": "σ", "phi": "φ", "rho": "ρ",
}
_SYM = {
    "to": "→", "approx": "≈", "in": "∈", "ge": "≥", "le": "≤",
    "gg": "≫", "ll": "≪", "times": "×", "cdot": "·", "pm": "±",
    "leftarrow": "←", "neq": "≠", "infty": "∞",
}
_ACC = {"dot": omml.DOT, "hat": omml.HAT, "ddot": omml.DDOT}


class _P:
    def __init__(self, s):
        self.s = s
        self.i = 0

    def eof(self):
        return self.i >= len(self.s)

    def peek(self):
        return self.s[self.i] if not self.eof() else ""

    def _atom(self):
        """Parse one atom (no trailing sub/sup), return OMML inner."""
        c = self.peek()
        if c == "\\":
            return self._command()
        if c == "{":
            self.i += 1
            inner = self._group_until("}")
            return inner
        # a single literal character (letter, digit, operator, paren, space)
        self.i += 1
        if c == " ":
            return '<m:r><m:t xml:space="preserve"> </m:t></m:r>'
        return omml.t(c)

    def _command(self):
        self.i += 1  # consume backslash
        name = ""
        while not self.eof() and self.s[self.i].isalpha():
            name += self.s[self.i]
            self.i += 1
        if name in _ACC:
            # accent applies to the next atom
            self._skip_spaces()
            base = self._postfix(self._atom())
            return omml.acc(base, _ACC[name])
        if name in _GREEK:
            return omml.t(_GREEK[name])
        if name in _SYM:
            return omml.t(_SYM[name])
        # unknown command: emit literally (best effort)
        return omml.t("\\" + name)

    def _skip_spaces(self):
        while self.peek() == " ":
            self.i += 1

    def _postfix(self, base):
        """Attach any _sub / ^sup chain to a parsed base atom."""
        while self.peek() in ("_", "^"):
            op = self.peek()
            self.i += 1
            arg = self._atom()
            if op == "_":
                base = omml.sub(base, arg)
            else:
                base = omml.sup(base, arg)
        return base

    def _group_until(self, end):
        out = []
        while not self.eof() and self.peek() != end:
            out.append(self._postfix(self._atom()))
        if self.peek() == end:
            self.i += 1
        return "".join(out)

    def parse(self):
        out = []
        while not self.eof():
            out.append(self._postfix(self._atom()))
        return "".join(out)


def compile_inline(src):
    """Compile a LaTeX-ish inline expression to OMML inner XML."""
    return _P(src).parse()


def split_text(text):
    """Yield (is_math, chunk) over a string with ``$...$`` math spans.

    A literal dollar sign can be written ``\\$``.
    """
    parts = []
    buf = []
    math = False
    i = 0
    while i < len(text):
        c = text[i]
        if c == "\\" and i + 1 < len(text) and text[i + 1] == "$":
            buf.append("$")
            i += 2
            continue
        if c == "$":
            parts.append((math, "".join(buf)))
            buf = []
            math = not math
            i += 1
            continue
        buf.append(c)
        i += 1
    parts.append((math, "".join(buf)))
    return [(m, chunk) for m, chunk in parts if chunk != ""]
