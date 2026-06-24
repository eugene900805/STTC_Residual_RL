"""Minimal Office Math Markup Language (OMML) builder.

Each helper returns an OMML *inner* XML string (no namespace decl); wrap the
final expression with ``omath()`` which adds the m: namespace so the result can
be parsed by python-docx and appended into a <w:p>. Letters inside <m:t> are
rendered by Word in math italic automatically, matching IEEE equation style.
"""
from xml.sax.saxutils import escape

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"

# combining accents
DOT = "̇"   # x-dot  (first derivative)
DDOT = "̈"  # x-ddot (second derivative) -> rendered as diaeresis; use two
HAT = "̂"   # x-hat  (estimate)


def t(s):
    """A math run with literal text (operators, digits, variables, Greek)."""
    return f'<m:r><m:t xml:space="preserve">{escape(str(s))}</m:t></m:r>'


def sub(base, s):
    return f"<m:sSub><m:e>{base}</m:e><m:sub>{s}</m:sub></m:sSub>"


def sup(base, s):
    return f"<m:sSup><m:e>{base}</m:e><m:sup>{s}</m:sup></m:sSup>"


def subsup(base, sb, sp):
    return (f"<m:sSubSup><m:e>{base}</m:e><m:sub>{sb}</m:sub>"
            f"<m:sup>{sp}</m:sup></m:sSubSup>")


def frac(num, den):
    return f"<m:f><m:num>{num}</m:num><m:den>{den}</m:den></m:f>"


def sqrt(e):
    return ('<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr>'
            f"<m:deg/><m:e>{e}</m:e></m:rad>")


def delim(e, beg="(", end=")"):
    return ('<m:d><m:dPr>'
            f'<m:begChr m:val="{escape(beg)}"/><m:endChr m:val="{escape(end)}"/>'
            f"</m:dPr><m:e>{e}</m:e></m:d>")


def acc(e, chr_=DOT):
    return (f'<m:acc><m:accPr><m:chr m:val="{chr_}"/></m:accPr>'
            f"<m:e>{e}</m:e></m:acc>")


def dot(e):
    return acc(e, DOT)


def hat(e):
    return acc(e, HAT)


def matrix(rows, jc="center"):
    ncol = len(rows[0])
    mpr = (f'<m:mPr><m:mcs><m:mc><m:mcPr><m:count m:val="{ncol}"/>'
           f'<m:mcJc m:val="{jc}"/></m:mcPr></m:mc></m:mcs></m:mPr>')
    body = ""
    for row in rows:
        cells = "".join(f"<m:e>{c}</m:e>" for c in row)
        body += f"<m:mr>{cells}</m:mr>"
    return f"<m:m>{mpr}{body}</m:m>"


def bmat(rows):
    return delim(matrix(rows), "[", "]")


def eqarr(lines):
    """Stacked, =-aligned equation array (a single numbered display block)."""
    body = "".join(f"<m:e>{ln}</m:e>" for ln in lines)
    return f"<m:eqArr>{body}</m:eqArr>"


def cases(lines):
    """Left-brace system of equations."""
    return ('<m:d><m:dPr><m:begChr m:val="{"/><m:endChr m:val=""/>'
            f"<m:grow m:val=\"1\"/></m:dPr><m:e>{eqarr(lines)}</m:e></m:d>")


def omath(inner):
    """Wrap inner OMML with the math namespace -> parseable oMath element."""
    return f'<m:oMath xmlns:m="{M_NS}">{inner}</m:oMath>'
