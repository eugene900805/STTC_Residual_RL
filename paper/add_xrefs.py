"""Add Word cross-reference fields (Fig / Table / equation) to the final-project
docx: bookmark every caption / equation number, then replace each in-text
mention with a hyperlinked REF field so it is clickable and auto-updatable.

    python3 paper/add_xrefs.py  <in.docx>  <out.docx>
"""
import copy
import re
import sys

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
_bid = [2000]


def _new_id():
    _bid[0] += 1
    return _bid[0]


# -- run builders ----------------------------------------------------------- #
def _clone_rpr(r):
    rpr = r.find(qn("w:rPr"))
    return copy.deepcopy(rpr) if rpr is not None else None


def _run(rpr, text):
    r = OxmlElement("w:r")
    if rpr is not None:
        r.append(copy.deepcopy(rpr))
    t = OxmlElement("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = text
    r.append(t)
    return r


def _fld(rpr, typ):
    r = OxmlElement("w:r")
    if rpr is not None:
        r.append(copy.deepcopy(rpr))
    fc = OxmlElement("w:fldChar")
    fc.set(qn("w:fldCharType"), typ)
    r.append(fc)
    return r


def _instr(rpr, txt):
    r = OxmlElement("w:r")
    if rpr is not None:
        r.append(copy.deepcopy(rpr))
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = txt
    r.append(it)
    return r


def _ref_field(rpr, bookmark, display):
    return [_fld(rpr, "begin"),
            _instr(rpr, f" REF {bookmark} \\h "),
            _fld(rpr, "separate"),
            _run(rpr, display),
            _fld(rpr, "end")]


# -- paragraph surgery ------------------------------------------------------ #
def _find_run(p, pattern):
    for r in p._p.findall(qn("w:r")):
        t = r.find(qn("w:t"))
        if t is None or t.text is None:
            continue
        m = pattern.search(t.text)
        if m:
            return r, t, m
    return None, None, None


def replace_phrase(p, pattern, parts):
    """Replace first match of `pattern` with `parts`: list of ('t',text) or
    ('ref',bookmark,display)."""
    r, t, m = _find_run(p, pattern)
    if r is None:
        raise ValueError(f"phrase {pattern.pattern!r} not found")
    text = t.text
    rpr = _clone_rpr(r)
    right = text[m.end():]
    t.text = text[:m.start()]
    nodes = []
    for pt in parts:
        if pt[0] == "t":
            nodes.append(_run(rpr, pt[1]))
        else:
            nodes += _ref_field(rpr, pt[1], pt[2])
    nodes.append(_run(rpr, right))
    parent = r.getparent()
    idx = list(parent).index(r)
    for off, el in enumerate(nodes, start=1):
        parent.insert(idx + off, el)


def _seq_field(rpr, category, fmt, cached):
    """A { SEQ <category> \\* <fmt> } field showing `cached` until updated."""
    return [_fld(rpr, "begin"),
            _instr(rpr, f" SEQ {category} \\* {fmt} "),
            _fld(rpr, "separate"),
            _run(rpr, cached),
            _fld(rpr, "end")]


def bookmark_number(p, pattern, name, category=None, fmt="ARABIC"):
    """Wrap group(1) of `pattern` in a bookmark.  If `category` is given, the
    static number is replaced by a live { SEQ } auto-number field so Word's
    cross-reference system recognises it."""
    r, t, m = _find_run(p, pattern)
    if r is None:
        raise ValueError(f"bookmark target {pattern.pattern!r} not found")
    text = t.text
    rpr = _clone_rpr(r)
    a, b = m.start(1), m.end(1)
    cached = text[a:b]
    right = text[b:]
    t.text = text[:a]
    bid = _new_id()
    bs = OxmlElement("w:bookmarkStart")
    bs.set(qn("w:id"), str(bid))
    bs.set(qn("w:name"), name)
    be = OxmlElement("w:bookmarkEnd")
    be.set(qn("w:id"), str(bid))
    inner = _seq_field(rpr, category, fmt, cached) if category else [_run(rpr, cached)]
    nodes = [bs] + inner + [be, _run(rpr, right)]
    parent = r.getparent()
    idx = list(parent).index(r)
    for off, el in enumerate(nodes, start=1):
        parent.insert(idx + off, el)


# -- main ------------------------------------------------------------------- #
def main(inp, outp):
    d = Document(inp)
    paras = d.paragraphs

    # 1) bookmark caption / equation numbers ------------------------------- #
    nb = " "
    fig_pat = re.compile(r"Fig\.[  ](\d+)")
    tab_pat = re.compile(r"TABLE\s+([IVX]+)")
    eq_pat = re.compile(r"\((\d+)\)")
    for p in paras:
        s = p.style.name
        if s == "Figure Caption":
            m = fig_pat.search(p.text)
            if m:
                bookmark_number(p, fig_pat, f"fig{int(m.group(1))}",
                                category="Figure", fmt="ARABIC")
        elif s == "Table Title":
            m = tab_pat.search(p.text)
            if m:
                bookmark_number(p, tab_pat, f"tbl{ROMAN[m.group(1)]}",
                                category="Table", fmt="ROMAN")
        elif s == "Equation":
            m = eq_pat.search(p.text)
            if m:
                bookmark_number(p, eq_pat, f"eq{int(m.group(1))}",
                                category="Equation", fmt="ARABIC")

    # 2) in-text references ------------------------------------------------ #
    def tab(roman, n):
        return (re.compile(rf"Table {roman}(?![IVX])"),
                [("t", "Table "), ("ref", f"tbl{n}", roman)])

    tasks = [
        (13, re.compile(rf"Fig\.{nb}2"), [("t", f"Fig.{nb}"), ("ref", "fig2", "2")]),
        (58, *tab("I", 1)),
        (72, *tab("II", 2)),
        (76, *tab("III", 3)),
        (76, re.compile(rf"Fig\.{nb}5"), [("t", f"Fig.{nb}"), ("ref", "fig5", "5")]),
        (86, re.compile(r"Figure 4"), [("t", "Figure "), ("ref", "fig4", "4")]),
        (86, re.compile(r"Fig\. 3"), [("t", "Fig. "), ("ref", "fig3", "3")]),
        (87, re.compile(r"Figs\. 5 and 6"),
         [("t", "Figs. "), ("ref", "fig5", "5"), ("t", " and "), ("ref", "fig6", "6")]),
        (94, *tab("IV", 4)),
        (94, re.compile(rf"Figs\.{nb}7[–-]9"),
         [("t", f"Figs.{nb}"), ("ref", "fig7", "7"), ("t", "–"), ("ref", "fig9", "9")]),
        (121, *tab("V", 5)),
        (126, *tab("II", 2)),
        (126, *tab("IV", 4)),
        (41, re.compile(r"\(4\)"), [("t", "("), ("ref", "eq4", "4"), ("t", ")")]),
        (42, re.compile(r"\(5\)"), [("t", "("), ("ref", "eq5", "5"), ("t", ")")]),
        (90, re.compile(r"\(6\)"), [("t", "("), ("ref", "eq6", "6"), ("t", ")")]),
    ]
    for idx, pat, parts in tasks:
        replace_phrase(paras[idx], pat, parts)

    d.save(outp)
    print("saved", outp)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
