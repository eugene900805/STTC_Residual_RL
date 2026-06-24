"""The paper's display equations, expressed with the OMML builder. Returned as
a dict number -> inner OMML string so both language versions share them."""
from paper.omml import (t, sub, sup, subsup, frac, sqrt, delim, dot, hat, acc,
                        bmat, eqarr, cases)

# convenience tokens
def g(s):            # a plain symbol/operator run
    return t(s)

a_dot = dot(g("α"))                      # α̇
ak_dot = sub(dot(g("α")), g("k"))        # α̇_k  (dot on alpha then subscript)


def build():
    E = {}

    # (1) slip ratio  s_k = (r*α̇_k - v_k)/(r*α̇_k)
    rad = g("r") + sub(dot(g("α")), g("k"))
    E[1] = (sub(g("s"), g("k")) + g(" = ") +
            frac(rad + g(" − ") + sub(g("v"), g("k")), rad))

    # (2) slipping kinematics  [ẋ ẏ θ̇]^T = [[cosθ,0],[sinθ,0],[0,1]] [v ω]^T
    lhs = bmat([[dot(g("x"))], [dot(g("y"))], [dot(g("θ"))]])
    Smat = bmat([[g("cos θ"), g("0")], [g("sin θ"), g("0")], [g("0"), g("1")]])
    vvec = bmat([[g("v")], [g("ω")]])
    E[2] = lhs + g(" = ") + Smat + vvec

    # (3) pose error  e = R(θ)(q_r - q)
    evec = bmat([[sub(g("x"), g("e"))], [sub(g("y"), g("e"))],
                 [sub(g("θ"), g("e"))]])
    Rmat = bmat([[g("cos θ"), g("sin θ"), g("0")],
                 [g("− sin θ"), g("cos θ"), g("0")],
                 [g("0"), g("0"), g("1")]])
    qdiff = bmat([[sub(g("x"), g("r")) + g(" − ") + g("x")],
                  [sub(g("y"), g("r")) + g(" − ") + g("y")],
                  [sub(g("θ"), g("r")) + g(" − ") + g("θ")]])
    E[3] = evec + g(" = ") + Rmat + qdiff

    # (4) kinematic virtual control law (paper eq. 12)
    the = sub(g("θ"), g("e"))
    one_cos = g("d(1 − cos ") + the + g(")")
    line_v = (sub(g("v"), g("c")) + g(" = ") + sub(g("v"), g("r")) +
              g(" cos ") + the + g(" + ") + sub(g("H"), g("x")) +
              delim(sub(g("x"), g("e")) + g(" + ") + one_cos) + g(" − ") +
              sub(g("H"), g("s")) + the + g(" ω"))
    line_w = (sub(g("ω"), g("c")) + g(" = ") + sub(g("ω"), g("r")) + g(" + ") +
              sub(g("v"), g("r")) +
              delim(sub(g("H"), g("y")) + delim(g("1 − λ")) +
                    delim(sub(g("y"), g("e")) + g(" − d sin ") + the + g(" + ") +
                          sub(g("H"), g("s")) + the) + g(" + ") +
                    frac(g("λ"), sub(g("H"), g("s"))) + g(" sin ") + the,
                    "[", "]"))
    E[4] = eqarr([line_v, line_w])

    # (5) slip adaptive law (paper eq. 13/14) with H_e1, H_e2
    idR = sub(dot(hat(g("ı"))), g("R"))
    idL = sub(dot(hat(g("ı"))), g("L"))
    He1 = sub(g("H"), g("e1")); He2 = sub(g("H"), g("e2"))
    base = g("b ") + sub(g("x"), g("e")) + g(" + bd(1 − cos ") + the + g(")")
    l5R = (idR + g(" = ") + frac(g("1"), g("2b")) + sub(g("ρ"), g("2")) +
           delim(sub(g("v"), g("c")) + g(" + b ") + sub(g("ω"), g("c"))) +
           delim(base + g(" + ") + He1 + g(" + ") + He2))
    l5L = (idL + g(" = ") + frac(g("1"), g("2b")) + sub(g("ρ"), g("1")) +
           delim(sub(g("v"), g("c")) + g(" − b ") + sub(g("ω"), g("c"))) +
           delim(base + g(" − ") + He1 + g(" − ") + He2))
    l5h1 = (He1 + g(" = ") + sub(g("H"), g("s")) +
            delim(sub(g("y"), g("e")) + g(" − d sin ") + the + g(" + ") +
                  sub(g("x"), g("e")) + the + g(" + d") + the +
                  delim(g("1 − cos ") + the) + g(" + ") + sub(g("H"), g("s")) +
                  the))
    l5h2 = (He2 + g(" = ") + frac(g("1"), sub(g("H"), g("y"))) + g(" sin ") + the)
    E[5] = eqarr([l5R, l5L, l5h1, l5h2])

    # (6) wheel-speed command with slip compensation
    line_R = (subsup(dot(g("α")), g("R"), g("*")) + g(" = ") +
              frac(sub(hat(g("ı")), g("R")), g("r")) +
              delim(sub(g("v"), g("c")) + g(" + ") + g("b ") +
                    sub(g("ω"), g("c"))))
    line_L = (subsup(dot(g("α")), g("L"), g("*")) + g(" = ") +
              frac(sub(hat(g("ı")), g("L")), g("r")) +
              delim(sub(g("v"), g("c")) + g(" − ") + g("b ") +
                    sub(g("ω"), g("c"))))
    E[6] = eqarr([line_R, line_L])

    # (7) dynamic model (paper eq. 8):  M̄ v̇ + L̄ v + B̄ W̄ F = B̄ τ
    mac = "̄"   # combining macron (overbar)
    E[7] = (acc(g("M"), mac) + dot(g("v")) + g(" + ") + acc(g("L"), mac) +
            g(" v") + g(" + ") + acc(g("B"), mac) + acc(g("W"), mac) + g(" F") +
            g(" = ") + acc(g("B"), mac) + g(" τ"))

    # (8) RL reward
    E[8] = (sub(g("r"), g("t")) + g(" = − ") +
            delim(sup(sub(g("x"), g("e")), g("2")) + g(" + ") +
                  sup(sub(g("y"), g("e")), g("2")) + g(" + ") +
                  sup(sub(g("θ"), g("e")), g("2"))) +
            g(" − λ ") + sup(g("‖u‖"), g("2")))

    return E
