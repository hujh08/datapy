Module to handle linear constraints

# Motivation

This module aims to provide ways to hold linear constraints, including equalities and inequalites. Meanwhile any conflicts should be noticed at once when new constraint added.

# Equality constraint

Equality constraints are held by expressing one of variables as linear combination of others. From this idea, in global, some variables in control are chosen as basis, which are linearly independent. And then other variables are linear combination of basis.

As new equality constraint is introduced, 2 cases might happend. If there exists new variable, then this new variable could be expressed via basis (or there are more than one unknown variables, then new basis may be added). This is trivial situation. Otherwise, all variables are known, after substituting with basis, it is actually restriction among basis (or just conflict or identical case). Then one of basis vars could be removed, rewritting as combination of left variables.

Through this dynamical procedure, variables and possible conflict could be traced flexibly. Relation between combinations, like ratio between two terms, could be calculated.

# Inequality linear constraint

Similaryly as equality, inequality constraints could be converted to lower and upper bound of basis vars. If basis vars are sorted in a order, then bound of a variable is expression with only previous vars. For example,
```
                 lc1 <= v1 <= uc1
          l21*v1+lc2 <= v2 <= uc2+u21*v1
   l31*v1+l32*v2+lc3 <= v3 <= uc2+u31*v1+u32*v2
...
```

There are 2 special cases should be noted:

- Degenerate case in var: identical lower and upper bound for a var
    for example, `v1+v2+c <= v3 <= v1+v2+c`

 - Conflict case in a var: identical coeffs, but conflict in const bias
        for example, `v1+v2+c1 <= v3 <= v1+v2+c2`, with `c2 < c1`

Note the observation:
set
```
    {(y, x1, x2, ..) | y <= ui(x1, ..), y>= lj(x1, ..),
                       bk(x1, x2, ..) <= 0}
```
is non-empty if and only if
set
```
    {(x1, x2, ..) | lj(x1, ..) <= ui(x1, ..) for all (i, j),
                    bk(x1, x2, ..) <= 0}
```
is non-empty.

It is also true by substituing with strict inequality, like '>=' to '>'

Then if additional inequality from pair of lower and upper bounds are also included in bounds of previous vars, global conflict and degenerate cases could be easily detected recursively from lower and lower vars in the predefined order.