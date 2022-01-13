Module to handle linear constraints

# Motivation

This module aims to provide ways to hold linear constraints, including equalities and inequalites. Meanwhile any conflicts should be noticed at once when new constraint added.

# Equality constraint

Equality constraints are held by expressing one of variables as linear combination of others. From this idea, in global, some variables in control are chosen as basis, which are linearly independent. And then other variables are linear combination of basis.

As new equality constraint is introduced, 2 cases might happend. If there exists new variable, then this new variable could be expressed via basis (or there are more than one unknown variables, then new basis may be added). This is trivial situation. Otherwise, all variables are known, after substituting with basis, it is actually restriction among basis (or just conflict or identical case). Then one of basis vars could be removed, rewritting as combination of left variables.

Through this dynamical procedure, variables and possible conflict could be traced flexibly. Relation between combinations, like ratio between two terms, could be calculated.
