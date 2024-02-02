from collections import namedtuple


# Mutation namedtutple
Mut = namedtuple("Mut", "pos src targ")

# Mutation with no targ
Mut_with_no_targ = namedtuple("Mut", "pos src")
