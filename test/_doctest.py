import doctest

verbose = False
RUN_ALL = False

# ========== examples ==========

# testing a file
# doctest.testfile("../src/eep/models/gpytorch_models.py", verbose=True)

# testing a module
# import eep.models.gpytorch_models as gpytorch_models
# doctest.testmod(gpytorch_models, verbose=True)

# ========== doc tests ================
# ========== pedata modules ==========
if RUN_ALL:
    import pedata.util
    import pedata.disk_cache
    import pedata.pytorch_dataloaders

    doctest.testmod(pedata.util, verbose=verbose)
    doctest.testmod(pedata.disk_cache, verbose=verbose)
    # TODO fix load_similarity example in disk_cache: needs files which do not exist
    doctest.testmod(pedata.pytorch_dataloaders, verbose=verbose)

# ========== mutation package ==========
if RUN_ALL:
    from pedata.mutation import (
        mutation,
        mutation_util,
        mutation_converter,
        mutation_extractor,
    )

    doctest.testmod(mutation, verbose=verbose)  # WORKS
    doctest.testmod(mutation_util, verbose=verbose)  # WORKS
    doctest.testmod(mutation_converter, verbose=verbose)  # WORKS
    doctest.testmod(mutation_extractor, verbose=verbose)  # WORKS

# ========== encoding package ==========
if True:
    from pedata.encoding import (
        base,
        embeddings,
        transform,
        transforms_graph,
        util,
        embeddings_transform,
    )

    doctest.testmod(embeddings, verbose=verbose)  # WORKS
    doctest.testmod(
        transform, verbose=verbose
    )  # WORKS - MORE DOCTESTS and PYTESTs NEEDED

    doctest.testmod(transforms_graph, verbose=verbose)  # WORKS
    doctest.testmod(util, verbose=verbose)

    doctest.testmod(embeddings_transform, verbose=verbose)

# ========== config package ==========
if RUN_ALL:
    from pedata.config import encoding_specs

    doctest.testmod(encoding_specs, verbose=verbose)


# ========== datasets package ========== #FIXME
if True:
    # from pedata. import utils
    # from pedata. import vis
    # from pedata.preprocessing import split

    # doctest.testmod(utils, verbose=verbose)
    # doctest.testmod(vis, verbose=verbose)
    # doctest.testmod(split, verbose=verbose)
