README_BASE = """

**Business**

- Provider: *the-company-or-provider-s-name*
- Goal: *their-motivation-project*
- Licence: *extensive-details-about-how-can-we-use-it*


**Machine Learning - Properties / Targets**

- property_1
- target_1 - regression


**Machine Learning - Embeddings**

- emb_1
- emb_2


**Biology**

- Protein full name: *protein-full-name-here*
- Protein abbreviation: *protein-abreviation-here*
- Sequence length: *n*
- Organism: *organism-here*
- Base: *for-example-A1-wild-type*


**Usage for Company** 

- *provider-s-goal*
- *optional-Company-s-additional-goal*


**Notes**

Any other relevant information, literature on the dataset


**Ideas for visualizations**
Amino acid sequences
- length of the sequence
- wild-type or base sequence representation according to the amino-acid properties (hydrophobicity / hydrophilicity / charge / size / polarity / aromaticity)
- corresponding linear map of the total amount of mutations from Cterm to Nterm
- distance map (calculated from AA properties) of the different mutated sequence from the wild-type sequence (check blossum matrix)

Visualization of the target
- histogram of the target values -> DONE

Target vs distance map
- plot the target values against the distance map


**Other ideas - for later**
motifs and domains / database annotation
residues in the active sites:
- residues relevant for substrate binding
- residues relevant for co-factor binding
- good to see if we have mutations there and what they do

homology: highly conserved residues could be critical for the folding and modifying there could be a bad idea - but not necessarily

"""

README_FORMATTING = {
    "targets": "**Machine Learning - Properties / Targets**",
    "embeddings": "**Machine Learning - Embeddings**",
    "figures": "**Figures**",
    "datapoints": "**Dataset Size**",
    "available_splits": "**Available Data Splits**",
    "in_section_sep": "\n",  # between each element in a section
    "section_end": "\n\n",  # end of sections are separated by two new lines
}
