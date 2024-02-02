import pedata as ped
import datasets as ds


# Example 1: estimate offset from mutation codes
dataset = ds.Dataset.from_dict(
    {"aa_mut": ["C9X_E11Y", "D10K"], "aa_seq": [None, None], "target foo": [1, 2]}
)
parent_seq = "ABCDEFGH"  # TODO: how to get it out of the dataset?
offset_est = ped.mutation.Mutation.estimate_offset(dataset, parent=parent_seq)
print(offset_est.to_pandas())
# Results in:
#   offset  matching_ratio
# 0      -6             1.0
# 1      -3             0.0
# 2      -4             0.0
# 3      -5             0.0
# 4      -7             0.0
# 5      -8             0.0

# the offset here is -6, because  for this offset the matching ratio is 1.0
# if several offsets have matching ratio 1.0, then the offset is ambiguous
# – in this case, ask the user to pick one of the offsets
offset = offset_est[0]["offset"]

# if you want to get the most likely offset directly,
# in this case -6, use the most_likely argument
print(
    ped.mutation.Mutation.estimate_offset(dataset, parent=parent_seq, most_likely=True)
)

# Example 2: parse all mutations with the estimated offset, resulting in a list of mut tuples

parsed, parse_errors = ped.mutation.Mutation.parse_all_mutations(dataset, offset=offset)

# there are no parse errors in this example – if there are, the parse_errors variable will contain a list of tuples (index, error)
print(parsed)

# Results in:
# [[Mut(pos=2, src='C', targ='X'), Mut(pos=4, src='E', targ='Y')], [Mut(pos=3, src='D', targ='K')]]
# These positions are now 0-based (i.e. the offset has been subtracted from the original positions)

# Example 3: parse a single variants mutations with corrected offset
print(ped.mutation.Mutation.parse_variant_mutations("D10K_E11Y", offset=offset))

# Results in:
# [Mut(pos=3, src='D', targ='K'), Mut(pos=4, src='E', targ='Y')]

# Example 4: generate a mutation tuples from aa sequences
# you do not give an offset since you want zero-based positions
(
    mut,
    invalid,
) = ped.mutation.mutation_extractor.extract_mutation_namedtuples_from_sequences(
    ["AGCDHFGH", "ABCKEFGH"],
    parent_seq,
)
print(mut)

# Results in:
# [[Mut(pos=1, src='B', targ='G'), Mut(pos=4, src='E', targ='H')],
#  [Mut(pos=3, src='D', targ='K')]]

# Example 5: generate a mutation codes from aa sequences and an offset.
# This can be used to show the user the mutations.
mut = ped.mutation.mutation_extractor.extract_mutation_str_from_sequences(
    ["AGCDHFGH", "ABCKEFGH"], parent_seq, offset=offset
)
print(mut)
