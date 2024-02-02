from os.path import abspath, dirname, isfile
from dataclasses import dataclass
from datasets import Dataset
from Bio import Seq
from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy as sp

from pedata.util import load_full_dataset

"""
This is the main module of TagFinder. 
The main class to import is:

TagFinder

The main methods of the TagFinder class are:

find_tags: Finds tags in a sequence
strip_tags: Strips tags from a sequence
get_strip_report: Strips tags from a sequence and returns a dictionary report.
tag_strip_csv: Strips tags from a .csv file and adds a report of the stripped tags in a new column in the output file.
clean_dataset: Cleans a huggingface dataset by converting sequences to protein and removing tags.
"""


@dataclass
class ProteinTag:
    """
    A container representing a protein purification tag
    """

    def __init__(
        self,
        name: str,
        sequence: str,
        origin: str,
        method: str,
        reference: str,
        length: int,
        fusion: bool = False,
        leader: bool = False,
        cleavage: bool = False,
    ):
        """
        Initialize a tag from basic tag information.

        Args:
            name: The name of the tag.
            sequence: The sequence of the tag.
            origin: The origin of the tag, e.g. a protein this tag is derived from.
            method: The purification method used with this tag.
            reference: The pubmed link referencing this tag.
            length: The length of the tag.
            fusion: True if the tag is a fusion with another protein or protein domain.
            leader: True if the tag is a leader sequence used to promote secretion in mammalian systems.
            cleavage: True if the tag is a recognition site for a protease used to remove tags from proteins.
        """
        self._name = name
        self._sequence = sequence
        self._origin = origin
        self._method = method
        self._reference = reference
        self._length = length
        self._fusion = fusion
        self._leader = leader
        self._cleavage = cleavage
        if self._fusion:
            self._type = "Protein Fusion"
        elif self._leader:
            self._type = "Leader Sequence"
        elif self._cleavage:
            self._type = "Cleavage Site"
        else:
            self._type = "Purification Tag"

    @property
    def name(self) -> str:
        """getting name value"""
        return self._name

    @name.setter
    def name(self, value: bool):
        """setting name value"""
        self._name = value

    @property
    def sequence(self) -> str:
        """getting sequence value"""
        return self._sequence

    @sequence.setter
    def sequence(self, value: str):
        """setting sequence value"""
        self._sequence = value

    @property
    def origin(self) -> str:
        """getting origin value"""
        return self._origin

    @origin.setter
    def origin(self, value: str):
        """setting origin value"""
        self._origin = value

    @property
    def method(self) -> str:
        """getting method value"""
        return self._method

    @method.setter
    def method(self, value: str):
        """setting origin value"""
        self._method = value

    @property
    def reference(self) -> str:
        """getting reference value"""
        return self._reference

    @reference.setter
    def reference(self, value: str):
        """setting reference value"""
        self._reference = value

    @property
    def length(self) -> int:
        """getting length value"""
        return self._length

    @length.setter
    def length(self, value: int):
        """setting length value"""
        self._length = value

    @property
    def fusion(self) -> bool:
        """getting fusion value"""
        return self._fusion

    @fusion.setter
    def fusion(self, value: bool):
        """setting fusion value"""
        self._fusion = value

    @property
    def leader(self) -> bool:
        """getting leader value"""
        return self._leader

    @leader.setter
    def leader(self, value: bool):
        """setting leader value"""
        self._leader = value

    @property
    def cleavage(self) -> bool:
        """getting cleavage value"""
        return self._cleavage

    @cleavage.setter
    def cleavage(self, value: bool):
        """setting cleavage value"""
        self._cleavage = value

    @property
    def type(self) -> str:
        """getting type value"""
        return self._type

    @type.setter
    def type(self, value: str):
        """setting type value"""
        self._type = value

    def __repr__(self):
        """
        A nice print for a ProteinTag.

        Returns:
            info_string (str): The string that is displayed when printing a ProteinTag

        Example:
        >>> tag_finder = TagFinder()
        >>> print(tag_finder._tags[0])

        Output:
        Name:      AviTag
        Sequence:  GLNDIFEAQKIEWHE
        Length:    15
        Origin:    Synthetic peptide
        Method:    Binding to Streptavidin via biotinylation
        Reference: https://pubmed.ncbi.nlm.nih.gov/17379573/
        """
        info_string = (
            f"Name:      {self.name}\n"
            f"Sequence:  {self.sequence}\n"
            f"Length:    {self.length}\n"
            f"Type:      {self.type}\n"
            f"Origin:    {self.origin}\n"
            f"Method:    {self.method}\n"
            f"Reference: {self.reference}\n"
        )
        return info_string


class ProteinTagHit(ProteinTag):
    """
    A container representing a hit of a tag in a query sequence. This is the class that will be returned when TagFinder
    finds a tag in a sequence, and contains information of the tag and where in the sequence the tag was found.
    """

    def __init__(
        self,
        tag: ProteinTag,
        start: int,
        end: int,
        query_aln: str,
        tag_aln: str,
        location: str,
        identity: float,
        conservation: float,
    ):
        """
        Initialize the ProteinTagHit with the necessary information.

        Args:
            tag (ProteinTag): The tag that was found in the query sequence.
            start (int): The start position of the tag within the query sequence.
            end (int): The end position of the tag within the query sequence.
            query_aln (str): A string representing the query alignment to the tag.
            tag_aln (str): A string representing the tag alignment to the query.
            location (str): The location of the tag. This can be N-Terminus, C-Terminus or Internal.
            identity (float): The sequence identity between the tag and a query sequence.
            conservation (float): The conservation of a tag relative to a reference sequence.
        """
        super(ProteinTagHit, self).__init__(
            name=tag.name,
            sequence=tag.sequence,
            origin=tag.origin,
            method=tag.method,
            reference=tag.reference,
            length=tag.length,
            fusion=tag.fusion,
            leader=tag.leader,
            cleavage=tag.cleavage,
        )
        self._start = start
        self._end = end
        self._query_aln = query_aln
        self._tag_aln = tag_aln
        self._location = location
        self._identity = identity
        self._conservation = conservation

    @property
    def start(self) -> int:
        """getting start value"""
        return self._start

    @start.setter
    def start(self, value: int):
        """setting start value"""
        self._start = value

    @property
    def end(self) -> int:
        """getting end value"""
        return self._end

    @end.setter
    def end(self, value: int):
        """setting end value"""
        self._end = value

    @property
    def query_aln(self) -> str:
        """getting query_aln value"""
        return self._query_aln

    @query_aln.setter
    def query_aln(self, value: str):
        """setting query_aln value"""
        self._query_aln = value

    @property
    def tag_aln(self) -> str:
        """getting tag_aln value"""
        return self._tag_aln

    @tag_aln.setter
    def tag_aln(self, value: str):
        """setting tag_aln value"""
        self._tag_aln = value

    @property
    def location(self) -> str:
        """getting location value"""
        return self._location

    @location.setter
    def location(self, value: str):
        """setting location value"""
        self._location = value

    @property
    def identity(self) -> float:
        """getting identity value"""
        return self._identity

    @identity.setter
    def identity(self, value: float):
        """setting identity value"""
        self._identity = value

    @property
    def conservation(self) -> float:
        """getting conservation value"""
        return self._conservation

    @conservation.setter
    def conservation(self, value: float):
        """setting conservation value"""
        self._conservation = value

    def __repr__(self) -> str:
        """
        A nice print for a ProteinTagHit.

        Returns:
            info_string (str): The string that is displayed when printing a ProteinTagHit

        Example:
            >>> tag_finder = TagFinder()
            >>> my_tags, my_termini = tag_finder.find_tags(sequence='GHHHHHHMYNAMEISCARL')
            >>> print(my_tags[0])
            Output:
            Name:      His-Tag
            Sequence:  GHHHHHH
            Location:  1-7
            Identity:  100.0
            Length:    7
            Origin:    Synthetic peptide
            Method:    Divalent Ion Chelate (Ni2+, Co2+, Cu2+, Zn2+)
            Reference: https://pubmed.ncbi.nlm.nih.gov/34520613/
        """
        info_string = (
            f"Name:      {self.name}\n"
            f"Sequence:  {self.sequence}\n"
            f"Location:  {self.start+1}-{self.end}\n"
            f"Identity:  {self.identity}\n"
            f"Length:    {self.length}\n"
            f"Type:      {self.type}\n"
            f"Origin:    {self.origin}\n"
            f"Method:    {self.method}\n"
            f"Reference: {self.reference}\n"
        )
        return info_string


class TagFinder:
    """
    This is the main class used for finding tags in sequences. When initialized this class builds an internal tag
    library to use when searching for tags in sequences.
    """

    def __init__(self, identity_cutoff: float = 90.0, margin_cutoff: int = 4):
        """
        Initialize the tag finder class and build the internal list of tags used to search for tags in sequences.

        Args:
            identity_cutoff (float): The minimum required identity for a tag to be considered a hit. This is also the
            minimum requires conservation for a sequence to be removed if a reference alignment is provided.
            [Default=90.0].
            margin_cutoff (float): The maximum number of amino-acids between tag and termini or between two tags
            [Default=4].
        """
        self._tags = []
        self._identity_cutoff = identity_cutoff
        self._margin_cutoff = margin_cutoff

        # Read in the tag database
        tag_file = f"{dirname(abspath(__file__))}/static/protein_tags.csv"
        if not isfile(tag_file):
            raise FileNotFoundError(f"Could not find tag database file: {tag_file}")
        tag_data = pd.read_csv(tag_file, delimiter=",", low_memory=True)
        for i, row in tag_data.iterrows():
            self._add_tag(
                name=row["name"],
                sequence=row["sequence"],
                method=row["method"],
                origin=row["origin"],
                reference=row["reference"],
            )

    @property
    def identity_cutoff(self) -> float:
        """getting identity_cutoff value"""
        return self._identity_cutoff

    @identity_cutoff.setter
    def identity_cutoff(self, value: float):
        """setting identity_cutoff value"""
        self._identity_cutoff = value

    @property
    def margin_cutoff(self) -> int:
        """getting margin_cutoff value"""
        return self._margin_cutoff

    @margin_cutoff.setter
    def margin_cutoff(self, value: int):
        """setting margin_cutoff value"""
        self._margin_cutoff = value

    def _add_tag(
        self, name: str, sequence: str, method: str, origin: str, reference: str
    ) -> ProteinTag:
        """
        Add a tag to the internal tag library.

        Args:
            name (str): The name of the tag.
            sequence (str): The amino-acid sequence of the tag.
            method (str): The method used for protein purification with the tag.
            origin (str): The origin of the tag.
            reference (str): The reference describing the tag.

        Returns:
            Nothing. Updates the internal tag library.
        """
        is_fusion = name.endswith("-Fusion")
        is_leader = name.endswith("-Leader")
        is_cleavage = name.endswith("-Cleavage")
        tag = ProteinTag(
            name=name,
            sequence=sequence,
            method=method,
            origin=origin,
            reference=reference,
            length=len(sequence),
            fusion=is_fusion,
            leader=is_leader,
            cleavage=is_cleavage,
        )
        self._tags.append(tag)
        return tag

    def _tag_search(self, tag: ProteinTag, query_seq: str) -> list[ProteinTagHit]:
        """
        Search for a single in a query sequence and return a list of ProteinTagHits.

        Args:
            tag (ProteinTag): The ProteinTag to search for.
            query_seq (str): The sequence to search for the tag in.

        Returns:
            hits (list[ProteinTagHit]): A list of ProteinTagHits for the query sequence.
        """
        query_len = len(query_seq)
        hits = []

        # Don't count protein fusions if the non-fusion part
        # is smaller than 25 amino acid residues.
        if tag.fusion and query_len - tag.length < 25:
            return []

        if tag.length < query_len:
            for i in range(query_len - tag.length + 1):
                #  TODO: This can be done faster at some point
                #  TODO: (doing the comparison for positions only once)
                #  TODO: Similar to offset estimation and
                #  TODO: Ingmars PosXobsKernel
                query_sub = query_seq[i : i + tag.length]
                identity = round(
                    [t == q for t, q in zip(tag.sequence, query_sub)].count(True)
                    / tag.length
                    * 100,
                    2,
                )
                if identity >= self.identity_cutoff:
                    start = i
                    end = i + tag.length
                    tag_aln = (
                        "-" * i + tag.sequence + "-" * (query_len - i - tag.length)
                    )
                    hit = ProteinTagHit(
                        tag=tag,
                        start=start,
                        end=end,
                        query_aln=query_seq,
                        tag_aln=tag_aln,
                        identity=identity,
                        location="Unknown",
                        conservation=0,
                    )
                    hits.append(hit)
        if len(hits) == 0:
            return []
        return hits

    @staticmethod
    def _tag_hit_overlap(hit1: ProteinTagHit, hit2: ProteinTagHit) -> bool:
        """
        Test if two tag hits are overlapping.

        Args:
            hit1 (ProteinTagHit): First ProteinTagHit.
            hit2 (ProteinTagHit): Second ProteinTagHit.

        Returns:
            overlap (bool): True if the two tags overlap otherwise False.
        """
        overlap12 = (
            hit2.start <= hit1.start <= hit2.end and hit2.start <= hit1.end <= hit2.end
        )
        overlap21 = (
            hit1.start <= hit2.start <= hit1.end and hit1.start <= hit2.end <= hit1.end
        )
        return overlap12 or overlap21

    def _get_non_redundant_tag_hits(self, sequence: str) -> list[ProteinTagHit]:
        """
        Get non-redundant tags in an input sequence.

        Args:
            sequence (str): The input sequence in which the tags are found.

        Returns:
            tag_hits (list[ProteinTagHit]): Output non-redundant list of ProteinTagHits.
        """
        # Find all hits and group by tag name or prefix
        all_hits = {}
        all_prefixes = {}
        for tag in self._tags:
            hits = self._tag_search(tag, sequence)
            for hit in hits:
                prefix = hit.name.split("-")[0]
                if hit.name in all_hits:
                    all_hits[hit.name] += [hit]
                elif prefix in all_prefixes:
                    all_hits[all_prefixes[prefix]] += [hit]
                else:
                    all_hits[hit.name] = [hit]
                    all_prefixes[prefix] = hit.name

        # Remove redundant hits that overlap
        non_redundant = []
        for name in all_hits:
            kept_hits = []
            for hit1 in sorted(all_hits[name], key=lambda x: x.length, reverse=True):
                overlap = False
                for hit2 in kept_hits:
                    if self._tag_hit_overlap(hit1, hit2):
                        overlap = True
                        break
                if not overlap:
                    kept_hits.append(hit1)
            non_redundant += kept_hits

        return non_redundant

    def _get_termini(
        self, tag_hits: list[ProteinTagHit], sequence: str
    ) -> dict[str, tuple[int, int]]:
        """
        Find N-terminal and C-terminal regions containing tags.

        Args:
            tag_hits (list[ProteinTagHit]): Input list of ProteinTagHits.
            sequence (str): Input sequence in which the tags are found.

        Returns:
            termini (dict[str, tuple[int, int]]): Output dictionary with N-terminal and C-terminal regions with tags.
        """

        # Check if within the limit of a given distance
        def within_limit(input_hit, distance):
            if input_hit.length <= self.margin_cutoff:
                limit = input_hit.length - 1
            else:
                limit = self.margin_cutoff
            return distance <= limit

        # Find the maximum n-terminus cutoff and minimum c-terminus cutoff
        max_n_terminus = 0
        for h, hit in enumerate(sorted(tag_hits, key=lambda x: x.start)):
            if not within_limit(hit, abs(hit.start - max_n_terminus)):
                break
            max_n_terminus = hit.end

        min_c_terminus = len(sequence)
        for h, hit in enumerate(sorted(tag_hits, key=lambda x: x.end, reverse=True)):
            if not within_limit(hit, abs(hit.end - min_c_terminus)):
                break
            min_c_terminus = hit.start

        # Adjust the maximum n terminus by extending it if a methionine is found
        # close to where the tag ends, as this methionine is likely to be the
        # biological start of the sequence.
        for pos in range(max_n_terminus, max_n_terminus + self.margin_cutoff):
            if sequence[pos] == "M":
                max_n_terminus = pos
                break

        termini = {
            "N-terminus": (0, max_n_terminus),
            "C-terminus": (min_c_terminus, len(sequence)),
        }
        return termini

    def _assign_locations(
        self, tag_hits: list[ProteinTagHit], termini: dict[str, tuple[int, int]]
    ) -> list[ProteinTagHit]:
        """
        Assigns tags a location flag specifying if they are in the N-terminus, C-terminus or internal.

        Args:
            tag_hits (list[ProteinTagHit]): Input list of ProteinTagHits.
            termini (dict[str, tuple[int, int]]): Output dictionary with N-terminal and C-terminal regions with tags.

        Returns:
            tag_hits (list[ProteinTagHit]): Output list of ProteinTagHits with updated location attributes.
        """

        # Separate hits into n-terminal, c-terminal and internal hits
        max_n_terminus = termini["N-terminus"][-1]
        min_c_terminus = termini["C-terminus"][0]

        # Find N-terminal hits
        n_terminal_hits = []
        for hit in sorted(tag_hits, key=lambda x: x.start):
            if hit.end <= max_n_terminus:
                # If the tag is small, it has to be close to the terminus to be found
                if hit.length <= self.margin_cutoff and hit.start > hit.length - 1:
                    continue
                hit.location = "N-terminus"
                n_terminal_hits.append(hit)

        # Find C-terminal hits
        c_terminal_hits = []
        for hit in sorted(tag_hits, key=lambda x: x.start):
            # Leader sequences can only be in the N-terminus
            if hit.leader:
                continue
            if hit.start >= min_c_terminus:
                # If the tag is small, it has to be close to the terminus to be found
                if (
                    hit.length <= self.margin_cutoff
                    and abs(hit.end - len(hit.query_aln)) > hit.length - 1
                ):
                    continue
                hit.location = "C-terminus"
                c_terminal_hits.append(hit)

        # Find internal hits
        internal_hits = []
        for hit in sorted(tag_hits, key=lambda x: x.start):
            # Leader sequences, cleavage sites, and protein fusions can not be internal
            if hit.leader or hit.fusion or hit.cleavage:
                continue
            if hit not in n_terminal_hits and hit not in c_terminal_hits:
                hit.location = "Internal"
                internal_hits.append(hit)

        combined_hits = n_terminal_hits + c_terminal_hits + internal_hits
        assigned_hits = sorted(combined_hits, key=lambda x: x.start)
        return assigned_hits

    @staticmethod
    def _assign_conservation(
        tag_hits: list[ProteinTagHit], query_aln: str, reference_aln: str
    ) -> list[ProteinTagHit]:
        """
        Assigns conservation to tags based on a pairwise alignment between the query sequence and a biological reference
        sequence. Calculate the conservation of the tags based on the alignment. Conserved tags (high identity) are
        likely to be false positives, since the sequence is conserved in the reference.

        Args:
            tag_hits (list[ProteinTagHit]): Input list of ProteinTagHits.
            query_aln (str): Input alignment of the query sequence.
            reference_aln (str): Input alignment of the reference sequence.

        Returns:
            tag_hits (list[ProteinTagHit]): Output list of ProteinTagHits with updated conservation.

        Raises:
            ValueError: If the input alignments do not have equal length.
        """
        if query_aln is None or reference_aln is None:
            return tag_hits

        if len(query_aln) != len(reference_aln):
            raise ValueError(
                "Alignments query_aln and reference_aln must have equal length."
            )

        identity_mask = []
        for que, ref in zip(query_aln, reference_aln):
            if que == "-":
                continue
            elif ref == "-" or que != ref:
                identity_mask.append(0)
            else:
                identity_mask.append(1)

        for tag in tag_hits:
            reference_identity = identity_mask[tag.start : tag.end]
            tag.conservation = round(sum(reference_identity) / tag.length * 100, 2)

        return tag_hits

    def find_tags(
        self,
        sequence: str,
        query_aln: str = None,
        reference_aln: str = None,
    ) -> tuple[list[ProteinTagHit], dict[str, tuple[int, int]]]:
        """
        Find tags in an input sequence and return tags and termini.

        Args:
            sequence (str): Input sequence to find tags in.
            query_aln (str): (Optional) Input query alignment [Default=None].
            reference_aln (str): (Optional) Input reference alignment [Default=None].

        Returns:
            tags (tuple[list[ProteinTagHit]]): A list of ProteinTagHits.
            termini (dict[str, tuple[int, int]]): Output dictionary with N-terminal and C-terminal regions with tags.
        Raises:
            ValueError: If the query_aln and reference_aln do not have equal length or if the alignments are empty or if
            the query does not match the query sequence.

        Example:
            >>> tag_finder = TagFinder()
            >>> my_tags, my_termini = tag_finder.find_tags(sequence='GHHHHHHMYNAMEISCARL')
            >>> print(my_tags[0].sequence)
            >>> 'GHHHHHH'
            >>> print(my_termini)
            >>> '{"N-terminus": (0, 7), "C-terminus": (18, 18)}'
        """
        tag_hits = self._get_non_redundant_tag_hits(sequence)

        # Validate tags if a query and reference alignment is provided
        if query_aln is None or reference_aln is None:
            query_aln = None
            reference_aln = None
        else:
            if len(query_aln) == 0:
                raise ValueError("Error: The query alignment is of length 0")

            if len(reference_aln) == 0:
                raise ValueError("Error: The reference alignment is of length 0")

            if len(query_aln) != len(reference_aln):
                raise ValueError(
                    "Error: The query and reference alignments must have equal length"
                )

            if query_aln.replace("-", "") != sequence:
                raise ValueError(
                    "Error: The query alignment does not match the input sequence"
                )

        valid_hits = self._assign_conservation(
            tag_hits=tag_hits, query_aln=query_aln, reference_aln=reference_aln
        )
        # Keep tags with low conservation, as these are not found in the reference sequence
        tag_hits = [
            tag for tag in valid_hits if tag.conservation <= self.identity_cutoff
        ]
        termini = self._get_termini(tag_hits, sequence)
        all_tags = self._assign_locations(tag_hits, termini)

        return all_tags, termini

    def strip_tags(
        self,
        sequence: str,
    ) -> tuple[str, list[ProteinTagHit]]:
        """
        Stips tags from the N- and C-terminus of a sequence and reports the tags stripped away.

        Args:
            sequence (str): Input sequence to strip tags from.

        Returns:
            sequence (str): The output sequence without tags.
            remove_tags (list[ProteinTagHit]): The output list of tags that were removed from the input sequence.

        Raises:
            TypeError: If the input file is not a .csv file

        Example:
            >>> tag_finder = TagFinder()
            >>> no_tag_seq, tag_list = tag_finder.strip_tags(sequence='GHHHHHHMYNAMEISCARL')
            >>> print(no_tag_seq)
            >>> 'MYNAMEISCARL'
            >>> print(tag_list[0].sequence)
            >>> 'GHHHHHH'
        """
        tags, termini = self.find_tags(sequence=sequence)
        stripped_seq = sequence[termini["N-terminus"][-1] : termini["C-terminus"][0]]
        removed_tags = [tag for tag in tags if tag.location != "Internal"]
        return stripped_seq, removed_tags

    @staticmethod
    def _tag_report(removed_tag_list: list[ProteinTagHit], terminus: str) -> str:
        """
        Create a report string from a list of ProteinTagHits and a terminus.

        Args:
            removed_tag_list (list[ProteinTagHit]): The input list of ProteinTagHits.
            terminus (str): The input terminus, must be either 'N' or 'C'.

        Returns:
            report (str): The output report string of the hits.
        """
        terminal_tags = []
        for t in sorted(removed_tag_list, key=lambda x: x.start):
            if t.location == f"{terminus}-terminus":
                description = f"[{t.name}:{t.start}-{t.sequence}-{t.end}]"
                terminal_tags.append(description)
        if len(terminal_tags) == 0:
            report_string = "None"
        else:
            report_string = " ".join(terminal_tags)
        return report_string

    def get_strip_report(self, sequence: str) -> dict:
        """
        Strip tags from a sequence and return a report dictionary of the stripped tags.

        Args:
            sequence (str): Input sequence to strip tags from.

        Returns:
            output (dict): A dictionary of the following form:
            output = {'tags_found': (bool), # True if tags are found in the sequence
                      'names': ([str]), # List of removed tag names
                      'locations': ([str]), # List of removed tag locations
                      'summary': (str),  # Summary of the removed tags
                      'sequence': (str),  # Sequence without tags}

        Example:
            >>> my_sequence = 'MAHHHHHHMSFFRMKRRLNFVVKRGIEELWENSFLDNNVDMKKIEYSKTG'
            >>>               'DAWPCVLLRKKSFEDLHKLYYICLKEKNKLLGEQYFHLQNSTKMLQHGRL'
            >>>               'KKVKLTMKRILTVLSRRAIHDQCLRAKDMLKKQEEREFYEIQKFKLNEQL'
            >>>               'LCLKHKMNILKKYNSFSLEQISLTFSIKKIENKIQQIDIILNPLRKETMY'
            >>>               'LLIPHFKYQRKYSDLPGFISWKKQNIIALRNNMSKLHRLY'
            >>> tag_finder = TagFinder()
            >>> report = tag_finder.get_strip_report(sequence=my_sequence)
            >>> print(report)
            Output:
            {'tags_found': True,
             'names': ['PolyHis-Tag'],
             'locations': ['N-terminus'],
             'summary': 'N-term: [PolyHis-Tag:2-HHHHHH-8] C-term: None',
             'sequence': 'MSFFRMKRRLNFVVKRGIEELWENSFLDNNVDMKKIEYSKTGDAWPCVLL
                          RKKSFEDLHKLYYICLKEKNKLLGEQYFHLQNSTKMLQHGRLKKVKLTMK
                          RILTVLSRRAIHDQCLRAKDMLKKQEEREFYEIQKFKLNEQLLCLKHKMN
                          ILKKYNSFSLEQISLTFSIKKIENKIQQIDIILNPLRKETMYLLIPHFKY
                          QRKYSDLPGFISWKKQNIIALRNNMSKLHRLY'
            }
        """
        found_tag = True
        removed_names = []
        removed_locations = []
        stripped_sequence, removed_tags = self.strip_tags(sequence)

        # Create a tag report for both termini
        if len(removed_tags) != 0:
            for tag in removed_tags:
                removed_names.append(tag.name)
                removed_locations.append(tag.location)
            n_terminal_tags = self._tag_report(removed_tags, terminus="N")
            c_terminal_tags = self._tag_report(removed_tags, terminus="C")
            removal_report = f"N-term: {n_terminal_tags} C-term: {c_terminal_tags}"
        else:
            found_tag = False
            removal_report = f"N-term: None C-term: None"

        output = {
            "tags_found": found_tag,
            "names": removed_names,
            "locations": removed_locations,
            "summary": removal_report,
            "sequence": stripped_sequence,
        }

        return output

    def tag_strip_csv(self, input_file: str, output_file: str, col_name: str) -> str:
        """
        Strip terminal purification tags from an input .csv file containing protein sequences.

        Args:
            input_file (str): Full path of the input CSV file.
            output_file (str): Full path of the output CSV file.
            col_name (str): The column name containing the sequence.

        Returns:
            output_file (str): An output file where the sequences in the column specified by 'col_name' have their
            terminal tags stripped off. A column called 'removed_tags' is added to the output file with information
            about the removed tags.

        Raises:
            TypeError: If the input file is not a .csv file.

        Example:
            >>> tag_finder = TagFinder()
            >>> tag_finder.tag_strip_csv.(input_file='/path/to/my_input.csv',
            >>>                           output_file='/path/to/my_output.csv',
            >>>                           col_name='aa_seq')
        """
        if not input_file.endswith(".csv"):
            raise TypeError("Input File is not a .csv file!")

        dataframe = pd.read_csv(input_file, delimiter=",", low_memory=True)
        dataframe["removed_tags"] = dataframe.apply(lambda _: "", axis=1)
        tag_counter = 0
        seq_counter = 0
        tag_names = []
        tag_locations = []
        for i, row in tqdm(
            dataframe.iterrows(), total=dataframe.index.size, desc="Processing"
        ):
            report = self.get_strip_report(sequence=row[col_name])
            tag_names += report["names"]
            tag_locations += report["locations"]
            dataframe.at[i, col_name] = report["sequence"]
            dataframe.at[i, "removed_tags"] = report["summary"]
            if report["tag_found"]:
                tag_counter += 1
            seq_counter += 1

        # Generate basic report
        tag_percent = int(round(tag_counter / seq_counter * 100))
        tag_total = len(tag_locations)
        n_term_counter = tag_locations.count("N-terminus")
        n_term_percent = int(round(n_term_counter / tag_total * 100))
        c_term_counter = tag_locations.count("C-terminus")
        c_term_percent = int(round(c_term_counter / tag_total * 100))
        report = f"Found and removed tags in {tag_counter}/{seq_counter} sequences ({tag_percent}%)\n"
        report += "Tag locations:\n"
        report += f"N-terminus: {n_term_counter}/{tag_total} ({n_term_percent}%)\n"
        report += f"C-terminus: {c_term_counter}/{tag_total} ({c_term_percent}%)\n"
        report += "Tag types:\n"
        report += "\n".join(
            str(pd.DataFrame(tag_names)[0].value_counts()).split("\n")[1:-1]
        )
        print(report)
        dataframe.to_csv(output_file, index=False)
        return output_file

    def clean_dataset(
        self,
        dataset: Dataset,
        seq_col_name: str,
        num_proc: int = 1,
        convert_nucleotide: bool = False,
        remove_artificial: bool = False,
    ) -> Dataset:
        """
        Clean an input Huggingface Dataset by converting to protein sequences and removing tags. This will add a column
        with a summary of removed tags to the dataset. This function can run using multiple processors by specifying
        the num_procs flags, however if using multiple processors it will not present a summary of the removed flags.

        Args:
            dataset (Dataset): Input HuggingFace dataset.
            seq_col_name (str): Input column name containing the sequences.
            num_proc (int): Number of processors used for finding tags [Default=1].
            convert_nucleotide (bool): If True, convert DNA and RNA sequences to protein sequences before removing tags.
            remove_artificial (bool): If True, remove artificial low entropy sequences with a Jensen-Shannon distance
            over 0.6 to naturally occuring proteins from the data-set if they do not start with methionine.
        Returns:
            dataset: Modified HuggingFace dataset. The seq_col_name column has been replaced with a sequence where
            N-terminal and C-terminal tags are removed. The dataset also has a new column called 'removed_tags' where a
            summary of the removed N-terminal and C-terminal tags is shown. If convert_nucleotide is True, DNA and RNA
            sequences have been converted to protein. If remove_artificial is True, artificial low entropy sequences
            with a Jensen-Shannon distance over 0.6 to naturally occuring proteins have been removed.

        Example:
            >>> tag_finder = TagFinder()
            >>> raw_dataset = load_full_dataset("Company/ProteineaSolubility")
            >>> no_tags_dataset = tag_finder.clean_dataset(dataset=raw_dataset,
            >>>                                            seq_col_name="aa_seq")
        """
        dataset_size = len(dataset)
        tag_column = ["N-term: None C-term: None"] * dataset_size
        artificial_column = [False] * dataset_size
        dataset = dataset.add_column("removed_tags", tag_column)
        dataset = dataset.add_column("artificial_sequence", artificial_column)
        cleaner = SequenceCleaner(
            seq_col_name=seq_col_name,
            tag_col_name="removed_tags",
            artificial_col_name="artificial_sequence",
            tag_finder=self,
            convert_nucleotide=convert_nucleotide,
            label_artificial=remove_artificial,
        )
        dataset = dataset.map(cleaner, num_proc=num_proc)
        if num_proc == 1:
            print(cleaner)
        else:
            no_tags = dataset["removed_tags"].count("N-term: None C-term: None")
            tag_num = dataset_size - no_tags
            tag_percent = int(round(tag_num / dataset_size * 100))
            report = f"Found and removed tags in {tag_num}/{dataset_size} sequences ({tag_percent}%)\n"
            print(report)
            print("No detailed Cleaning summary when using multiple processors.")

        if remove_artificial:
            dataset = dataset.filter(
                lambda row: row["artificial_sequence"] is False, num_proc=num_proc
            )
        dataset = dataset.remove_columns("artificial_sequence")

        return dataset


class SequenceCleaner:
    """
    This class is used to interface with huggingface datasets. It is a callable class is used for the huggingface .map()
    method to clean sequences by detecting and converting DNA and RNA sequences to protein sequences and use TagFinder
    on the sequences converted sequences to remove tags.

    This class operates on individual samples (rows) and returns new samples where the sequence column has been replaced
    with a cleaned up sequence without tags and where the removed_tags column has a summary of the tags that were
    removed from the input sequence.
    """

    def __init__(
        self,
        seq_col_name: str,
        tag_col_name: str,
        artificial_col_name: str,
        tag_finder: TagFinder,
        convert_nucleotide: bool = False,
        label_artificial: bool = False,
    ):
        """
        A Class that can be mapped across a HuggingFace dataset to clean and convert sequences and find and remove
        terminal protein purification tags, fusion proteins, cleavage sites, and leader sequences.

        Args:
            seq_col_name (str): The column name of the column containing the query sequence in which to remove tags.
            tag_col_name (str): The column name to store the summary of identified terminal tags that were removed.
            tag_finder (TagFinder): A TagFinder instance used for tag identification.
            convert_nucleotide (bool): If True, convert DNA and RNA sequences to protein before removing tags.
            label_artificial (bool): If True, label low complexity sequences that do not start with M.
        """
        self._tag_finder = tag_finder
        self._seq_col_name = seq_col_name
        self._tag_col_name = tag_col_name
        self._artificial_col_name = artificial_col_name
        self._ambiguous_counter = 0
        self._protein_counter = 0
        self._dna_counter = 0
        self._rna_counter = 0
        self._tag_counter = 0
        self._tag_names = []
        self._tag_locations = []
        self._convert_nucleotide = convert_nucleotide
        self._label_artificial = label_artificial

    @staticmethod
    def get_sequence_type(sequence: str) -> str:
        """
        Calculate the sequence type (DNA, RNA, Protein, Ambiguous Protein) of an input sequence.

        Args:
            sequence (str): The input sequence.

        Returns:
            seq_type (str): The type of the input sequence.
        """
        alphabet = set(sequence)
        dna_fraction = sum([sequence.count(x) for x in "ATGC"]) / len(sequence)
        rna_fraction = sum([sequence.count(x) for x in "AUGC"]) / len(sequence)
        if (
            dna_fraction > 0.99
            and len(sequence) % 3 == 0
            and all(x in "ATGCWSMKRYBDHVN.X" for x in alphabet)
        ):
            seq_type = "DNA"
        elif (
            rna_fraction > 0.99
            and len(sequence) % 3 == 0
            and all(x in "AUGCWSMKRYBDHVN.X" for x in alphabet)
        ):
            seq_type = "RNA"
        elif all(x in "QWERTYIPASDFGHKLCVNM" for x in alphabet):
            seq_type = "Protein"
        elif all(x in "QWERTYIPASDFGHKLCVNMX" for x in alphabet):
            seq_type = "Ambiguous_Protein"
        else:
            seq_type = "Unknown"
        return seq_type

    @staticmethod
    def js_distance(sequence: str) -> float:
        """
        Calculates the Jensen-Shannon distance of a sequence based on comparison to natural amino-acid frequencies.

        Args:
            sequence (str): The input sequence.

        Returns:
            entropy (float): Shannon Entropy of the input sequence.
        """

        # Amino-Acid frequencies based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7127678/
        aa_frequencies = {
            "A": 0.0777,
            "C": 0.0157,
            "D": 0.0530,
            "E": 0.0656,
            "F": 0.0405,
            "G": 0.0691,
            "H": 0.0227,
            "I": 0.0591,
            "K": 0.0595,
            "L": 0.0960,
            "M": 0.0238,
            "N": 0.0427,
            "P": 0.0469,
            "Q": 0.0393,
            "R": 0.0526,
            "S": 0.0694,
            "T": 0.0550,
            "V": 0.0667,
            "W": 0.0118,
            "Y": 0.0311,
        }
        aa_list = list(aa_frequencies.keys())
        sequence = sequence.upper().replace("X", "")
        observed_frequencies = np.array(
            [sequence.count(aa) / len(sequence) for aa in aa_list]
        )
        expected_frequencies = np.array([aa_frequencies[aa] for aa in aa_list]) / sum(
            aa_frequencies.values()
        )
        # Add a small number to avoid division by zero
        observed_frequencies += 1e-16
        observed_frequencies = observed_frequencies / sum(observed_frequencies)

        # Calculate JS-distance
        js_distance = sp.spatial.distance.jensenshannon(
            observed_frequencies, expected_frequencies
        )

        return js_distance

    def convert_sequence(self, sequence: str) -> str:
        """
        Cleans up an input sequence by checking the sequence type and converting to a protein sequence if possible.

        Args:
            sequence (str): Input sequence which can be DNA, RNA or protein.

        Returns:
            sequence (str): Output sequence converted to protein.
        """
        seq_type = self.get_sequence_type(sequence)
        if seq_type == "RNA":
            self._rna_counter += 1
        elif seq_type == "DNA":
            self._dna_counter += 1
        elif seq_type == "Protein":
            self._protein_counter += 1
        elif seq_type == "Ambiguous_Protein":
            self._protein_counter += 1
            self._ambiguous_counter += 1
        else:
            seq_type = "Unknown"

        if seq_type in ["DNA", "RNA"]:
            translated = str(Seq.Seq(sequence).translate().replace("*", ""))
            seq_type = self.get_sequence_type(translated)
            sequence = translated

            if seq_type == "Protein":
                self._protein_counter += 1
            elif seq_type == "Ambiguous_Protein":
                self._protein_counter += 1
                self._ambiguous_counter += 1
            else:
                raise TypeError(f"Unknown sequence: {sequence}")

        return sequence

    def __call__(self, sample):
        """
        The function called by the HuggingFace .map() function.

        Args:
            sample: The input row in the HuggingFace dataset.

        Returns:
            sample: The modified row in the HuggingFace dataset.
        """
        if self._convert_nucleotide:
            sequence = sample[self._seq_col_name]
            converted_sequence = self.convert_sequence(sequence)
            if sequence != converted_sequence:
                print("Converting Nucleotide Sequence:")
                print(sequence)
                print()

        else:
            converted_sequence = sample[self._seq_col_name]

        # Calculate a probability that the sequence is natural based on AA frequency
        if self._label_artificial:
            js_dist = self.js_distance(converted_sequence)
            artificial_candidate = js_dist > 0.6 and converted_sequence[0] != "M"
            if artificial_candidate:
                sample[self._artificial_col_name] = True
                print("Removing artificial sequence:")
                print(converted_sequence)
                print()
        report = self._tag_finder.get_strip_report(sequence=converted_sequence)
        self._tag_names += report["names"]
        self._tag_locations += report["locations"]
        sample[self._seq_col_name] = report["sequence"]
        sample[self._tag_col_name] = report["summary"]
        if report["tags_found"]:
            self._tag_counter += 1
        return sample

    def __repr__(self):
        """
        Print a summary of the tag removal across the dataset.

        Returns:
            map_report (str): A string report of the TagFinder results of the dataset.
        """
        nucleic_acid_count = self._dna_counter + self._rna_counter
        convert_percent = int(round(nucleic_acid_count / self._protein_counter * 100))
        ambiguous_percent = int(
            round(self._ambiguous_counter / self._protein_counter * 100)
        )
        tag_percent = int(round(self._tag_counter / self._protein_counter * 100))
        tag_total = len(self._tag_locations)
        n_term_counter = self._tag_locations.count("N-terminus")
        n_term_percent = int(round(n_term_counter / tag_total * 100))
        c_term_counter = self._tag_locations.count("C-terminus")
        c_term_percent = int(round(c_term_counter / tag_total * 100))
        map_report = (
            f"Converted {nucleic_acid_count}/{self._protein_counter} "
            f"sequences from DNA/RNA to protein ({convert_percent}%)\n"
        )
        map_report += (
            f"Found {self._ambiguous_counter}/{self._protein_counter} "
            f"proteins sequences with ambiguous residues ({ambiguous_percent}%)\n"
        )
        map_report += (
            f"Removed tags in {self._tag_counter}/{self._protein_counter} "
            f"protein sequences ({tag_percent}%)\n"
        )
        map_report += "Tag locations:\n"
        map_report += f"N-terminus: {n_term_counter}/{tag_total} ({n_term_percent}%)\n"
        map_report += f"C-terminus: {c_term_counter}/{tag_total} ({c_term_percent}%)\n"
        map_report += "Tag types:\n"
        map_report += "\n".join(
            str(pd.DataFrame(self._tag_names)[0].value_counts()).split("\n")[1:-1]
        )
        return map_report


if __name__ == "__main__":
    tag_finder = TagFinder()
    tf = TagFinder()
    # ds = load_full_dataset('Company/TemStaProLabelled')
    # ds = load_full_dataset('Company/fpbase_original')
    ds = load_full_dataset("Company/ProteineaSolubility")
    ds = tf.clean_dataset(
        dataset=ds,
        seq_col_name="aa_seq",
        num_proc=10,
        convert_nucleotide=True,
        remove_artificial=True,
    )
