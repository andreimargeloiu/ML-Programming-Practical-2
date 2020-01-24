import os
from glob import iglob
from typing import List, Dict, Any, Iterable, Optional, Iterator

import numpy as np
from more_itertools import chunked
from dpu_utils.mlutils.vocabulary import Vocabulary

from graph_pb2 import Graph
from graph_pb2 import FeatureNode
from collections import Counter

DATA_FILE_EXTENSION = "proto"
START_SYMBOL = "%START%"
END_SYMBOL = "%END%"


def get_data_files_from_directory(data_dir: str, max_num_files: Optional[int] = None) -> List[str]:
    files = iglob(
        os.path.join(data_dir, "**/*.%s" % DATA_FILE_EXTENSION), recursive=True
    )
    if max_num_files:
        files = sorted(files)[: int(max_num_files)]
    else:
        files = list(files)
    return files


def method_split_tokens(g):
    """
    Output the tokens of all methods in a file.
    For each node "METHOD", print the tokens between node.startPosition and node.endPosition

    Complexity: no_nodes + no_tokens_to_print
    """
    tokens = list(filter(lambda n: n.type in (
        FeatureNode.TOKEN, FeatureNode.IDENTIFIER_TOKEN), g.node))
    poz = 0  # keep track of the starting position of the token of the previous method

    list_of_samples = []
    for node in g.node:
        if node.contents == "METHOD":
            sample = []
            # advance with the position in the tokens list
            while poz < len(tokens) and tokens[poz].startPosition < node.startPosition:
                poz += 1

            # print all tokens inside this method
            aux = poz
            while aux < len(tokens) and tokens[aux].endPosition <= node.endPosition:
                sample.append(tokens[aux].contents.lower())
                aux += 1

            list_of_samples.append(sample)

    return list_of_samples


def load_data_file(file_path: str) -> Iterable[List[str]]:
    """
    Load a single data file, returning token streams.

    Args:
        file_path: The path to a data file.

    Returns:
        Iterable of lists of strings, each a list of tokens observed in the data.
    """
    with open(file_path, "rb") as f:
        g = Graph()
        g.ParseFromString(f.read())

        return method_split_tokens(g)


def build_vocab_from_data_dir(
    data_dir: str, vocab_size: int, max_num_files: Optional[int] = None
) -> Vocabulary:
    """
    Compute model metadata such as a vocabulary.

    Args:
        data_dir: Directory containing data files.
        vocab_size: Maximal size of the vocabulary to create.
        max_num_files: Maximal number of files to load.
    """

    data_files = get_data_files_from_directory(data_dir, max_num_files)

    vocab = Vocabulary(add_unk=True, add_pad=True)
    # Make sure to include the START_SYMBOL in the vocabulary as well:
    vocab.add_or_get_id(START_SYMBOL)
    vocab.add_or_get_id(END_SYMBOL)

    #TODO 3# Insert your vocabulary-building code here
    counter = Counter()
    for file in data_files:
        list_of_samples = load_data_file(file)

        for list_tokens in list_of_samples:
            for token in list_tokens:
                counter[token] += 1

    # most common tokens in vocabulary
    for elem, cnt in counter.most_common(vocab_size - 2):
        vocab.add_or_get_id (elem)

    return vocab


def tensorise_token_sequence(
    vocab: Vocabulary, length: int, token_seq: Iterable[str],
) -> List[int]:
    """
    Tensorise a single example.

    Args:
        vocab: Vocabulary to use for mapping tokens to integer IDs
        length: Length to truncate/pad sequences to.
        token_seq: Sequence of tokens to tensorise.

    Returns:
        List with length elements that are integer IDs of tokens in our vocab.
    """
    #TODO 4# Insert your tensorisation code here
    tokens_ids = []
    tokens_ids.append(vocab.get_id_or_unk(START_SYMBOL))
    tokens_ids.extend(vocab.get_id_or_unk_multiple(token_seq, pad_to_size=length-2))
    tokens_ids.append(vocab.get_id_or_unk(END_SYMBOL))

    return tokens_ids


def load_data_from_dir(
    vocab: Vocabulary, length: int, data_dir: str, max_num_files: Optional[int] = None
) -> np.ndarray:
    """
    Load and tensorise data.

    Args:
        vocab: Vocabulary to use for mapping tokens to integer IDs
        length: Length to truncate/pad sequences to.
        data_dir: Directory from which to load the data.
        max_num_files: Number of files to load at most.

    Returns:
        numpy int32 array of shape [None, length], containing the tensorised
        data.
    """
    data_files = get_data_files_from_directory(data_dir, max_num_files)
    data = np.array(
        list(
            tensorise_token_sequence(vocab, length, token_seq)
            for data_file in data_files
            for token_seq in load_data_file(data_file)
        ),
        dtype=np.int32,
    )
    return data


def get_minibatch_iterator(
    token_seqs: np.ndarray,
    batch_size: int,
    is_training: bool,
    drop_remainder: bool = True,
) -> Iterator[np.ndarray]:
    indices = np.arange(token_seqs.shape[0])
    if is_training:
        np.random.shuffle(indices)

    for minibatch_indices in chunked(indices, batch_size):
        if len(minibatch_indices) < batch_size and drop_remainder:
            break  # Drop last, smaller batch

        minibatch_seqs = token_seqs[minibatch_indices]
        yield minibatch_seqs
