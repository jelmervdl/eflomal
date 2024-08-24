"""eflomal package"""
import sys
import os
import math
import subprocess
import logging
from collections import Counter
from operator import itemgetter
from tempfile import NamedTemporaryFile
from typing import NamedTuple, Optional, List, Tuple, Iterable

from .cython import read_text, write_text


logger = logging.getLogger(__name__)

Priors = NamedTuple('Priors', [
    ('priors', list),  # list of (srcword, trgword, alpha)
    ('hmmf', dict),    # dict of jump: alpha
    ('hmmr', dict),    # dict of jump: alpha
    ('ferf', list),    # list of (wordform, alpha)
    ('ferr', list),    # list of (wordform, alpha)
])

SentencePair = Tuple[List[str], List[str]]

class Aligner:
    """Aligner class"""

    priors: Optional[Priors]

    def __init__(self, *, model=3, score_model=0,
                 n_iterations=None, n_samplers=3,
                 rel_iterations=1.0, null_prior=0.2,
                 source_prefix_len=0, source_suffix_len=0,
                 target_prefix_len=0, target_suffix_len=0,
                 priors_file:Optional[Iterable[str]]=None):
        self.model = model
        self.score_model = score_model
        self.n_iterations = n_iterations
        self.n_samplers = n_samplers
        self.rel_iterations = rel_iterations
        self.null_prior = null_prior
        self.source_prefix_len = source_prefix_len
        self.source_suffix_len = source_suffix_len
        self.target_prefix_len = target_prefix_len
        self.target_suffix_len = target_suffix_len
        self.priors = read_priors(priors_file) if priors_file else None

    def prepare_files(self, input:List[SentencePair], src_output_file, trg_output_file, priors_output_file):
        """Convert text files to formats used by eflomal

        Inputs should be file objects or any iterables over lines. Outputs
        should be file objects.

        """
        src_index, n_src_sents, src_voc_size = to_eflomal_text_file(
            [pair[0] for pair in input], src_output_file,
            self.source_prefix_len, self.source_suffix_len)
        trg_index, n_trg_sents, trg_voc_size = to_eflomal_text_file(
            [pair[1] for pair in input], trg_output_file,
            self.target_prefix_len, self.target_suffix_len)
        logger.info('Prepared %d sentences for alignment', n_src_sents)
        if self.priors:
            to_eflomal_priors_file(
                self.priors, src_index, trg_index, priors_output_file)

    def align(self, batch:List[SentencePair], *, quiet=True, reverse=False, use_gdb=False) -> List[List[Tuple[int,int]]]:
        """Run alignment for the input"""
        pid = os.getpid()

        # executable = os.path.join(os.path.dirname(__file__), 'bin', 'eflomal')
        executable = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'eflomal')

        n_sentences = len(batch)

        n_iterations = self.n_iterations

        if n_iterations is None:
            iters = max(2, int(round(self.rel_iterations*5000 / math.sqrt(n_sentences))))
            iters4 = max(1, iters//4)
            if self.model == 1:
                n_iterations = (iters, 0, 0)
            elif self.model == 2:
                n_iterations = (max(2, iters4), iters, 0)
            else:
                n_iterations = (max(2, iters4), iters4, iters)

        with NamedTemporaryFile(delete=True, dir='.', prefix=f'{pid}_', suffix='.src', mode='wb') as srcf, \
             NamedTemporaryFile(delete=True, dir='.', prefix=f'{pid}_', suffix='.trg', mode='wb') as trgf, \
             NamedTemporaryFile(delete=True, dir='.', prefix=f'{pid}_', suffix='.priors', mode='w', encoding='utf-8') as priorsf:
            
            # Write input files for the eflomal binary
            self.prepare_files(batch, srcf, trgf, priorsf)
            
            # Run the eflomal binary
            args = [
                executable,
                '-m', str(self.model),
                '-s', srcf.name,
                '-t', trgf.name,
                '-n', str(self.n_samplers),
                '-N', str(self.null_prior),
                '-1', str(n_iterations[0]),
                ('-r' if reverse else '-f'), '-', # forward or reverse links to stdout
            ]
            
            if quiet: args.append('-q')
            if self.model >= 2: args.extend(['-2', str(n_iterations[1])])
            if self.model >= 3: args.extend(['-3', str(n_iterations[2])])
            if self.priors: args.extend(['-p', priorsf.name])
            if use_gdb: args = ['gdb', '-ex=run', '--args'] + args
        
            print(args, file=sys.stderr)
            output = subprocess.check_output(args)

        return [
            [parse_link(link) for link in line.split(b' ')] if line != b'' else []
            for line in output[:-1].split(b'\n')
        ]


class TextIndex:
    """Word to index mapping with lowercasing and prefix/suffix removal"""

    def __init__(self, index, prefix_len=0, suffix_len=0):
        self.index = index
        self.prefix_len = prefix_len
        self.suffix_len = suffix_len

    def __len__(self):
        return len(self.index)

    def __getitem__(self, word):
        word = word.lower()
        if self.prefix_len != 0:
            word = word[:self.prefix_len]
        if self.suffix_len != 0:
            word = word[-self.suffix_len:]
        e = self.index.get(word)
        if e is not None:
            e = e + 1
        return e


def parse_link(tupstr:bytes) -> Tuple[int,int]:
    try:
        i, j = tupstr.split(b'-')
        return int(i), int(j)
    except:
        raise ValueError(f'Cannot parse "{tupstr.decode()}" as link pair')


class Dictionary(dict):
    def __missing__(self, key):
        index = len(self)
        self[key] = index
        return index


def to_eflomal_text_file(sentences, outfile, prefix_len=0, suffix_len=0):
    """Write sentences to a file read by eflomal binary

    Arguments:

    sentencefile - input text list
    outfile - output file object
    prefix_len - prefix length to remove
    suffix_len - suffix length to remove

    Returns TextIndex object.

    """
    def normalize(token):
        token = token.lower()
        if prefix_len > 0: token = token[:prefix_len]
        if suffix_len > 0: token = token[-suffix_len:]
        return token

    index = Dictionary()
    sents = [
        [index[normalize(token)] for token in tokens]
        for tokens in sentences
    ]

    n_sents = len(sents)
    voc_size = len(index)

    outfile.write(f'{n_sents:d} {voc_size:d}\n'.encode())
    for sent in sents:
        if len(sent) < 0x400:
            outfile.write((f'{len(sent):d}' + ''.join(f' {token:d}' for token in sent) + '\n').encode())
        else:
            outfile.write(b'0\n')
    outfile.flush()

    return TextIndex(index, prefix_len, suffix_len), n_sents, voc_size


def sentences_from_joint_file(joint_file, index=None):
    """Yield sentences from joint sentences file"""
    for i, line in enumerate(joint_file):
        fields = line.strip().split(' ||| ')
        if len(fields) != 2:
            raise ValueError(f'line {i+1} does not contain a single ||| separator, or sentence(s) are empty!')
        if index is None:
            yield fields[0], fields[1]
        else:
            yield fields[index]


def calculate_priors(src_sentences, trg_sentences,
                     fwd_alignments, rev_alignments):
    """Calculate priors from alignments"""
    priors = Counter()
    hmmf_priors = Counter()
    hmmr_priors = Counter()
    ferf_priors = Counter()
    ferr_priors = Counter()
    for lineno, (src_sent, trg_sent, fwd_line, rev_line) in enumerate(
            zip(src_sentences, trg_sentences, fwd_alignments, rev_alignments)):
        src_sent = src_sent.strip().split()
        trg_sent = trg_sent.strip().split()
        fwd_links = [tuple(map(int, s.split('-'))) for s in fwd_line.split()]
        rev_links = [tuple(map(int, s.split('-'))) for s in rev_line.split()]
        for i, j in fwd_links:
            if i >= len(src_sent) or j >= len(trg_sent):
                raise ValueError(f'alignment out of bounds in line {lineno + 1}: ({i}, {j})')
            priors[(src_sent[i], trg_sent[j])] += 1

        last_j = -1
        last_i = -1
        for i, j in sorted(fwd_links, key=itemgetter(1)):
            if j != last_j:
                hmmf_priors[i - last_i] += 1
            last_i = i
            last_j = j
        hmmf_priors[len(src_sent) - last_i] += 1

        last_j = -1
        last_i = -1
        for i, j in sorted(rev_links, key=itemgetter(0)):
            if i != last_i:
                hmmr_priors[j - last_j] += 1
            last_i = i
            last_j = j
        hmmr_priors[len(trg_sent) - last_j] += 1

        fwd_fert = Counter(i for i, j in fwd_links)
        rev_fert = Counter(j for i, j in rev_links)
        for i, fert in fwd_fert.items():
            ferf_priors[(src_sent[i], fert)] += 1
        for j, fert in rev_fert.items():
            ferr_priors[(trg_sent[j], fert)] += 1
    # TODO: confirm EOF in all files
    return priors, hmmf_priors, hmmr_priors, ferf_priors, ferr_priors


def write_priors(priorsf, priors_list, hmmf_priors, hmmr_priors,
                 ferf_priors, ferr_priors):
    """Write priors to file object"""
    for (src, trg), alpha in sorted(priors_list.items()):
        print('LEX\t%s\t%s\t%g' % (src, trg, alpha), file=priorsf)
    for (src, fert), alpha in sorted(ferf_priors.items()):
        print('FERF\t%s\t%d\t%g' % (src, fert, alpha), file=priorsf)
    for (trg, fert), alpha in sorted(ferr_priors.items()):
        print('FERR\t%s\t%d\t%g' % (trg, fert, alpha), file=priorsf)
    for jump, alpha in sorted(hmmf_priors.items()):
        print('HMMF\t%d\t%g' % (jump, alpha), file=priorsf)
    for jump, alpha in sorted(hmmr_priors.items()):
        print('HMMR\t%d\t%g' % (jump, alpha), file=priorsf)


def read_priors(priors_file):
    """Load priors from file object"""
    priors_list = []    # list of (srcword, trgword, alpha)
    ferf_priors = []    # list of (wordform, alpha)
    ferr_priors = []    # list of (wordform, alpha)
    hmmf_priors = {}    # dict of jump: alpha
    hmmr_priors = {}    # dict of jump: alpha
    # 5 types of lines valid:
    #
    # LEX   srcword     trgword     alpha   | lexical prior
    # HMMF  jump        alpha               | target-side HMM prior
    # HMMR  jump        alpha               | source-side HMM prior
    # FERF  srcword     fert   alpha        | source-side fertility p.
    # FERR  trgword     fert    alpha       | target-side fertility p.
    for i, line in enumerate(priors_file):
        fields = line.rstrip('\n').split('\t')
        try:
            alpha = float(fields[-1])
        except ValueError as err:
            logger.error(f'priors line {i+1} contains alpha value of "{fields[2]}" which is not numeric')
            raise err
        if fields[0] == 'LEX' and len(fields) == 4:
            priors_list.append((fields[1], fields[2], alpha))
        elif fields[0] == 'HMMF' and len(fields) == 3:
            hmmf_priors[int(fields[1])] = alpha
        elif fields[0] == 'HMMR' and len(fields) == 3:
            hmmr_priors[int(fields[1])] = alpha
        elif fields[0] == 'FERF' and len(fields) == 4:
            ferf_priors.append((fields[1], int(fields[2]), alpha))
        elif fields[0] == 'FERR' and len(fields) == 4:
            ferr_priors.append((fields[1], int(fields[2]), alpha))
        else:
            logger.error('priors line %d is invalid', i + 1)
            raise ValueError(f'Invalid input on line {i+1}')
    return Priors(priors_list, hmmf_priors, hmmr_priors, ferf_priors, ferr_priors)


def to_eflomal_priors_file(priors, src_index, trg_index, outfile):
    """Write priors to a file read by eflomal binary

    Arguments:

    priors - tuple of priors (priors_list, hmmf_priors, hmmr_priors,
             ferf_priors, ferr_priors)
    src_index - vocabulary index for source text
    tgt_index - vocabulary index for target text
    outfile - file object for output

    """
    priors_list, hmmf_priors, hmmr_priors, ferf_priors, ferr_priors = priors
    priors_indexed = {}
    for src_word, trg_word, alpha in priors_list:
        if src_word == '<NULL>':
            e = 0
        else:
            e = src_index[src_word]

        if trg_word == '<NULL>':
            f = 0
        else:
            f = trg_index[trg_word]

        if (e is not None) and (f is not None):
            priors_indexed[(e, f)] = priors_indexed.get((e, f), 0.0) \
                + alpha
    ferf_indexed = {}
    for src_word, fert, alpha in ferf_priors:
        e = src_index[src_word]
        if e is not None:
            ferf_indexed[(e, fert)] = \
                ferf_indexed.get((e, fert), 0.0) + alpha
    ferr_indexed = {}
    for trg_word, fert, alpha in ferr_priors:
        f = trg_index[trg_word]
        if f is not None:
            ferr_indexed[(f, fert)] = \
                ferr_indexed.get((f, fert), 0.0) + alpha
    logger.info('%d (of %d) pairs of lexical priors used',
                len(priors_indexed), len(priors_list))
    print('%d %d %d %d %d %d %d' % (
        len(src_index)+1, len(trg_index)+1, len(priors_indexed),
        len(hmmf_priors), len(hmmr_priors),
        len(ferf_indexed), len(ferr_indexed)),
          file=outfile)
    for (e, f), alpha in sorted(priors_indexed.items()):
        print('%d %d %g' % (e, f, alpha), file=outfile)
    for jump, alpha in sorted(hmmf_priors.items()):
        print('%d %g' % (jump, alpha), file=outfile)
    for jump, alpha in sorted(hmmr_priors.items()):
        print('%d %g' % (jump, alpha), file=outfile)
    for (e, fert), alpha in sorted(ferf_indexed.items()):
        print('%d %d %g' % (e, fert, alpha), file=outfile)
    for (f, fert), alpha in sorted(ferr_indexed.items()):
        print('%d %d %g' % (f, fert, alpha), file=outfile)
    outfile.flush()
