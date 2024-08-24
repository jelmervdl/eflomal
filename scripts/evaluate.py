#!/usr/bin/env python3

# Script to evaluate efmaral or fast_align using WPT shared task data sets.
#
#   time python3 scripts/evaluate.py efmaral test.eng.hin.wa \
#       test.eng test.hin training.eng training.hin
#
# or, to use fast_align:
#
#   time python3 scripts/evaluate.py fast_align test.eng.hin.wa \
#       test.eng test.hin training.eng training.hin
#
# atools (from the fast_align package) must be installed and in $PATH

from contextlib import ExitStack
import re, sys, subprocess, os
from multiprocessing import Pool
from tempfile import NamedTemporaryFile
from functools import partial
from shutil import copyfileobj

RE_NUMBERED = re.compile(r'<s snum=(\d+)>(.*?)</s>\s*$')

def wpteval(align, train_filenames, test_filename, gold_wa, *, train=True, test=True, batch_size=1, postprocessor=None):
    test_numbers = []
    test_lines = []

    mosesf = NamedTemporaryFile('w+', dir='.', prefix=f'{os.getpid()}_test_', suffix='.fwd', delete=False, encoding='utf-8')

    with NamedTemporaryFile('w+', dir='.', prefix=f'{os.getpid()}_text_', suffix='.1', delete=False, encoding='utf-8') as outf1, \
         NamedTemporaryFile('w+', dir='.', prefix=f'{os.getpid()}_text_', suffix='.2', delete=False, encoding='utf-8') as outf2:
        if test:
            with open(test_filename[0], 'r', encoding='utf-8') as f:
                for i,line in enumerate(f):
                    m = RE_NUMBERED.match(line)
                    assert m, 'Test data file %s not numbered properly!' % \
                              test_filename[0]
                    test_numbers.append(m.group(1))
                    print(m.group(2).strip(), file=outf1)
                    test_lines.append([m.group(2).strip(), None])
            with open(test_filename[1], 'r', encoding='utf-8') as f:
                for i,line in enumerate(f):
                    m = RE_NUMBERED.match(line)
                    assert m, 'Test data file %s not numbered properly!' % \
                              test_filename[1]
                    assert test_numbers[i] == m.group(1)
                    print(m.group(2).strip(), file=outf2)
                    test_lines[i][1] = m.group(2).strip()

        if train:
            for filename1,filename2 in train_filenames:
                with open(filename1, 'r', encoding='utf-8') as f1, \
                     open(filename2, 'r', encoding='utf-8') as f2:
                    while True:
                        line1 = f1.readline()
                        line2 = f2.readline()
                        assert (not line1) == (not line2), \
                               'Number of lines differs between %s and %s!' % (
                               filename1, filename2)
                        if (not line1) or (not line2): break
                        line1 = line1.strip()
                        line2 = line2.strip()
                        if line1 and line2:
                            print(line1, file=outf1)
                            print(line2, file=outf2)

        outf1.flush()
        outf2.flush()

        outf1.seek(0)
        outf2.seek(0)

        at_end = False
        itf1 = iter(outf1)
        itf2 = iter(outf2)

        while not at_end:
            with NamedTemporaryFile('w', encoding='utf-8') as batchf1, \
                 NamedTemporaryFile('w', encoding='utf-8') as batchf2, \
                 NamedTemporaryFile('r', encoding='utf-8') as batchout:
                try:
                    for n in range(batch_size):
                        batchf1.write(next(itf1))
                        batchf2.write(next(itf2))
                except StopIteration:
                    at_end = True

                batchf1.flush()
                batchf2.flush()

                if batchf1.tell() > 0 and batchf2.tell() > 0:
                    align(batchf1.name, batchf2.name, batchout.name)
                    for line in batchout:
                        print(line.rstrip('\n'), file=mosesf)

    mosesf.flush()
    mosesf.seek(0)
    
    if postprocessor:
        postprocessor(mosesf)

    mosesf.seek(0)

    if test:
        with NamedTemporaryFile('w', encoding='utf-8') as outf, \
             open('./evaluation.tsv', 'w', encoding='utf-8') as fdebug:
            for lineno, test_pair, line in zip(test_numbers, test_lines, mosesf):
                print(f'{test_pair[0]}\t{test_pair[1]}\t{line.rstrip()}', file=fdebug)
                try:
                    for i,j in map(lambda s: s.split('-'), line.rstrip().split()):
                        print('%s %d %d' % (lineno, int(i)+1, int(j)+1), file=outf)
                except ValueError:
                    print(f'Unexpected line {lineno}: {line}', file=sys.stderr)
                    raise

            outf.flush()
            subprocess.call(
                    ['perl', '3rdparty/wa_check_align.pl', outf.name])
            subprocess.call(
                    ['perl', '3rdparty/wa_eval_align.pl', gold_wa, outf.name])

    mosesf.close()


def fastalign(args):
    in_filename, out_filename, reverse = args
    with open(out_filename, 'w') as outf:
        subprocess.call(
            ['fast_align', '-i', in_filename, '-d', '-o', '-v']
            if reverse else 
            ['fast_align', '-i', in_filename, '-d', '-o', '-v', '-r'],
            stdout=outf)


def main():
    symmetrization = 'grow-diag-final-and'
    if len(sys.argv) >= 8 and sys.argv[7] == '--symmetrization':
        symmetrization = sys.argv[8]
        extra_opts = sys.argv[9:]
    else:
        extra_opts = sys.argv[7:]

    def align_efmaral(text1, text2, output):
        subprocess.call(['scripts/align_symmetrize.sh', text1, text2, output,
                         symmetrization] + extra_opts)

    def align_fastalign(text1, text2, output):
        with NamedTemporaryFile('w', encoding='utf-8') as outf, \
             NamedTemporaryFile('w', encoding='utf-8') as fwdf, \
             NamedTemporaryFile('w', encoding='utf-8') as backf:
            subprocess.call(['scripts/wpt2fastalign.py', text1, text2],
                            stdout=outf)
            outf.flush()

            with Pool(2) as p:
                r = p.map(fastalign,
                          [(outf.name, fwdf.name, False),
                           (outf.name, backf.name, True)])

            with open(output, 'w') as outputf:
                subprocess.call(['atools', '-i', fwdf.name, '-j', backf.name,
                                 '-c', symmetrization], stdout=outputf)

    def train_eflomal_with_priors(priors, text1, text2, output):
        with NamedTemporaryFile('w', dir='.', prefix=f'{os.getpid()}_links_', suffix='.fwd', delete=False, encoding='utf-8') as fwdf, \
             NamedTemporaryFile('w', dir='.', prefix=f'{os.getpid()}_links_', suffix='.rev', delete=False, encoding='utf-8') as revf:
            print("Aligning...")
            subprocess.check_call(['eflomal-align', '--overwrite',
                '-v',
                '-s', text1, '-t', text2,
                '-f', fwdf.name, '-r', revf.name] + extra_opts)

            print("Making priors...")
            subprocess.check_call(['eflomal-makepriors',
                '-v',
                '-s', text1, '-t', text2,
                '-f', fwdf.name, '-r', revf.name,
                '-p', priors])

    def test_eflomal_python(text1, text2, output):
        print("Evaluating...")
        subprocess.check_call(['eflomal-align', '--overwrite',
            '-s', text1, '-t', text2,
            '-f', output] + extra_opts)

    def test_eflomal(text1, text2, output):
        print("Evaluating...")
        subprocess.check_call(['src/eflomal',
            '-s', text1,
            '-t', text2,
            '-f', output,
            '-m', '3',
            *extra_opts
        ])

    def retokenize(langs, vocab, fh):
        with NamedTemporaryFile('w+', encoding='utf-8') as fout:
            print('Re-tokenising...')
            subprocess.check_call([
                os.path.dirname(__file__) + '/retokenize-tests.py',
                *langs,
                vocab,
                *sys.argv[3:5],
                fh.name], stdout=fout)
            fout.seek(0)
            fh.seek(0)
            copyfileobj(fout, fh)
            assert fout.tell() == fh.tell()

            # Cut off the tail of the file
            fh.flush()
            os.truncate(fh.name, fh.tell())

    if sys.argv[1] == 'eflomal-with-priors':
        with ExitStack() as ctx:
            if '-p' not in extra_opts:
                tmp = ctx.enter_context(NamedTemporaryFile('w+b', dir='.', prefix=f'{os.getpid()}_priors_', suffix='.priors', delete=False))
                # Occupies the priors file
                wpteval(partial(train_eflomal_with_priors, tmp.name),
                    zip(sys.argv[5].split(','), sys.argv[6].split(',')),
                    (sys.argv[3], sys.argv[4]),
                    sys.argv[2], test=False, batch_size=int(1e9))
                extra_opts.append('-p')
                extra_opts.append(tmp.name)
            # Runs the actual evaluation
            postprocessor = None
            
            # Hack to check for SPM mode
            try:
                idx = extra_opts.index('--vocab')
                vocab = extra_opts[idx+1]
                del extra_opts[idx:idx+2]

                idx = extra_opts.index('--langs')
                langs = extra_opts[idx+1:idx+3]
                del extra_opts[idx:idx+3]

                postprocessor = partial(retokenize, langs, vocab)
            except ValueError:
                print("Not using spm->moses retokenization")
                pass

            wpteval(test_eflomal,
                zip(sys.argv[5].split(','), sys.argv[6].split(',')),
                (sys.argv[3], sys.argv[4]),
                sys.argv[2], train=False, batch_size=int(1e9), postprocessor=postprocessor)
    else:
        aligner = align_efmaral if sys.argv[1] in ('efmaral', 'eflomal') \
                  else align_fastalign
        wpteval(aligner,
                zip(sys.argv[5].split(','), sys.argv[6].split(',')),
                (sys.argv[3], sys.argv[4]),
                sys.argv[2], batch_size=int(1e9))

if __name__ == '__main__': main()

