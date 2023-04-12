"""Unit tests for eflomal"""

import io
import os
import tempfile
import unittest

import eflomal


DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'testdata')


class TestAlign(unittest.TestCase):

    def setUp(self):
        self.src_data = [
            'En svart katt .\n',
            'En gul fågel .\n',
            'En vit elefant .\n',
        ]
        self.trg_data = [
            'A black cat .\n',
            'A yellow bird .\n',
            'A white elephant .\n',
        ]
        self.priors_data = [
            'LEX\tsvart\tblack\t100\n',
            'LEX\tkatt\tcat\t100\n',
            'LEX\tgul\tyellow\t100\n',
            'LEX\tfågel\tbird\t100\n',
            'LEX\tvit\twhite\t100\n',
            'LEX\telefant\telephant\t100\n',
        ]

    def test_aligner(self):
        """Test aligner without priors producing rubbish"""
        aligner = eflomal.Aligner()
        with tempfile.NamedTemporaryFile('w+') as fwd_links, \
             tempfile.NamedTemporaryFile('w+') as rev_links:
            aligner.align(self.src_data, self.trg_data,
                          links_filename_fwd=fwd_links.name, links_filename_rev=rev_links.name,
                          quiet=False)
            fwd_links.seek(0)
            self.assertEqual(fwd_links.readlines(), [
                '3-1 1-2 3-3\n',
                '2-1 1-2 3-3\n',
                '2-1 1-2 3-3\n'
            ])
            
            rev_links.seek(0)
            self.assertEqual(rev_links.readlines(), [
                '1-3 2-1 3-3\n',
                '1-2 2-1 3-3\n',
                '1-2 2-1 3-3\n'
            ])

    def test_aligner_with_priors(self):
        """Test aligner with priors"""
        aligner = eflomal.Aligner()
        with tempfile.NamedTemporaryFile('w+') as fwd_links, \
             tempfile.NamedTemporaryFile('w+') as rev_links:
            aligner.align(self.src_data, self.trg_data,
                          links_filename_fwd=fwd_links.name, links_filename_rev=rev_links.name,
                          priors_input=self.priors_data, quiet=False)
            fwd_links.seek(0)
            self.assertEqual(fwd_links.readlines(), [
                '0-0 1-1 2-2 3-3\n',
                '0-0 1-1 2-2 3-3\n',
                '0-0 1-1 2-2 3-3\n'
            ])
            
            rev_links.seek(0)
            self.assertEqual(rev_links.readlines(), [
                '0-0 1-1 2-2 3-3\n',
                '0-0 1-1 2-2 3-3\n',
                '0-0 1-1 2-2 3-3\n'
            ])

    def test_makepriors(self):
        """Test creating priors"""
        aligner = eflomal.Aligner()
        with tempfile.NamedTemporaryFile('w+') as fwd_links, \
             tempfile.NamedTemporaryFile('w+') as rev_links:
            aligner.align(self.src_data, self.trg_data,
                          links_filename_fwd=fwd_links.name, links_filename_rev=rev_links.name,
                          quiet=False)
            fwd_links.seek(0)
            rev_links.seek(0)
            priors_tuple = eflomal.calculate_priors(
                self.src_data, self.trg_data, fwd_links.readlines(), rev_links.readlines())
            self.assertEqual(len(priors_tuple), 5)
            for prior_list in priors_tuple:
                self.assertGreater(len(prior_list), 0)
        with io.StringIO() as priorsf:
            eflomal.write_priors(priorsf, *priors_tuple)
            priorsf.seek(0)
            self.assertGreater(len(priorsf.readlines()), 5)
