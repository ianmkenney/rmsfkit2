"""
Unit and regression test for the rmsfkit package.

The TestRMSF class was taken from the MDAnalysis rms tests file and
the relevant modules were switched.
"""

# Import package, test suite, and other packages as needed
from MDAnalysisTests.datafiles import GRO, XTC, rmsfArray
import MDAnalysis as mda

from numpy.testing import assert_equal, assert_almost_equal
import numpy as np
import os
import pytest

import rmsfkit
import sys

def test_rmsfkit_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "rmsfkit" in sys.modules


class TestRMSF(object):
    @pytest.fixture()
    def universe(self):
        return mda.Universe(GRO, XTC)

    def test_rmsf(self, universe):
        rmsfs = rmsfkit.RMSF(universe.select_atoms('name CA'))
        rmsfs.run()
        test_rmsfs = np.load(rmsfArray)

        assert_almost_equal(rmsfs.results.rmsf, test_rmsfs, 5,
                            err_msg="error: rmsf profile should match test "
                            "values")

    def test_rmsf_single_frame(self, universe):
        rmsfs = rmsfkit.RMSF(universe.select_atoms('name CA')).run(start=5, stop=6)

        assert_almost_equal(rmsfs.results.rmsf, 0, 5,
                            err_msg="error: rmsfs should all be zero")

    def test_rmsf_identical_frames(self, universe, tmpdir):

        outfile = os.path.join(str(tmpdir), 'rmsf.xtc')

        # write a dummy trajectory of all the same frame
        with mda.Writer(outfile, universe.atoms.n_atoms) as W:
            for _ in range(universe.trajectory.n_frames):
                W.write(universe)

        universe = mda.Universe(GRO, outfile)
        rmsfs = rmsfkit.RMSF(universe.select_atoms('name CA'))
        rmsfs.run()
        assert_almost_equal(rmsfs.results.rmsf, 0, 5,
                            err_msg="error: rmsfs should all be 0")