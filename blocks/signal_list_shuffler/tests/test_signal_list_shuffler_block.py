from unittest.mock import patch
from nio import Signal
from nio.testing.block_test_case import NIOBlockTestCase
from ..signal_list_shuffler_block import SignalListShuffler


class TestSignalListShuffler(NIOBlockTestCase):

    def test_process_signals(self):
        """Signals pass through block unmodified."""
        blk = SignalListShuffler()
        # set a random seed for deterministic results
        self.configure_block(blk, {'seed': 1})
        blk.start()
        blk.process_signals([
            Signal({'foo': 0}), Signal({'foo': 1}), Signal({'foo': 2})])
        blk.stop()
        self.assert_num_signals_notified(3)
        # the order of this list is known because the seed has been set
        self.assert_signal_list_notified([
            Signal({'foo': 1}), Signal({'foo': 2}), Signal({'foo': 0})])

    @patch(SignalListShuffler.__module__ + '.random')
    def test_zero_seed(self, mock_random):
        """Random seed is set to 0"""
        blk = SignalListShuffler()
        self.configure_block(blk, {'seed': 0})
        blk.start()
        blk.stop()
        mock_random.seed.assert_called_once_with(0) 
