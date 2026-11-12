"""Regression checks for the separated collection and training entry points."""
from pathlib import Path
import unittest


class LayoutTests(unittest.TestCase):
    def test_collection_and_training_have_separate_packages(self):
        from eeg_collection.data_collecting import collect_data, collect_randomized
        from eeg_collection.training.cnn import focus_pipeline
        from eeg_collection.training.smoothing import tune_attention

        root = Path(__file__).resolve().parent
        self.assertEqual(Path(collect_data.__file__).parent, root / 'data_collecting')
        self.assertEqual(Path(collect_randomized.__file__).parent, root / 'data_collecting')
        self.assertEqual(Path(tune_attention.__file__).parent, root / 'training' / 'smoothing')
        self.assertEqual(focus_pipeline.DEFAULT_DATA, root / 'data_collecting' / 'data' / 'half_half')
