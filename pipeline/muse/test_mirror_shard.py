"""Tests for disjoint Muse archive shard planning."""

import unittest

from pipeline.muse.mirror_shard import selected_names


class MirrorShardTest(unittest.TestCase):
    """Verify bounded workers never receive overlapping archives."""

    def test_splits_english_archives_through_part_30(self) -> None:
        """Assign odd and even archives without overlap."""
        odd = selected_names("en", 0, 2, 1, 30)
        even = selected_names("en", 1, 2, 1, 30)

        self.assertEqual(odd[0], "en_part01_of_35.tar")
        self.assertEqual(odd[-1], "en_part29_of_35.tar")
        self.assertEqual(even[0], "en_part02_of_35.tar")
        self.assertEqual(even[-1], "en_part30_of_35.tar")
        self.assertFalse(set(odd) & set(even))
        self.assertEqual(len(odd) + len(even), 30)

    def test_assigns_last_five_to_tail_worker(self) -> None:
        """Assign parts 31 through 35 to one independent worker."""
        tail = selected_names("en", 0, 1, 31, 35)

        self.assertEqual(
            tail,
            [f"en_part{part:02d}_of_35.tar" for part in range(31, 36)],
        )


if __name__ == "__main__":
    unittest.main()
