import unittest

from aciq.imagenet.class_containers import ImagenetClass, ImagenetClassIndex


class TestImagenetClassIndexLoad(unittest.TestCase):
  def test_loads_all_1000_classes(self):
    index = ImagenetClassIndex.load()
    self.assertEqual(len(index.classes), 1000)
    self.assertTrue(all(isinstance(c, ImagenetClass) for c in index.classes))

  def test_idx_matches_position(self):
    index = ImagenetClassIndex.load()
    for i, c in enumerate(index.classes):
      self.assertEqual(c.idx, i)

  def test_known_entries(self):
    classes = ImagenetClassIndex.load().classes
    self.assertEqual((classes[0].synset, classes[0].name), ("n01440764", "tench"))
    self.assertEqual((classes[999].synset, classes[999].name), ("n15075141", "toilet_tissue"))

  def test_synsets_unique(self):
    classes = ImagenetClassIndex.load().classes
    self.assertEqual(len({c.synset for c in classes}), len(classes))


class TestImagenetClassIndexSynsetToIdx(unittest.TestCase):
  def test_maps_synset_to_idx(self):
    mapping = ImagenetClassIndex.load().synset_to_idx
    self.assertEqual(len(mapping), 1000)
    self.assertEqual(mapping["n01440764"], 0)
    self.assertEqual(mapping["n15075141"], 999)


if __name__ == "__main__":
  unittest.main()
