import unittest

from aciq.helpers import Context, ContextVar

VARIABLE = ContextVar("VARIABLE", 0)

class TestContextVars(unittest.TestCase):
  # Ensuring that the test does not modify variables outside the tests.
  ctx = Context()
  def setUp(self): TestContextVars.ctx.__enter__()
  def tearDown(self): TestContextVars.ctx.__exit__()

  def test_initial_value_is_set(self):
    _TMP = ContextVar("_TMP", 5)
    self.assertEqual(_TMP.value, 5)

  def test_cannot_recreate(self):
    _TMP2 = ContextVar("_TMP2", 1)
    with self.assertRaises(RuntimeError):
      _TMP2 = ContextVar("_TMP2", 2)

  def test_new_var_inside_context(self):
    with Context(VARIABLE=1):
      _TMP3 = ContextVar("_TMP3", 1)
    with self.assertRaises(RuntimeError):
      _TMP3 = ContextVar("_TMP3", 2)

  def test_value_across_modules(self):
    exec('from aciq.helpers import ContextVar;C = ContextVar("C", 13)', {})
    with self.assertRaises(RuntimeError):
      _C = ContextVar("C", 0)

  def test_assignment_across_modules(self):
    B = ContextVar("B", 1)
    B.value = 2
    self.assertEqual(B.value, 2)
    with self.assertRaises(RuntimeError):
      exec('from aciq.helpers import ContextVar;B = ContextVar("B", 0);B.value = 3;', {})

  def test_context_assignment(self):
    with Context(VARIABLE=1):
      self.assertEqual(VARIABLE.value, 1)
    self.assertEqual(VARIABLE.value, 0)

  def test_unknown_param_to_context(self):
    with self.assertRaises(KeyError):
      with Context(SOMETHING_ELSE=1):
        pass

  def test_nested_context(self):
    with Context(VARIABLE=1):
      with Context(VARIABLE=2):
        MORE = ContextVar("MORE", 2)
        with Context(VARIABLE=3, MORE=3):
          self.assertEqual(VARIABLE.value, 3)
          self.assertEqual(MORE.value, 3)
        self.assertEqual(VARIABLE.value, 2)
        self.assertEqual(MORE.value, 2)
      self.assertEqual(VARIABLE.value, 1)
      self.assertEqual(MORE.value, 2)
    self.assertEqual(VARIABLE.value, 0)

  def test_decorator(self):
    @Context(VARIABLE=1, DEBUG=4)
    def test():
      self.assertEqual(VARIABLE.value, 1)

    self.assertEqual(VARIABLE.value, 0)
    test()
    self.assertEqual(VARIABLE.value, 0)

  def test_context_exit_reverts_updated_values(self):
    D = ContextVar("D", 1)
    D.value = 2
    with Context(D=3):
      ...
    assert D.value == 2, f"Expected D to be 2, but was {D.value}. Indicates that Context.__exit__ did not restore to the correct value."


if __name__ == "__main__":
  unittest.main()
