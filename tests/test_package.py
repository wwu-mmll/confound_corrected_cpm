import unittest
from cpm.module import hello_world


class TestHelloWorld(unittest.TestCase):
    def test_hello_world(self):
        self.assertEqual(hello_world(), "Hello, world!")


if __name__ == '__main__':
    unittest.main()