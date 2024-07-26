import unittest


class TestHelloWorld(unittest.TestCase):
    def test_hello_world(self):
        self.assertEqual("Hello, world!", "Hello, world!")


if __name__ == '__main__':
    unittest.main()
