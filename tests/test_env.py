import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inference_router.env import load_dotenv


class EnvTests(unittest.TestCase):
    def test_loads_key_and_preserves_shell_override(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / '.env'
            path.write_text('# comment\nDEEPINFRA_API_KEY=test-key\n')
            load_dotenv(path)
            self.assertEqual(os.environ['DEEPINFRA_API_KEY'], 'test-key')
            os.environ['DEEPINFRA_API_KEY'] = 'shell-key'
            load_dotenv(path)
            self.assertEqual(os.environ['DEEPINFRA_API_KEY'], 'shell-key')

    def test_quotes_comments_and_no_shell_execution_or_expansion(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / '.env'
            path.write_text('export A="literal#value" # comment\nB=plain # comment\nC=\'$(touch should-not-exist)$HOME\'\n')
            load_dotenv(path)
            self.assertEqual(os.environ['A'], 'literal#value')
            self.assertEqual(os.environ['B'], 'plain')
            self.assertEqual(os.environ['C'], '$(touch should-not-exist)$HOME')

    def test_missing_file_is_optional_and_errors_do_not_expose_value(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / '.env'
            load_dotenv(path)
            path.write_text('FIRST=not-applied\nBAD KEY=secret-example\n')
            with self.assertRaises(ValueError) as error:
                load_dotenv(path)
            self.assertNotIn('secret-example', str(error.exception))
            self.assertNotIn('FIRST', os.environ)
