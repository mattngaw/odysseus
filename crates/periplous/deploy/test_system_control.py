"""Privileged controller boundaries, loaded in isolation without host service access."""
import importlib.machinery
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch


class SystemControl(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='periplous-system-control-test-')
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source = Path(__file__).parent
        shutil.copyfile(source / 'periplousctl', root / 'periplousctl')
        shutil.copyfile(source / 'system/prodctl', root / 'prodctl')
        loader = importlib.machinery.SourceFileLoader('test_prodctl', str(root / 'prodctl'))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        self.control = importlib.util.module_from_spec(spec)
        loader.exec_module(self.control)

    def test_unprivileged_user_cannot_mutate_system_prod(self):
        for args in [['promote','/tmp/package'], ['bootstrap','/tmp/package'], ['rollback'],
                     ['recover'], ['start'], ['stop'], ['restart'], ['enable'], ['disable'],
                     ['tunnel','start'], ['tunnel','stop']]:
            with self.subTest(args=args), patch('sys.argv',['prodctl',*args]), \
                    patch.object(self.control.os,'geteuid',return_value=1000), \
                    patch.object(self.control,'manager') as manager:
                with self.assertRaisesRegex(RuntimeError,'sudo'):
                    self.control.main()
                manager.assert_not_called()

    def test_system_manager_has_no_dev_target(self):
        self.assertEqual(self.control.prod_unit('prod'),'periplous-prod.service')
        with self.assertRaises(ValueError):
            self.control.prod_unit('dev')


if __name__ == '__main__':
    unittest.main()
