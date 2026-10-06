# -*- coding: utf-8 -*-
"""Exercise the Bash workflow without network, audio dependencies, or a GPU.

Run: python -m unittest discover -s tests -p 'test_setup_and_train.py' -v
The real generator main functions run with a small file-writing test double.
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REMOTE = 'https://github.com/aqiilaah/TA-speech-separation.git'
LEGACY_REMOTE = 'https://github.com/Fadil-Tao/TA-speech-separation.git'

FAKE_PYTHON = r'''import ast
import json
import os
import shlex
import shutil
import sys
from pathlib import Path

def record(event, **values):
    with open(os.environ['FAKE_TRACE'], 'a') as stream:
        stream.write(json.dumps({'event': event, **values}) + '\n')

args = sys.argv[1:]
if Path(sys.argv[0]).name == 'uv':
    managed = Path(os.environ['FAKE_MANAGED']) / 'python3.11'
    if args == ['python', 'install', '3.11']:
        managed.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sys.argv[0], managed)
        managed.chmod(0o755)
        record('python_install')
    elif args == ['python', 'find', '--managed-python', '3.11']:
        assert managed.exists()
        print(managed)
    else:
        raise AssertionError(f'Unexpected uv command: {args}')
    sys.exit(0)
if Path(sys.argv[0]).name == 'gdown' or args[:2] == ['-m', 'gdown']:
    import argparse
    if args[:2] == ['-m', 'gdown']:
        args = args[2:]
    parser = argparse.ArgumentParser(prog='gdown')
    parser.add_argument('url_or_id')
    parser.add_argument('-O', required=True)
    parsed = parser.parse_args(args)
    shutil.copyfile(os.environ['FAKE_ZIP'], parsed.O)
    record('download', args=args,
           via_module=Path(sys.argv[0]).name != 'gdown')
    sys.exit(0)
if args[:2] == ['-m', 'venv']:
    root = Path(args[2])
    (root / 'bin').mkdir(parents=True)
    for name in ('python', 'gdown'):
        target = root / 'bin' / name
        shutil.copyfile(sys.argv[0], target)
        target.chmod(0o755)
    (root / 'bin' / 'activate').write_text(
        'export PATH=' + shlex.quote(str(root.resolve() / 'bin')) + ':"$PATH"\n'
    )
    record('venv_create')
    sys.exit(0)
if (args[:2] in (['-m', 'pip'], ['-m', 'ensurepip'])
        or args == ['-c', 'import pip']):
    sys.exit(0)
if args and args[0] == '-c' and 'sys.version_info' in args[1]:
    invalid = (
        os.environ.get('FAKE_BAD_SYSTEM') == '1'
        and Path(sys.argv[0]).parent == Path(os.environ['FAKE_BIN'])
    ) or (Path(sys.argv[0]).parent.parent / '.invalid-python').exists()
    if invalid:
        if 'assert' in args[1]:
            raise AssertionError(
                'Use Python 3.10 or 3.11 for this legacy ESPnet setup'
            )
        sys.exit(1)
    sys.exit(0)
if args and args[0].startswith('dataset/generator/'):
    import argparse
    file = Path(args[0])
    tree = ast.parse(file.read_text())
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == 'main')
    speakers = 3 if '3spk' in file.name else 2

    class Generator:
        def __init__(self, **kwargs):
            self.output = Path(kwargs['output_dir'])

        def split_utterances(self, **kwargs):
            return ({}, {}, {})

        def generate_mixtures_from_utterances(self, **kwargs):
            split, count = kwargs['split_name'], kwargs['num_mixtures']
            record('generate', speakers=speakers, split=split)
            for name in ['mix'] + [f's{i}' for i in range(1, speakers + 1)]:
                directory = self.output / split / name
                directory.mkdir(parents=True, exist_ok=True)
                for index in range(count):
                    (directory / f'{index}.wav').touch()
            return count

        def generate_dataset_info(self, train, dev, test, duration):
            (self.output / 'dataset_info.json').write_text(
                json.dumps({'train': train, 'dev': dev, 'test': test})
            )

    namespace = {
        'argparse': argparse,
        'get_raw_dir': lambda value: value,
        'get_synthetic_dir': lambda name, value: value,
        f'TITMLMixGenerator{speakers}Spk': Generator,
    }
    sys.argv = args
    exec(compile(ast.Module(body=[main], type_ignores=[]), str(file), 'exec'),
         namespace)
    namespace['main']()
    sys.exit(0)
if args and args[0] == '-u':
    file = Path(args[1])
    tree = ast.parse(file.read_text())
    flags = {
        node.args[0].value for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'add_argument' and node.args
    }
    assert '--num-epochs' in flags
    if '--resume-from' in args:
        assert '--resume-from' in flags
    dataset_call = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'DynamicMixDataset'
    )
    epoch_size_node = next(keyword.value for keyword in dataset_call.keywords
                           if keyword.arg == 'epoch_size')
    epoch_size = eval(compile(ast.Expression(epoch_size_node), str(file), 'eval'),
                      {'os': os})
    epochs = int(args[args.index('--num-epochs') + 1])
    if os.environ.get('FAKE_INTERRUPT') == '1':
        epochs -= 1
    speakers, variant = file.parts[1].replace('speaker', ''), file.parts[2]
    if '_transfer.py' in file.name:
        variant += '-transfer'
    destination = Path('checkpoints') / f'{speakers}speaker' / variant
    destination.mkdir(parents=True, exist_ok=True)
    history = {'train_losses': [-1] * epochs, 'val_losses': [-1] * epochs}
    (destination / 'training_history.json').write_text(json.dumps(history))
    (destination / 'best_model.pth').write_text(json.dumps({'epoch': epochs}))
    record('train', speakers=speakers, variant=variant, args=args,
           epoch_size=epoch_size)
    sys.exit(0)
os.execv(os.environ['REAL_PYTHON'], [os.environ['REAL_PYTHON']] + args)
'''


class SetupAndTrainTest(unittest.TestCase):
    """Validate clone, extraction, split generation, model routing, and reruns."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        for name in ('train', 'dataset/generator', 'utils'):
            shutil.copytree(
                ROOT / name, self.source / name,
                ignore=shutil.ignore_patterns('__pycache__'),
            )
        self.git('init', '--initial-branch=main')
        self.git('add', '.')
        self.git('-c', 'user.name=Fixture', '-c', 'user.email=test@example.test',
                 'commit', '-m', 'test fixture')
        self.script = self.root / 'setup_and_train.sh'
        shutil.copyfile(ROOT / 'setup_and_train.sh', self.script)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        for name in ('bash', 'sh', 'git', 'dirname', 'mkdir', 'mv', 'touch',
                     'tee', 'env', 'mktemp', 'cp', 'chmod'):
            (self.bin / name).symlink_to(shutil.which(name))
        self.python = self.bin / 'python3.11'
        self.python.write_text(f'#!{sys.executable}\n{FAKE_PYTHON}')
        self.python.chmod(0o755)
        for name in ('python', 'python3', 'python3.10'):
            shutil.copyfile(self.python, self.bin / name)
            (self.bin / name).chmod(0o755)
        curl = self.bin / 'curl'
        curl.write_text(
            f'#!{sys.executable}\n'
            'import json, os, sys\n'
            'assert "https://astral.sh/uv/install.sh" in sys.argv\n'
            'with open(os.environ["FAKE_TRACE"], "a") as stream:\n'
            '    stream.write(json.dumps({"event": "uv_install"}) + "\\n")\n'
            'if os.environ.get("FAKE_INSTALL_FAILURE") == "1":\n'
            '    sys.exit(22)\n'
            "print('mkdir -p \"$UV_INSTALL_DIR\"')\n"
            "print('cp \"$FAKE_TEMPLATE\" \"$UV_INSTALL_DIR/uv\"')\n"
            "print('chmod +x \"$UV_INSTALL_DIR/uv\"')\n"
        )
        curl.chmod(0o755)
        modules = self.root / 'modules'
        modules.mkdir()
        (modules / 'pyzipper.py').write_text(
            'from zipfile import ZipFile as AESZipFile\n'
        )
        (modules / 'torch.py').write_text(
            'import json\n'
            'from pathlib import Path\n'
            'def load(path, **kwargs):\n'
            '    return json.loads(Path(path).read_text())\n'
        )
        self.archive = self.root / 'fixture.zip'
        with zipfile.ZipFile(self.archive, 'w') as archive:
            for speaker in ('m01', 'm02', 'f01'):
                archive.writestr(f'TITML/Speech/{speaker}/audio.wav', b'audio')
        self.trace = self.root / 'trace.jsonl'
        self.env = {
            **os.environ,
            'REAL_PYTHON': sys.executable,
            'PYTHONPATH': str(modules),
            'FAKE_TRACE': str(self.trace),
            'FAKE_ZIP': str(self.archive),
            'FAKE_BIN': str(self.bin),
            'FAKE_MANAGED': str(self.root / 'managed python'),
            'FAKE_TEMPLATE': str(self.python),
            'HOME': str(self.root / 'home'),
            'PATH': str(self.bin),
            'GIT_CONFIG_COUNT': '2',
            'GIT_CONFIG_KEY_0': f'url.{self.source.as_uri()}.insteadOf',
            'GIT_CONFIG_VALUE_0': REMOTE,
            'GIT_CONFIG_KEY_1': f'url.{self.source.as_uri()}.insteadOf',
            'GIT_CONFIG_VALUE_1': LEGACY_REMOTE,
        }

    def git(self, *args):
        subprocess.run(
            ['git', '-C', str(self.source), *args],
            check=True, capture_output=True, text=True,
        )

    def run_wizard(self, model=6, epochs=1, configure=True, hours='1'):
        self.project = self.root / f'project with spaces {model}'
        answers = [str(self.project), 'cpu', '0',
                   str(model), str(epochs)]
        if model in (5, 6, 9):
            answers.append('1')
        answers.extend([str(hours), ''])
        return subprocess.run(
            ['bash', str(self.script), *(['--configure'] if configure else [])],
            input='\n'.join(answers) + '\n' if configure else '',
            capture_output=True, text=True, env=self.env, timeout=30,
        )

    def events(self):
        if not self.trace.exists():
            return []
        return [json.loads(line) for line in self.trace.read_text().splitlines()]

    def test_python_312_automatically_installs_311(self):
        self.env['FAKE_BAD_SYSTEM'] = '1'
        shutil.copyfile(self.python, self.bin / 'uv')
        (self.bin / 'uv').chmod(0o755)
        self.project = self.root / 'project with spaces 1'
        choices = {
            'PROJECT_DIR': str(self.project), 'PYTHON_BIN': str(self.python),
            'DEVICE': 'cpu', 'GPU_ID': '0', 'MODEL': '1', 'EPOCHS': '1',
            'PRETRAIN_EPOCHS': '1', 'TARGET_HOURS': '1',
            'ZIP_SOURCE': 'https://drive.google.com/uc?id=fixture',
        }
        (self.root / 'setup_and_train.env').write_text(''.join(
            f'{key}={shlex.quote(value)}\n' for key, value in choices.items()
        ))
        result = self.run_wizard(model=1, configure=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('[7/7] Finished', result.stdout)
        self.assertEqual(sum(e['event'] == 'python_install'
                             for e in self.events()), 1)
        self.assertNotIn('Python executable', result.stdout)
        before = self.events()
        result = self.run_wizard(model=1, configure=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.events(), before)

    def test_missing_uv_is_bootstrapped_without_changing_system_python(self):
        self.env['FAKE_BAD_SYSTEM'] = '1'
        original = self.python.read_bytes()
        result = self.run_wizard(model=1)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.python.read_bytes(), original)
        events = [e['event'] for e in self.events()]
        self.assertEqual(events.count('uv_install'), 1)
        self.assertEqual(events.count('python_install'), 1)
        self.assertTrue((self.root / 'home/.local/bin/uv').exists())
        before = self.events()
        result = self.run_wizard(model=1, configure=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.events(), before)

    def test_correct_python_and_venv_skip_installation(self):
        result = self.run_wizard(model=1)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        events = [e['event'] for e in self.events()]
        self.assertNotIn('uv_install', events)
        self.assertNotIn('python_install', events)
        self.assertEqual(events.count('venv_create'), 1)
        self.assertNotIn('Python executable', result.stdout)
        before = self.events()
        result = self.run_wizard(model=1, configure=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('creation skipped', result.stdout)
        self.assertEqual(self.events(), before)

    def test_incompatible_venv_is_preserved_and_replaced(self):
        result = self.run_wizard(model=1)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        venv = self.project / '.venv'
        (venv / '.invalid-python').touch()
        (venv / 'user-notes.txt').write_text('keep this')
        result = self.run_wizard(model=1, configure=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        backups = list(self.project.glob('.venv.backup.*/original'))
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / 'user-notes.txt').read_text(), 'keep this')
        self.assertFalse((venv / '.invalid-python').exists())
        self.assertEqual(sum(e['event'] == 'venv_create'
                             for e in self.events()), 2)
        self.assertEqual(sum(e['event'] == 'train'
                             for e in self.events()), 1)

    def test_failed_bootstrap_stops_before_clone(self):
        self.env.update(FAKE_BAD_SYSTEM='1', FAKE_INSTALL_FAILURE='1')
        result = self.run_wizard(model=1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Could not install uv', result.stderr)
        self.assertFalse(self.project.exists())
        self.assertEqual(self.events(), [{'event': 'uv_install'}])

    def test_gdown_download_works_without_removed_fuzzy_option(self):
        result = self.run_wizard(model=1)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        download = next(e for e in self.events()
                        if e['event'] == 'download')
        self.assertTrue(download['via_module'])
        self.assertNotIn('--fuzzy', download['args'])
        self.assertIn('1ETEzZhm5s5XAp1tKUtSj9zq4Ic-7tben',
                      download['args'][0])
        self.assertTrue((self.project / 'dataset/zips/TITML-IDN.zip').exists())
        self.assertIn('[7/7] Finished', result.stdout)

    def test_all_nine_models_and_only_dev_test(self):
        for model in range(1, 10):
            with self.subTest(model=model):
                result = self.run_wizard(model)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('[7/7] Finished', result.stdout)
                for info in self.project.glob('dataset/synthetic/*/dataset_info.json'):
                    self.assertEqual(json.loads(info.read_text())['train'], 0)
                    self.assertFalse((info.parent / 'train').exists())
        self.assertTrue(all(event['split'] in ('dev', 'test')
                            for event in self.events()
                            if event['event'] == 'generate'))
        training = [e for e in self.events() if e['event'] == 'train']
        self.assertTrue(all(event['epoch_size'] == 576 for event in training))

    def test_default_clone_uses_current_repository(self):
        result = self.run_wizard(model=8)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        remote = subprocess.check_output(
            ['git', '-C', str(self.project), 'config', '--get', 'remote.origin.url'],
            text=True,
        ).strip()
        self.assertEqual(remote, REMOTE)

    def test_default_ten_hours_controls_train_dev_and_test(self):
        result = self.run_wizard(hours='')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Total dataset hours', result.stdout)
        self.assertIn('Dynamic train: 5760', result.stdout)
        for info in self.project.glob('dataset/synthetic/*/dataset_info.json'):
            self.assertEqual(json.loads(info.read_text()),
                             {'train': 0, 'dev': 720, 'test': 720})
            self.assertFalse((info.parent / 'train').exists())
        training = [e for e in self.events() if e['event'] == 'train']
        self.assertEqual(len(training), 2)
        self.assertTrue(all(event['epoch_size'] == 5760 for event in training))
        config = self.root / 'setup_and_train.env'
        self.assertIn('TARGET_HOURS=10\n', config.read_text())
        before = self.events()
        result = self.run_wizard(configure=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.events(), before)

    def test_fifty_hours_reports_dynamic_train_and_static_duration(self):
        for model in (1, 3):
            with self.subTest(model=model):
                result = self.run_wizard(model=model, hours='50')
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('Dynamic train: 28800', result.stdout)
                self.assertIn('Target total hours: 50.0h', result.stdout)
                self.assertIn('Static WAV duration: ~10.0 hours', result.stdout)
                self.assertIn('Dynamic train per epoch: ~40.0 hours', result.stdout)
                self.assertNotIn('Total duration: ~10.0 hours', result.stdout)
                info = next(self.project.glob('dataset/synthetic/*/dataset_info.json'))
                self.assertEqual(json.loads(info.read_text()),
                                 {'train': 0, 'dev': 3600, 'test': 3600})
        training = [e for e in self.events() if e['event'] == 'train']
        self.assertTrue(all(e['epoch_size'] == 28800 for e in training))

    def test_missing_selected_training_script_stops_before_setup(self):
        path = Path('train/3speaker/skim-multiscale/train_skim_multiscale_3spk.py')
        (self.source / path).unlink()
        self.git('add', '.')
        self.git('-c', 'user.name=Fixture', '-c', 'user.email=test@example.test',
                 'commit', '-m', 'fixture missing selected model')
        result = self.run_wizard(model=8)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(str(path), result.stdout + result.stderr)
        self.assertEqual(self.events(), [], result.stdout + result.stderr)

    def test_transfer_pretraining_and_saved_settings_rerun(self):
        result = self.run_wizard()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        training = [e for e in self.events() if e['event'] == 'train']
        self.assertEqual([e['variant'] for e in training],
                         ['skim-attention', 'skim-attention-transfer'])
        before = self.events()
        result = self.run_wizard(configure=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.events(), before)
        config = self.root / 'setup_and_train.env'
        self.assertEqual(config.stat().st_mode & 0o777, 0o600)
        self.assertNotIn('PASSWORD', config.read_text())

    def test_changing_hours_preserves_existing_data_and_checkpoints(self):
        result = self.run_wizard(model=1, hours='1')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        before = self.events()
        result = self.run_wizard(model=1, hours='10')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('was not prepared for 10h', result.stderr)
        self.assertEqual(self.events(), before)

    def test_interrupted_training_resumes_without_false_finish(self):
        self.env['FAKE_INTERRUPT'] = '1'
        result = self.run_wizard(model=1, epochs=2)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('[7/7] Finished', result.stdout)
        self.env.pop('FAKE_INTERRUPT')
        result = self.run_wizard(model=1, configure=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        training = [e for e in self.events() if e['event'] == 'train']
        self.assertIn('--resume-from', training[-1]['args'])

    def test_old_remote_clone_is_rejected_before_setup(self):
        (self.source / 'train/datasets_utils.py').write_text('pass\n')
        self.git('add', '.')
        self.git('-c', 'user.name=Fixture', '-c', 'user.email=test@example.test',
                 'commit', '-m', 'old static fixture')
        result = self.run_wizard(model=1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Clone lacks DynamicMixDataset', result.stderr)
        self.assertEqual(self.events(), [])

    def test_zip_traversal_is_rejected(self):
        with zipfile.ZipFile(self.archive, 'a') as archive:
            archive.writestr('../escape.txt', 'unsafe')
        result = self.run_wizard(model=1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Unsafe ZIP entry', result.stderr)
        self.assertFalse((self.project / 'dataset/raw/escape.txt').exists())


if __name__ == '__main__':
    unittest.main()
