#!/usr/bin/env python3
"""Small real EGL discard/texture A/B; compiles only the probe, never runtimes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--pvrgpu-prefix', type=Path, required=True)
parser.add_argument('--llvmpipe-prefix', type=Path, required=True)
parser.add_argument('--bridge', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True, help='New directory outside source tree')
parser.add_argument('--cc', default='/usr/local/opt/llvm/bin/clang' if sys.platform == 'darwin' else 'cc')
parser.add_argument('--mode', choices=('color', 'helper-image', 'discard-image'), default='color')
parser.add_argument('--case', help='One width,mask,triangle triple; otherwise small matrix')
parser.add_argument('--timeout', type=int, default=60)
args = parser.parse_args()
root = args.output.resolve()
if root == REPO or REPO in root.parents or root.exists():
    parser.error('Choose a new output directory outside the repository')
prefixes = {'llvmpipe': args.llvmpipe_prefix.resolve(), 'pvrgpu': args.pvrgpu_prefix.resolve()}
bridge = args.bridge.resolve()
if not bridge.is_file():
    parser.error('Bridge snapshot missing')
mode = ('color', 'helper-image', 'discard-image').index(args.mode)
cases = [tuple(map(int, args.case.split(',')))] if args.case else (
    [(size, mask, triangle) for size in (2, 8) for mask in (0, 1, 2) for triangle in (0, 1)]
    if mode == 0 else [(8, 2, 1)])
if any(len(c) != 3 or c[0] not in (2, 8) or c[1] not in (0, 1, 2) or c[2] not in (0, 1) for c in cases):
    parser.error('Case must be width(2|8),mask(0|1|2),triangle(0|1)')
root.mkdir(parents=True)
source = REPO / 'tests/discard_texture_live_probe.c'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
suffix = 'dylib' if sys.platform == 'darwin' else 'so'
results = []
runtime_hashes = {}
for backend, prefix in prefixes.items():
    lib = prefix / 'lib'
    gallium = {p.resolve() for p in lib.glob('libgallium*.' + suffix + '*') if p.is_file()}
    if len(gallium) != 1:
        parser.error(f'Expected one actual Gallium payload under {lib}')
    gallium = gallium.pop()
    libraries = [lib / ('libEGL.' + suffix), lib / ('libGLESv2.' + suffix), gallium]
    if backend == 'pvrgpu':
        libraries.append(bridge)
    for p in libraries:
        runtime_hashes[str(p.resolve())] = sha(p)
    binary = root / ('probe-' + backend)
    command = [*shlex.split(args.cc), '-std=c11', '-O1', '-I' + str(prefix / 'include'),
               str(source), '-L' + str(lib), '-Wl,-rpath,' + str(lib), '-lEGL', '-lGLESv2', '-o', str(binary)]
    subprocess.run(command, check=True)
    for width, mask, triangle in cases:
        out = root / f'{backend}-{width}-{mask}-{triangle}'
        out.mkdir()
        for name in ('dri', 'model', 'tmp', 'cache'):
            (out / name).mkdir()
        (out / 'dri' / ('swrast_dri.' + suffix)).symlink_to(gallium)
        env = {k: v for k, v in os.environ.items() if not k.startswith(
            ('PVRGPU_', 'PCO_', 'MESA_', 'MESA_COUNTER_', 'GALLIUM_', 'LIBGL_', 'DYLD_', 'LD_', 'EGL_'))}
        configured = {'LC_ALL': 'C', 'EGL_PLATFORM': 'surfaceless', 'LIBGL_ALWAYS_SOFTWARE': '1',
            'GALLIUM_DRIVER': backend, 'MESA_LOADER_DRIVER_OVERRIDE': 'swrast',
            'MESA_SHADER_CACHE_DISABLE': 'true', 'MESA_GLES_VERSION_OVERRIDE': '3.1',
            'LIBGL_DRIVERS_PATH': str(out / 'dri'), 'TMPDIR': str(out / 'tmp'),
            'XDG_CACHE_HOME': str(out / 'cache')}
        if sys.platform == 'darwin':
            configured.update(DYLD_LIBRARY_PATH=str(lib), DYLD_PRINT_LIBRARIES='1')
        else:
            configured.update(LD_LIBRARY_PATH=str(lib), LD_DEBUG='libs')
        if backend == 'pvrgpu':
            configured.update(PVRGPU_SYSTEMC_API_LIB=str(bridge),
                PVRGPU_DRIVER_COMMAND_OUT=str(out / 'driver-command.txt'),
                PVRGPU_DRIVER_COUNTER_OUT=str(out / 'driver-counter.txt'),
                PVRGPU_SYSTEMC_JSONL_OUT=str(out / 'model.jsonl'),
                PVRGPU_SYSTEMC_STDERR_OUT=str(out / 'model.stderr.log'),
                PVRGPU_SYSTEMC_OUTDIR=str(out / 'model'))
        env.update(configured)
        command = [str(binary), str(out), str(width), str(mask), str(triangle), str(mode)]
        with (out / 'stdout.log').open('w') as stdout, (out / 'stderr.log').open('w') as stderr:
            try:
                code = subprocess.run(command, env=env, stdout=stdout, stderr=stderr,
                                      timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                code = 'timeout'
        log = (out / 'stderr.log').read_text(errors='replace')
        library_evidence = {str(p.resolve()): str(p.resolve()) in log for p in libraries}
        driver = (out / 'driver-counter.txt').read_text() if (out / 'driver-counter.txt').exists() else ''
        events = re.findall(r'\bevent=([^\s]+)', driver)
        models = []
        model_errors = []
        if (out / 'model.jsonl').exists():
            for line in (out / 'model.jsonl').read_text().splitlines():
                if not line.strip() or line.startswith('@CAPTURE: '):
                    continue
                try:
                    models.append(json.loads(line))
                except json.JSONDecodeError:
                    model_errors.append(line)
        native_done = backend != 'pvrgpu' or (
            events.count('draw_array_primitive_recorded') == 1 and 'systemc_api_done' in events and
            not model_errors and
            any(r.get('type') == 'hello' and r.get('driver_command') == 'draw_pco_sequence' and
                r.get('driver_command_width') == width and r.get('driver_command_height') == width for r in models) and
            any(r.get('type') == 'done' and r.get('physical_submissions') == 1 and
                r.get('pool_leaks') == 0 and r.get('pool_bytes_in_flight') == 0 and
                r.get('pool_allocations') == r.get('pool_releases') for r in models) and
            not any('error' in e or e == 'unsupported_draw' for e in events))
        row = {'backend': backend, 'case': [width, mask, triangle], 'mode': args.mode, 'exit': code,
               'output': str(out), 'command': command, 'environment': configured,
               'actual_library_loads': library_evidence, 'native_completed': native_done,
               'model_parse_errors': model_errors,
               'driver_events': {e: events.count(e) for e in sorted(set(events))},
               'counters': [r.get('counters') for r in models if r.get('type') == 'counter'],
               'pass': code == 0 and all(library_evidence.values()) and native_done}
        (out / 'receipt.json').write_text(json.dumps(row, indent=2) + '\n')
        results.append(row)
        print(f'{backend} {width}/{mask}/{triangle}: exit={code} native={native_done}', flush=True)
comparisons = []
for width, mask, triangle in cases:
    pair = {r['backend']: r for r in results if r['case'] == [width, mask, triangle]}
    files = {}
    for name in ('color.rgba8',) if mode == 0 else ('color.rgba8', 'image.r32ui'):
        a = Path(pair['llvmpipe']['output']) / name
        b = Path(pair['pvrgpu']['output']) / name
        files[name] = a.is_file() and b.is_file() and a.read_bytes() == b.read_bytes()
    comparisons.append({'case': [width, mask, triangle], 'raw_equal': files,
                        'pass': all(r['pass'] for r in pair.values()) and all(files.values())})
unchanged = all(sha(Path(p)) == value for p, value in runtime_hashes.items())
report = {'source': str(source), 'source_sha256': sha(source), 'runtime_sha256': runtime_hashes,
          'runtime_unchanged': unchanged, 'results': results, 'comparisons': comparisons,
          'note': 'Discard+image refusal is not a correct rendered image; mode discard-image may explicitly fail closed.',
          'pass': unchanged and all(c['pass'] for c in comparisons)}
(root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(f"{'PASS' if report['pass'] else 'FAIL'} {root / 'report.json'}")
sys.exit(0 if report['pass'] else 1)
