#!/usr/bin/env bash
# Compile and compare both native indirect probes against already-built Mesa.
# Requires Python 3 standard library and a C compiler matching the Mesa ABI.
# Paths default to config/local.env; CLI flags take precedence.
# Example:
#   bash script/run_mesa_indirect_draw_probe.sh --output /external/new-probe-run
# Flags: --pvrgpu-prefix DIR --llvmpipe-prefix DIR --bridge FILE --cc "cc ..."
# No Mesa/SystemC build or install; existing output directories are rejected.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -f "${REPO_DIR}/config/local.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_DIR}/config/local.env"
    set +a
fi
PY="${PVRGPU_UI_PYTHON:-python3}"
command -v "${PY}" >/dev/null 2>&1 || {
    echo "run_mesa_indirect_draw_probe: Python not found: ${PY}" >&2
    exit 2
}
exec "${PY}" - "${REPO_DIR}" "${SCRIPT_DIR}/run_mesa_indirect_draw_probe.sh" "$@" <<'PYTHON'
"""Compile immutable native indirect probes; run/diff PvrGPU and llvmpipe."""
import argparse
from collections import Counter
import datetime
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

REPO = Path(sys.argv[1]).resolve()
RUNNER = Path(sys.argv[2]).resolve()
parser = argparse.ArgumentParser(description=__doc__,
    epilog='Both probes always run: 108 scenarios/queries and 16,640 DWORDs per backend. '
           'Only the probe executables are compiled; Mesa/SystemC are never rebuilt or installed.')
parser.add_argument('--pvrgpu-prefix', type=Path,
                    default=os.environ.get('PVRGPU_MESA_PVRGPU_PREFIX'))
parser.add_argument('--llvmpipe-prefix', type=Path,
                    default=os.environ.get('PVRGPU_LLVMPIPE_MESA_PREFIX'))
parser.add_argument('--bridge', type=Path, default=os.environ.get('PVRGPU_SYSTEMC_API_LIB'))
parser.add_argument('--output', type=Path, help='New artifact directory outside the repository; never overwritten.')
parser.add_argument('--cc', default=os.environ.get('CC', 'cc'),
                    help='C compiler command (default: CC or cc); must match the Mesa library architecture.')
parser.add_argument('--timeout', type=int, default=600, help='Per-probe runtime timeout in seconds.')
args = parser.parse_args(sys.argv[3:])
if sys.platform not in ('darwin', 'linux'):
    parser.error('Actual dynamic-library load verification is supported only on macOS and Linux.')
args.probes = ('vertex_fetch', 'producers')
if not args.pvrgpu_prefix or not args.llvmpipe_prefix:
    parser.error('Set both Mesa prefixes in config/local.env or pass --pvrgpu-prefix and --llvmpipe-prefix.')
if not args.bridge:
    build = os.environ.get('PVRGPU_BUILD_DIR')
    if not build:
        parser.error('Set PVRGPU_SYSTEMC_API_LIB/PVRGPU_BUILD_DIR or pass --bridge.')
    args.bridge = Path(build) / 'lib' / (
        'libpvrgpu_systemc_bridge.dylib' if sys.platform == 'darwin' else 'libpvrgpu_systemc_bridge.so')
if args.timeout <= 0:
    parser.error('--timeout must be positive.')
if not args.output:
    output_root = os.environ.get('PVRGPU_OUTPUT_ROOT')
    if not output_root:
        parser.error('Set PVRGPU_OUTPUT_ROOT or pass --output.')
    args.output = Path(output_root) / 'indirect-native' / datetime.datetime.now(
        datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
BASE = args.output.resolve()
if BASE == REPO or REPO in BASE.parents:
    parser.error('Output must be outside the repository.')
if BASE.exists():
    parser.error('Output already exists; select a new directory (nothing is overwritten).')
PREFIXES = {'pvrgpu': args.pvrgpu_prefix.resolve(), 'llvmpipe': args.llvmpipe_prefix.resolve()}
BRIDGE = args.bridge.resolve()
COMPILER = shlex.split(args.cc)
if not COMPILER or not shutil.which(COMPILER[0]):
    parser.error('--cc compiler executable was not found.')
COMPILER[0] = str(Path(shutil.which(COMPILER[0])).resolve())
SOURCES = {'vertex_fetch': REPO / 'tests/indirect_draw_vertex_fetch_probe.c',
           'producers': REPO / 'tests/indirect_draw_live_probe.c'}
EXPECTED = {'vertex_fetch': (88, 14080), 'producers': (20, 2560)}
EXPECTED_NATIVE_EVENTS = {
    'vertex_fetch': {'draw_indirect_decoded': 88, 'draw_array_primitive_recorded': 88,
                     'systemc_api_submit': 88, 'systemc_api_done': 88,
                     'compute_api_submit': 0, 'compute_api_done': 0, 'stream_output_readback': 88,
                     'query_statistics_completed': 88, 'query_primitives_emitted_result': 88},
    'producers': {'draw_indirect_decoded': 20, 'draw_indirect_empty': 8,
                  'draw_array_primitive_recorded': 16, 'stream_output_readback': 16,
                  'systemc_api_submit': 36, 'systemc_api_done': 36,
                  'compute_api_submit': 4, 'compute_api_done': 4,
                  'query_statistics_completed': 16, 'query_primitives_emitted_result': 20}}
# Clear submissions can be deferred into a later draw's initial attachment.
# Only four of the producer probe's twenty clears run as standalone reports;
# all sixteen native draw reports must still be present and complete.
EXPECTED_MODEL_COMMANDS = {'vertex_fetch': {'draw_pco_sequence': 88},
                           'producers': {'draw_pco_sequence': 16, 'clear_color': 4}}
LIBRARIES = {}
for backend, prefix in PREFIXES.items():
    if not (prefix / 'include/EGL/egl.h').is_file() or not (prefix / 'include/GLES3/gl31.h').is_file():
        parser.error('Missing EGL/GLES3 development headers in ' + str(prefix))
    library_dir = prefix / 'lib'
    names = ('libEGL.1.dylib', 'libGLESv2.2.dylib') if sys.platform == 'darwin' else (
        'libEGL.so.1', 'libGLESv2.so.2')
    candidates = list(library_dir.glob('libgallium*.dylib' if sys.platform == 'darwin' else 'libgallium*.so*'))
    # Versioned symlinks may name the same library; hash the real payload once.
    gallium = {path.resolve() for path in candidates if path.is_file()}
    if len(gallium) != 1:
        parser.error('Expected exactly one installed Gallium payload in ' + str(library_dir))
    LIBRARIES[backend] = [library_dir / name for name in names] + [next(iter(gallium))]
    if backend == 'pvrgpu':
        LIBRARIES[backend].append(BRIDGE)
    for path in LIBRARIES[backend]:
        if not path.is_file():
            parser.error('Missing runtime library: ' + str(path))
BASE.mkdir(parents=True, exist_ok=False)


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def hashes(backend):
    return {str(path): sha(path) for path in LIBRARIES[backend]}


def loaded_library_records(stderr, platform):
    """Do not treat LD_DEBUG search paths/trying-file lines as loaded code."""
    if platform == 'darwin':
        pattern = re.compile(r'^dyld\[\d+\]: <[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}> (/.+)$')
    elif platform == 'linux':
        pattern = re.compile(r'^\s*\d+:\s+calling init:\s+(/.+?)\s*$')
    else:
        raise ValueError('Unsupported dynamic-loader record format: ' + platform)
    return [{'line': number, 'path': match.group(1)}
            for number, line in enumerate(stderr.splitlines(), 1)
            if (match := pattern.fullmatch(line))]


def natural(value):
    return type(value) is int and value >= 0


def parse_driver(text):
    events, compute = Counter(), Counter()
    unsupported, errors = [], []
    pending = {'systemc_api': None, 'compute_api': None}
    compute_keys = ('workgroups', 'invocations', 'alu_instructions', 'memory_instructions',
                    'atomic_instructions', 'load_instructions', 'store_instructions',
                    'dram_read_bytes', 'dram_write_bytes', 'direct_read_bytes',
                    'direct_write_bytes', 'readback_bytes', 'pool_allocations', 'pool_releases')
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = dict(re.findall(r'\b([a-zA-Z0-9_]+)=([^\s]+)', line))
        if (fields.get('schema') != 'pvrgpu.driver-counter.v1' or
                fields.get('producer') != 'pvrgpu-gallium-driver' or not fields.get('event')):
            errors.append({'line': number, 'reason': 'unexpected driver schema', 'text': line})
            continue
        event = fields['event']
        events[event] += 1
        if 'unsupported' in event:
            unsupported.append(line)
        if re.search('error|failed|failure', event):
            errors.append({'line': number, 'reason': 'driver diagnostic', 'text': line})
        for api in pending:
            if event == api + '_submit':
                if pending[api] is not None:
                    errors.append({'line': number, 'reason': 'overlapping uncompleted submission', 'api': api})
                pending[api] = fields
            elif event == api + '_done':
                before = pending[api]
                if before is None:
                    errors.append({'line': number, 'reason': 'completion without submission', 'api': api})
                elif api == 'systemc_api' and any(before.get(key) != fields.get(key) for key in ('command', 'case')):
                    errors.append({'line': number, 'reason': 'completion command identity mismatch', 'api': api})
                pending[api] = None
        if event == 'compute_api_done':
            for key in compute_keys:
                value = fields.get(key, '')
                if not re.fullmatch(r'[0-9]+', value):
                    errors.append({'line': number, 'reason': 'missing or invalid compute counter', 'field': key})
                else:
                    compute[key] += int(value)
            if fields.get('pool_allocations') != fields.get('pool_releases'):
                errors.append({'line': number, 'reason': 'unbalanced compute ownership'})
    for api, submission in pending.items():
        if submission is not None or events[api + '_submit'] != events[api + '_done']:
            errors.append({'reason': 'incomplete native submissions', 'api': api,
                           'submit': events[api + '_submit'], 'done': events[api + '_done']})
    return {'event_counts': dict(events), 'compute_counters': dict(compute),
            'unsupported_events': unsupported, 'driver_errors': errors}


def parse_model(text, stderr, expected_commands):
    """Validate each hello/counter/done session and nested ownership fields."""
    errors, diagnostics, types, commands, totals = [], [], Counter(), Counter(), Counter()
    session = None
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if re.fullmatch(r'@CAPTURE: [A-Za-z0-9_-]+ sample=[0-9]+ png=[A-Za-z0-9_.-]+\.png', line):
            if session is None or session['counter_count'] != 1:
                errors.append({'line': number, 'reason': 'capture outside completed counter session'})
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append({'line': number, 'reason': 'invalid JSONL', 'detail': str(error), 'text': line})
            continue
        if not isinstance(row, dict):
            errors.append({'line': number, 'reason': 'JSONL record is not an object', 'record': row})
            continue
        kind = row.get('type')
        if (str(kind).lower() in ('error', 'fatal') or
                str(row.get('severity', '')).lower() in ('error', 'fatal') or row.get('error') or row.get('fatal')):
            errors.append({'line': number, 'reason': 'model diagnostic', 'record': row})
        if (row.get('protocol') != 'pvrgpu-jsonl' or type(row.get('version')) is not int or
                row['version'] != 1 or row.get('schema') != 'pvrgpu.counter.v1' or row.get('backend') != 'pvrgpu'):
            errors.append({'line': number, 'reason': 'unexpected model schema', 'record': row})
            continue
        if kind not in ('hello', 'counter', 'done'):
            errors.append({'line': number, 'reason': 'unexpected model record type', 'record': row})
            continue
        types[kind] += 1
        for key in ('pool_leaks', 'pool_bytes_in_flight'):
            if key in row and (not natural(row[key]) or row[key] != 0):
                errors.append({'line': number, 'reason': 'model ownership is not empty', 'field': key})
        if kind in ('hello', 'counter'):
            if (row.get('command_source') != 'pvrgpu-gallium-driver-command' or
                    row.get('driver_command_ingest') is not True or
                    row.get('driver_command_schema') != 'pvrgpu.driver-command.v1' or
                    row.get('driver_command_producer') != 'pvrgpu-gallium-driver'):
                errors.append({'line': number, 'reason': 'unexpected model command provenance'})
        if kind == 'hello':
            if session is not None:
                errors.append({'line': number, 'reason': 'new hello before prior completion'})
            command = row.get('driver_command')
            if (not isinstance(command, str) or command not in expected_commands or
                    not natural(row.get('frames')) or row['frames'] != 1):
                errors.append({'line': number, 'reason': 'unexpected model command or frame count'})
            session = {'command': command, 'case': row.get('driver_command_case'), 'counter_count': 0}
        elif kind == 'counter':
            command = row.get('driver_command')
            if not isinstance(command, str):
                errors.append({'line': number, 'reason': 'invalid native counter command'})
                command = '<invalid>'
            commands[command] += 1
            if session is None:
                errors.append({'line': number, 'reason': 'counter without hello'})
            else:
                session['counter_count'] += 1
                if (session['counter_count'] != 1 or command != session['command'] or
                        row.get('driver_command_case') != session['case']):
                    errors.append({'line': number, 'reason': 'counter session identity/count mismatch'})
            if (row.get('source') != 'pvrgpu-systemc' or row.get('provenance') != 'modeled' or
                    not natural(row.get('frame')) or row['frame'] != 1):
                errors.append({'line': number, 'reason': 'unexpected native counter producer/frame'})
            counters = row.get('counters')
            if not isinstance(counters, dict):
                errors.append({'line': number, 'reason': 'missing nested counters'})
                continue
            for key, value in counters.items():
                if not natural(value):
                    errors.append({'line': number, 'reason': 'counter is not a nonnegative integer', 'field': key})
                else:
                    totals[key] += value
            for key in ('vs_invocations', 'ia_vertices', 'ia_primitives', 'ps_invocations',
                        'cs_invocations', 'pool_bytes_in_flight'):
                if key not in counters:
                    errors.append({'line': number, 'reason': 'missing required native counter', 'field': key})
            if counters.get('pool_bytes_in_flight') != 0 or counters.get('pool_leaks', 0) != 0:
                errors.append({'line': number, 'reason': 'nested ownership is not empty'})
            if any(key in counters for key in ('pool_allocations', 'pool_releases')) and (
                    not natural(counters.get('pool_allocations')) or
                    counters.get('pool_allocations') != counters.get('pool_releases')):
                errors.append({'line': number, 'reason': 'nested pool ownership is unbalanced'})
            if command == 'draw_pco_sequence':
                for key in ('vs_invocations', 'ia_vertices', 'ia_primitives'):
                    if not natural(counters.get(key)) or counters[key] == 0:
                        errors.append({'line': number, 'reason': 'missing executed native draw', 'field': key})
        else:
            if (session is None or session['counter_count'] != 1 or
                    not natural(row.get('frames')) or row['frames'] != 1):
                errors.append({'line': number, 'reason': 'incomplete model session'})
            if session and session['command'] == 'draw_pco_sequence' and row.get('physical_submissions') != 1:
                errors.append({'line': number, 'reason': 'missing physical draw completion'})
            for key in ('pool_allocations', 'pool_releases', 'pool_leaks', 'pool_bytes_in_flight'):
                if not natural(row.get(key)):
                    errors.append({'line': number, 'reason': 'missing completion ownership field', 'field': key})
            if row.get('pool_allocations') != row.get('pool_releases'):
                errors.append({'line': number, 'reason': 'unbalanced model ownership'})
            session = None
    expected_sessions = sum(expected_commands.values())
    if session is not None or any(types[kind] != expected_sessions for kind in ('hello', 'counter', 'done')):
        errors.append({'reason': 'incomplete model session inventory', 'expected': expected_sessions, 'actual': dict(types)})
    if dict(commands) != expected_commands:
        errors.append({'reason': 'incomplete native model command inventory', 'expected': expected_commands,
                       'actual': dict(commands)})
    for number, line in enumerate(stderr.splitlines(), 1):
        if not line.strip():
            continue
        diagnostic = {'line': number, 'text': line}
        diagnostics.append(diagnostic)
        if re.search(r'(?i)\b(error|fatal|exception|assertion|failed|failure|incomplete|unsupported|sanitizer|runtime error)\b', line):
            errors.append({'reason': 'model stderr diagnostic', **diagnostic})
        elif not re.fullmatch(r'\s*(?:Info: .+|SystemC [0-9].*|Copyright \(c\).*|ALL RIGHTS RESERVED.*)\s*', line):
            errors.append({'reason': 'unrecognized model stderr output', **diagnostic})
    return {'model_errors': errors, 'model_stderr_diagnostics': diagnostics, 'model_record_types': dict(types),
            'model_command_counts': dict(commands), 'model_counters': dict(totals)}


def environment(backend, out):
    prefix = PREFIXES[backend]
    configured = {'LIBGL_DRIVERS_PATH': str(prefix / 'lib/dri'), 'GALLIUM_DRIVER': backend,
        'MESA_LOADER_DRIVER_OVERRIDE': 'swrast', 'EGL_PLATFORM': 'surfaceless',
        'LIBGL_ALWAYS_SOFTWARE': 'true', 'MESA_SHADER_CACHE_DISABLE': 'true'}
    if sys.platform == 'darwin':
        configured.update({'DYLD_LIBRARY_PATH': str(prefix / 'lib'),
            'DYLD_FALLBACK_LIBRARY_PATH': str(prefix / 'lib'), 'DYLD_PRINT_LIBRARIES': '1'})
    else:
        configured.update({'LD_LIBRARY_PATH': str(prefix / 'lib'), 'LD_DEBUG': 'libs'})
    if backend == 'pvrgpu':
        configured.update({'PVRGPU_MODEL_MEMORY_MODE': 'cache', 'PVRGPU_DEQP_LIVE': '1',
            'PVRGPU_SYSTEMC_API_LIB': str(BRIDGE), 'PVRGPU_DEQP_OUTPUT_ROOT': str(out),
            'PVRGPU_DRIVER_COMMAND_OUT': str(out / 'driver-command.txt'),
            'PVRGPU_DRIVER_COUNTER_OUT': str(out / 'driver-counter.txt'),
            'PVRGPU_SYSTEMC_JSONL_OUT': str(out / 'systemc.jsonl'),
            'PVRGPU_SYSTEMC_STDERR_OUT': str(out / 'systemc.stderr.log'),
            'PVRGPU_SYSTEMC_OUTDIR': str(out / 'systemc')})
    env = {k: v for k, v in os.environ.items() if not k.startswith(
        ('PVRGPU_', 'PCO_', 'MESA_', 'GALLIUM_', 'DYLD_', 'LD_', 'LIBGL_', 'EGL_'))}
    env.update(configured)
    return env, configured


root = {'started_utc': stamp(), 'argv': sys.argv, 'runner_sha256': sha(RUNNER),
        'source_sha256': {str(SOURCES[name]): sha(SOURCES[name]) for name in args.probes},
        'runtime_sha256_before': {name: hashes(name) for name in PREFIXES},
        'compiler': COMPILER, 'compiler_sha256': sha(Path(COMPILER[0])),
        'compiler_version': subprocess.check_output([*COMPILER, '--version'], text=True),
        'expected_totals_per_backend': {'scenarios': 108, 'query_records': 108, 'dword_records': 16640},
        'git_head': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()}
write_json(BASE / 'runtime-receipt.json', root)
overall_start = time.monotonic()
all_passed, probe_results = True, {}
for name in args.probes:
    out = BASE / name
    out.mkdir()
    executable = out / 'probe'
    prefix = PREFIXES['pvrgpu']
    command = [*COMPILER, '-std=c11', '-Wall', '-Wextra', '-Werror', '-O2',
        '-I' + str(prefix / 'include'), str(SOURCES[name]), '-L' + str(prefix / 'lib'),
        '-lEGL', '-lGLESv2', '-o', str(executable)]
    build = {'started_utc': stamp(), 'argv': command, 'source_sha256': sha(SOURCES[name])}
    with (out / 'compile.log').open('wb') as log:
        build['exit_code'] = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120).returncode
    build['finished_utc'] = stamp()
    write_json(out / 'compile.json', build)
    assert build['exit_code'] == 0, 'Probe compile failed; see ' + str(out / 'compile.log')
    build['executable_sha256'] = sha(executable)
    write_json(out / 'compile.json', build)
    results = {}
    for backend in ('llvmpipe', 'pvrgpu'):
        directory = out / backend
        directory.mkdir()
        (directory / 'systemc').mkdir()
        env, configured = environment(backend, directory)
        receipt = {'started_utc': stamp(), 'argv': [str(executable)], 'environment': configured,
                   'runtime_sha256_before': hashes(backend), 'executable_sha256': sha(executable)}
        assert receipt['runtime_sha256_before'] == root['runtime_sha256_before'][backend]
        write_json(directory / 'command.json', receipt)
        start = time.monotonic()
        with (directory / 'stdout.txt').open('wb') as stdout, (directory / 'stderr.log').open('wb') as stderr:
            try:
                code = subprocess.run([str(executable)], cwd=directory, env=env,
                    stdout=stdout, stderr=stderr, timeout=args.timeout).returncode
                timed_out = False
            except subprocess.TimeoutExpired:
                code, timed_out = 124, True
        output = (directory / 'stdout.txt').read_text(errors='replace')
        stderr = (directory / 'stderr.log').read_text(errors='replace')
        counter_file = directory / 'driver-counter.txt'
        counters = counter_file.read_text(errors='replace') if counter_file.exists() else ''
        driver_evidence = parse_driver(counters)
        events = driver_evidence['event_counts']
        unsupported = driver_evidence['unsupported_events']
        errors = driver_evidence['driver_errors']
        model_file = directory / 'systemc.jsonl'
        model_stderr_file = directory / 'systemc.stderr.log'
        model_text = model_file.read_text(errors='replace') if model_file.exists() else ''
        model_stderr = model_stderr_file.read_text(errors='replace') if model_stderr_file.exists() else ''
        model_evidence = parse_model(model_text, model_stderr,
            EXPECTED_MODEL_COMMANDS[name] if backend == 'pvrgpu' else {})
        model_errors = model_evidence['model_errors']
        scenarios, words = EXPECTED[name]
        renderer = re.findall(r'^RENDERER: (.*)$', stderr, re.M)
        expected_libraries = [str(path) for path in LIBRARIES[backend]]
        load_records = loaded_library_records(stderr, sys.platform)
        actual_paths = {str(Path(record['path']).resolve()) for record in load_records}
        loaded_libraries = [path for path in expected_libraries if str(Path(path).resolve()) in actual_paths]
        relevant_load_records = [record for record in load_records if str(Path(record['path']).resolve()) in
                                 {str(Path(path).resolve()) for path in expected_libraries}]
        correct_renderer = len(renderer) == 1 and (
            renderer[0] == 'PvrGPU SystemC Gallium bring-up' if backend == 'pvrgpu'
            else renderer[0].startswith('llvmpipe ('))
        queries = len(re.findall(r'^QUERY ', output, re.M))
        output_words = len(re.findall(r'^WORD ', output, re.M))
        native_counts = EXPECTED_NATIVE_EVENTS[name] if backend == 'pvrgpu' else {}
        native_evidence_complete = all(events.get(event, 0) == count for event, count in native_counts.items())
        receipt.update(finished_utc=stamp(), wall_seconds=round(time.monotonic() - start, 3),
            exit_code=code, timed_out=timed_out, renderer=renderer,
            expected_loaded_libraries=expected_libraries, verified_loaded_libraries=loaded_libraries,
            verified_loader_records=relevant_load_records, loader_record_count=len(load_records),
            query_records=queries, dword_records=output_words, expected_scenarios=scenarios,
            expected_dword_records=words, stdout_sha256=sha(directory / 'stdout.txt'),
            **driver_evidence, **model_evidence, expected_native_events=native_counts,
            native_evidence_complete=native_evidence_complete,
            runtime_sha256_after=hashes(backend))
        receipt['runtime_unchanged'] = receipt['runtime_sha256_before'] == receipt['runtime_sha256_after']
        receipt['passed'] = bool(code == 0 and not timed_out and correct_renderer and
            queries == scenarios and output_words == words and not unsupported and not errors and
            not model_errors and receipt['runtime_unchanged'] and
            loaded_libraries == expected_libraries and native_evidence_complete and
            'PASS: scenarios=' + str(scenarios) + ' checked_words=' + str(words) in stderr)
        write_json(directory / 'result.json', receipt)
        results[backend] = receipt
        print(json.dumps({'probe': name, 'backend': backend, 'passed': receipt['passed'],
            'code': code, 'queries': queries, 'words': output_words, 'seconds': receipt['wall_seconds']}), flush=True)
        assert receipt['runtime_unchanged']
    left = (out / 'llvmpipe/stdout.txt').read_text().splitlines(keepends=True)
    right = (out / 'pvrgpu/stdout.txt').read_text().splitlines(keepends=True)
    diff = ''.join(difflib.unified_diff(left, right, fromfile='llvmpipe/stdout.txt', tofile='pvrgpu/stdout.txt'))
    (out / 'stdout.diff').write_text(diff)
    transcripts, extra_lines, valid_extras = {}, {}, True
    for backend, lines in (('llvmpipe', left), ('pvrgpu', right)):
        transcripts[backend] = [line for line in lines if line.startswith(('QUERY ', 'WORD '))]
        extra_lines[backend] = [line.rstrip('\n') for line in lines if not line.startswith(('QUERY ', 'WORD '))]
        allowed = ('', 'Info: /OSCI/SystemC: Simulation stopped by user.') if backend == 'pvrgpu' else ('',)
        valid_extras = valid_extras and all(line in allowed for line in extra_lines[backend])
        (out / backend / 'native-transcript.txt').write_text(''.join(transcripts[backend]))
    transcript_diff = ''.join(difflib.unified_diff(transcripts['llvmpipe'], transcripts['pvrgpu'],
        fromfile='llvmpipe/native-transcript.txt', tofile='pvrgpu/native-transcript.txt'))
    (out / 'native-transcript.diff').write_text(transcript_diff)
    comparison = {'stdout_identical': left == right,
                  'native_transcript_identical': transcripts['llvmpipe'] == transcripts['pvrgpu'],
                  'native_transcript_sha256': {backend: sha(out / backend / 'native-transcript.txt')
                                              for backend in PREFIXES},
                  'stdout_extra_lines': extra_lines, 'stdout_extras_are_known_runtime_messages': valid_extras,
                  'both_passed': all(r['passed'] for r in results.values()), 'results': results}
    write_json(out / 'comparison.json', comparison)
    probe_results[name] = comparison
    all_passed = all_passed and comparison['native_transcript_identical'] and valid_extras and comparison['both_passed']
totals = {backend: {'query_records': sum(probe_results[name]['results'][backend]['query_records'] for name in args.probes),
                    'dword_records': sum(probe_results[name]['results'][backend]['dword_records'] for name in args.probes)}
          for backend in PREFIXES}
source_hashes_after = {str(SOURCES[name]): sha(SOURCES[name]) for name in args.probes}
sources_unchanged = source_hashes_after == root['source_sha256']
runner_unchanged = sha(RUNNER) == root['runner_sha256']
all_passed = all_passed and sources_unchanged and runner_unchanged and all(
    values == {'query_records': 108, 'dword_records': 16640} for values in totals.values())
root.update(finished_utc=stamp(), wall_seconds=round(time.monotonic() - overall_start, 3),
            runtime_sha256_after={name: hashes(name) for name in PREFIXES},
            totals_per_backend=totals, source_sha256_after=source_hashes_after,
            sources_unchanged=sources_unchanged, runner_unchanged=runner_unchanged,
            all_passed=all_passed, probes=probe_results)
root['runtime_unchanged'] = root['runtime_sha256_before'] == root['runtime_sha256_after']
write_json(BASE / 'runtime-receipt.json', root)
print(json.dumps({'finished': str(BASE / 'runtime-receipt.json'), 'all_passed': all_passed,
                  'runtime_unchanged': root['runtime_unchanged']}), flush=True)
assert root['runtime_unchanged']
sys.exit(0 if all_passed else 1)
PYTHON
