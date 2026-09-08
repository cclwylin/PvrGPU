#!/usr/bin/env python3
"""Run exact stock dEQP GLES3 Transform Feedback cases on two fixed backends."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
import argparse
import csv
import datetime
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--work-root', type=Path, default=Path(os.environ.get(
    'PVRGPU_WORK_ROOT', str(Path.home()/'Downloads/_Codex/Working/PvrGPU'))))
parser.add_argument('--output-root', type=Path, help='Immutable discovery and backend run directories')
parser.add_argument('--cts-binary', type=Path, help='Stock deqp-gles3 executable')
parser.add_argument('--bridge', type=Path, help='SystemC bridge shared library')
parser.add_argument('--pvrgpu-prefix', type=Path, help='Installed PvrGPU Mesa prefix')
parser.add_argument('--llvmpipe-prefix', type=Path, help='Installed llvmpipe Mesa prefix')
parser.add_argument('--reference-run', type=Path, help='Existing llvmpipe run containing summary.tsv')
parser.add_argument('--prepare-only', action='store_true')
parser.add_argument('--output-label', default='baseline')
parser.add_argument('--backends', nargs='+', choices=['pvrgpu', 'llvmpipe'],
                    default=['pvrgpu', 'llvmpipe'])
parser.add_argument('--case-filter', action='append', default=[])
parser.add_argument('--workers', type=int, choices=range(1, 4), default=3)
parser.add_argument('--debug-pco-nir', action='store_true')
args = parser.parse_args()
if not re.fullmatch('[a-z0-9_-]+', args.output_label):
    parser.error('--output-label must contain only lowercase letters, digits, underscore or hyphen')
if len(args.backends) != len(set(args.backends)):
    parser.error('--backends must not contain duplicates')
WORK = args.work_root.expanduser().resolve()
BASE = (args.output_root or WORK/'out/runs/deqp_groups/transform_feedback').expanduser().resolve()
BASE.mkdir(parents=True, exist_ok=True)
candidates = [WORK/'build/deqp-mesa/build/modules/gles3/deqp-gles3',
              WORK.parent/'build/deqp-mesa/build/modules/gles3/deqp-gles3']
BINARY = (args.cts_binary or next((p for p in candidates if p.is_file()), candidates[0])).expanduser().resolve()
BRIDGE = (args.bridge or WORK/'build/lib/libpvrgpu_systemc_bridge.dylib').expanduser().resolve()
PREFIXES = {
    'pvrgpu': (args.pvrgpu_prefix or WORK/'tmp/pvrgpu-mesa-install').expanduser().resolve(),
    'llvmpipe': (args.llvmpipe_prefix or WORK/'mesa-counter/install').expanduser().resolve(),
}
PREFIX = 'dEQP-GLES3.functional.transform_feedback.'
EXACT_CASE = re.compile(r'^dEQP-GLES3\.[A-Za-z0-9_.-]+$')

def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def json_write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')

def write_new(path, value):
    if path.exists():
        if path.read_text() != value:
            raise RuntimeError('Existing immutable artifact differs: '+str(path))
    else:
        path.write_text(value)

def hashes(backend):
    prefix = PREFIXES[backend]
    paths = [BINARY, prefix/'lib/libEGL.1.dylib', prefix/'lib/libGLESv2.2.dylib',
             prefix/'lib/libgallium-26.2.1.dylib'] + ([BRIDGE] if backend == 'pvrgpu' else [])
    return {str(path): sha(path) for path in paths}

def receipt_binary_sha(receipt):
    return receipt.get('binary_sha256') or next(
        (value for name, value in receipt['runtime_sha256_before'].items()
         if Path(name).name == BINARY.name), None)

def environment(backend, directory=None):
    prefix = PREFIXES[backend]
    configured = {'DYLD_LIBRARY_PATH': str(prefix/'lib'), 'DYLD_FALLBACK_LIBRARY_PATH': str(prefix/'lib'),
        'LIBGL_DRIVERS_PATH': str(prefix/'lib/dri'), 'GALLIUM_DRIVER': backend,
        'MESA_LOADER_DRIVER_OVERRIDE': 'swrast', 'EGL_PLATFORM': 'surfaceless',
        'LIBGL_ALWAYS_SOFTWARE': 'true', 'MESA_SHADER_CACHE_DISABLE': 'true', 'DYLD_PRINT_LIBRARIES': '1'}
    if backend == 'pvrgpu' and directory is not None:
        configured.update({'PVRGPU_MODEL_MEMORY_MODE': 'cache', 'PVRGPU_DEQP_LIVE': '1',
            'PVRGPU_DEQP_OUTPUT_ROOT': str(directory), 'PVRGPU_SYSTEMC_API_LIB': str(BRIDGE),
            'PVRGPU_DRIVER_COMMAND_OUT': str(directory/'driver-command.txt'),
            'PVRGPU_DRIVER_COUNTER_OUT': str(directory/'driver-counter.txt'),
            'PVRGPU_SYSTEMC_JSONL_OUT': str(directory/'systemc.jsonl'),
            'PVRGPU_SYSTEMC_STDERR_OUT': str(directory/'systemc.stderr.log'),
            'PVRGPU_SYSTEMC_OUTDIR': str(directory/'systemc')})
        if args.debug_pco_nir:
            configured['PVRGPU_DEBUG_PCO_NIR'] = '1'
    env = {k:v for k,v in os.environ.items() if not k.startswith(('PVRGPU_', 'MESA_', 'GALLIUM_', 'DYLD_', 'LIBGL_', 'EGL_'))}
    env.update(configured)
    return env, configured

def discover():
    directory = BASE/'discovery'
    if not directory.exists():
        directory.mkdir()
        env, configured = environment('llvmpipe')
        argv = [str(BINARY), '--deqp-case=dEQP-GLES3.*', '--deqp-archive-dir='+str(BINARY.parent),
            '--deqp-runmode=txt-caselist', '--deqp-caselist-export-file='+str(directory/'all-cases.txt'),
            '--deqp-log-filename='+str(directory/'discovery.qpa'),
            '--deqp-surface-type=pbuffer', '--deqp-log-images=disable']
        receipt = {'argv':argv, 'environment':configured, 'started_utc':stamp(), 'binary_sha256':sha(BINARY), 'runtime_sha256_before':hashes('llvmpipe')}
        with (directory/'run.log').open('wb') as log:
            receipt['exit_code'] = subprocess.run(argv, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600).returncode
        receipt.update(finished_utc=stamp(), runtime_sha256_after=hashes('llvmpipe'))
        receipt['runtime_unchanged'] = receipt['runtime_sha256_before'] == receipt['runtime_sha256_after']
        json_write(directory/'receipt.json', receipt)
        if receipt['exit_code'] != 0 or not receipt['runtime_unchanged']:
            raise RuntimeError('Stock CTS discovery failed or its runtime changed')
    discovery_receipt = json.loads((directory/'receipt.json').read_text())
    discovery_binary_sha = receipt_binary_sha(discovery_receipt)
    if discovery_receipt['exit_code'] != 0 or not discovery_receipt['runtime_unchanged'] or discovery_binary_sha != sha(BINARY):
        raise RuntimeError('Discovery receipt does not match this CTS binary; select a new --output-root')
    discovered = [line.removeprefix('TEST: ') for line in (directory/'all-cases.txt').read_text().splitlines() if line.startswith('TEST: ')]
    if not discovered or len(discovered) != len(set(discovered)):
        raise RuntimeError('Stock CTS discovery has no cases or duplicate names')
    cases = [case for case in discovered if case.startswith(PREFIX)]
    if len(cases) != 1320 or len(set(cases)) != 1320:
        raise RuntimeError('Transform Feedback must contain exactly 1320 unique cases; found '+str(len(cases)))
    if not all(EXACT_CASE.fullmatch(case) for case in cases):
        raise RuntimeError('Transform Feedback discovery contains a non-exact case name')
    write_new(BASE/'cases.txt', '\n'.join(cases)+'\n')
    summary = {'discovered':len(discovered), 'selected':len(cases),
        'groups':dict(Counter(case.removeprefix(PREFIX).split('.')[0] for case in cases)),
        'manifest_sha256':sha(BASE/'cases.txt')}
    write_new(BASE/'scope.json', json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)
    return cases

all_cases = discover()
if args.prepare_only:
    raise SystemExit(0)
cases = [case for case in all_cases if not args.case_filter or any(fnmatch.fnmatchcase(case, pattern) for pattern in args.case_filter)]
if not cases:
    parser.error('--case-filter selected no Transform Feedback cases')
if args.reference_run:
    reference_path = args.reference_run.expanduser().resolve()
    reference_receipt = json.loads((reference_path/'runtime-receipt.json').read_text())
    if (reference_receipt.get('backend') != 'llvmpipe' or
            not reference_receipt.get('runtime_unchanged') or
            reference_receipt.get('runtime_sha256_before') != reference_receipt.get('runtime_sha256_after') or
            reference_receipt.get('manifest_sha256') != sha(BASE/'cases.txt') or
            receipt_binary_sha(reference_receipt) != sha(BINARY)):
        raise RuntimeError('Reference receipt must be an unchanged llvmpipe run of this exact CTS binary and 1320-case manifest')
flags = ['--deqp-surface-type=pbuffer', '--deqp-surface-width=256', '--deqp-surface-height=256',
         '--deqp-gl-config-name=rgba8888d24s8ms0', '--deqp-log-images=disable']
outputs, receipts = {}, {}
for backend in args.backends:
    out = BASE/(backend+'-'+args.output_label)
    out.mkdir(exist_ok=False)
    (out/'cases').mkdir()
    outputs[backend] = out
    receipt = {'backend':backend, 'label':args.output_label, 'case_count':len(cases), 'workers_total':args.workers,
        'case_filters':args.case_filter, 'debug_pco_nir':args.debug_pco_nir,
        'binary':str(BINARY), 'fixed_flags':flags,
        'timeout_seconds_per_case':600, 'manifest_sha256':sha(BASE/'cases.txt'),
        'started_utc':stamp(), 'runtime_sha256_before':hashes(backend),
        'runner':str(Path(__file__).resolve()), 'runner_sha256':sha(Path(__file__)), 'argv':sys.argv}
    receipts[backend] = receipt
    json_write(out/'runtime-receipt.json', receipt)

def audit_case(directory, backend, result):
    """Count explicit model/driver failures separately from stock CTS status."""
    event_counts = result['event_counts']
    driver_errors = {name: count for name, count in event_counts.items()
                     if re.search(r'(?:^|_)(?:error|failed|failure)(?:_|$)', name)}
    model_records = 0
    model_error_records = 0
    pool_leaks = 0
    jsonl = directory/'systemc.jsonl'
    if jsonl.exists():
        for line in jsonl.read_text(errors='replace').splitlines():
            if not line.startswith('{'):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                model_error_records += 1
                continue
            model_records += 1
            model_error_records += record.get('type') in ('error', 'fatal')
            pool_leaks += int(record.get('pool_leaks', 0))
    fatal_lines = []
    for name in ('run.log', 'systemc.stderr.log'):
        path = directory/name
        if path.exists():
            fatal_lines.extend(line for line in path.read_text(errors='replace').splitlines()
                               if 'FATAL ERROR:' in line or 'uncaught exception' in line)
    result['integrity'] = {
        'process_failed': result['exit_code'] != 0 or result['timed_out'],
        'renderer_mismatch': not result['renderer'].startswith('PvrGPU' if backend == 'pvrgpu' else 'llvmpipe'),
        'unsupported_event_count': len(result['unsupported_events']),
        'driver_error_events': driver_errors,
        'model_records': model_records, 'model_error_records': model_error_records,
        'pool_leaks': pool_leaks, 'fatal_lines': fatal_lines,
        'missing_model_records': backend == 'pvrgpu' and result['status'] == 'Pass' and model_records == 0,
    }
    result['integrity']['ok'] = not (
        result['integrity']['process_failed'] or result['integrity']['renderer_mismatch'] or
        result['unsupported_events'] or driver_errors or model_error_records or
        pool_leaks or fatal_lines or result['integrity']['missing_model_records'])

def run_case(backend, case):
    directory = outputs[backend]/'cases'/case
    directory.mkdir()
    (directory/'systemc').mkdir()
    env, configured = environment(backend, directory)
    qpa_path = directory/'results.qpa'
    argv = [str(BINARY), '--deqp-case='+case, '--deqp-archive-dir='+str(BINARY.parent),
            '--deqp-log-filename='+str(qpa_path), *flags]
    json_write(directory/'command.json', {'argv':argv, 'environment':configured})
    start = time.monotonic()
    timed_out = False
    with (directory/'run.log').open('wb') as log:
        try:
            code = subprocess.run(argv, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600).returncode
        except subprocess.TimeoutExpired:
            code, timed_out = 124, True
    qpa = qpa_path.read_text(errors='replace') if qpa_path.exists() else ''
    matches = re.findall(r'<Result\s+StatusCode="([^"]+)"[^>]*>(.*?)</Result>', qpa, re.S)
    status = matches[0][0] if len(matches) == 1 else 'NoResult'
    detail = re.sub(r'\s+', ' ', matches[0][1]).strip() if len(matches) == 1 else 'QPA count='+str(len(matches))
    renderer = re.findall(r'^#sessionInfo renderer "([^"]+)"', qpa, re.M)
    events_path = directory/'driver-counter.txt'
    events = events_path.read_text(errors='replace') if events_path.exists() else ''
    unsupported = [line for line in events.splitlines() if re.search(r'\bevent=[^\s]*unsupported[^\s]*', line)]
    result = {'case':case, 'status':status, 'exit_code':code, 'seconds':round(time.monotonic()-start,3),
        'qpa_result_count':len(matches), 'detail':detail, 'renderer':renderer[0] if len(renderer) == 1 else '',
        'event_counts':dict(Counter(re.findall(r'\bevent=([^\s]+)', events))), 'unsupported_events':unsupported,
        'timed_out':timed_out, 'qpa':str(qpa_path), 'log':str(directory/'run.log')}
    audit_case(directory, backend, result)
    json_write(directory/'result.json',result)
    return backend,result

completed = {backend:{} for backend in args.backends}
tasks = [(backend,case) for case in cases for backend in args.backends]
with ThreadPoolExecutor(max_workers=args.workers) as executor:
    futures = [executor.submit(run_case,*task) for task in tasks]
    for index,future in enumerate(as_completed(futures),1):
        backend,result = future.result()
        completed[backend][result['case']] = result
        if index <= 4 or index % 50 == 0 or index == len(tasks) or result['status'] not in ('Pass', 'NotSupported'):
            print(json.dumps({'completed':index, 'total':len(tasks), 'backend':backend, 'case':result['case'],
                'status':result['status'], 'detail':result['detail'], 'first_unsupported':result['unsupported_events'][:2],
                'counts':{b:dict(Counter(r['status'] for r in rows.values())) for b,rows in completed.items()}}),flush=True)
columns = ['case','status','exit_code','seconds','qpa_result_count','renderer','detail','qpa','log']
for backend,rows in completed.items():
    out = outputs[backend]
    with (out/'summary.tsv').open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=columns,delimiter='\t',lineterminator='\n',extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows[case] for case in cases)
    receipt = receipts[backend]
    receipt.update(finished_utc=stamp(), status_counts=dict(Counter(r['status'] for r in rows.values())),
        runtime_sha256_after=hashes(backend))
    receipt['runtime_unchanged'] = receipt['runtime_sha256_before'] == receipt['runtime_sha256_after']
    json_write(out/'runtime-receipt.json',receipt)
    print(json.dumps({'output':str(out),'counts':receipt['status_counts'],'runtime_unchanged':receipt['runtime_unchanged']}),flush=True)
# Publish comparison independently of QPA success: all selected exact cases must
# have identical status, including the implementation-limit NotSupported set.
reference = None
reference_path = None
if 'llvmpipe' in completed:
    reference = completed['llvmpipe']
    reference_path = outputs['llvmpipe']
elif args.reference_run:
    reference_path = args.reference_run.expanduser().resolve()
    with (reference_path/'summary.tsv').open() as stream:
        reference_rows = list(csv.DictReader(stream, delimiter='\t'))
    reference = {row['case']: row for row in reference_rows}
    if len(reference) != len(reference_rows):
        raise RuntimeError('Reference run contains duplicate exact cases')
comparison = None
if reference is not None and 'pvrgpu' in completed:
    mismatch = [case for case in cases if case not in reference or
                completed['pvrgpu'][case]['status'] != reference[case]['status']]
    comparison = {'pvrgpu_run':str(outputs['pvrgpu']), 'reference_run':str(reference_path),
        'selected_case_count':len(cases), 'exact_status_sets_equal':not mismatch,
        'mismatches':[{'case':case, 'pvrgpu':completed['pvrgpu'][case]['status'],
                      'llvmpipe':reference.get(case, {}).get('status', 'Missing')}
                     for case in mismatch]}
    json_write(BASE/('comparison-'+args.output_label+'.json'), comparison)
    print(json.dumps(comparison), flush=True)
failed = False
for backend, rows in completed.items():
    pass_rows = [row for row in rows.values() if row['status'] == 'Pass']
    integrity = {'pass_case_count':len(pass_rows),
        'process_failure_count':sum(row['integrity']['process_failed'] for row in rows.values()),
        'pass_cases_with_unsupported_events':sum(bool(row['unsupported_events']) for row in pass_rows),
        'pass_cases_with_model_errors':sum(not row['integrity']['ok'] for row in pass_rows),
        'unsupported_event_count':sum(row['integrity']['unsupported_event_count'] for row in pass_rows),
        'model_error_record_count':sum(row['integrity']['model_error_records'] for row in pass_rows),
        'model_record_count':sum(row['integrity']['model_records'] for row in pass_rows),
        'fatal_line_count':sum(len(row['integrity']['fatal_lines']) for row in pass_rows),
        'driver_error_event_count':sum(sum(row['integrity']['driver_error_events'].values()) for row in pass_rows),
        'pool_leak_count':sum(row['integrity']['pool_leaks'] for row in pass_rows),
        'invalid_cases':[row['case'] for row in pass_rows if not row['integrity']['ok']]}
    json_write(outputs[backend]/'integrity-summary.json', integrity)
    print(json.dumps({'backend':backend, 'integrity':integrity}), flush=True)
    failed |= not receipts[backend]['runtime_unchanged']
    failed |= any(row['status'] not in ('Pass', 'NotSupported') for row in rows.values())
    failed |= any(row['exit_code'] != 0 or row['timed_out'] for row in rows.values())
    failed |= bool(integrity['invalid_cases'])
failed |= comparison is not None and not comparison['exact_status_sets_equal']
raise SystemExit(1 if failed else 0)
