"""Fresh-process entcalc benchmarks. Run one of the three launcher scripts."""
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import time
import traceback


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def worker(connection, config, state_index, method):
    """Imports, state loading and RNG setup happen before the timed interval."""
    try:
        import numpy as np
        import qutip
        import psutil
        import entcalcpy as en

        dim = config['dim']
        states = np.load(config['states_path'], mmap_mode='r')
        rho = qutip.Qobj(np.array(states[state_index], copy=True), dims=[dim, dim])
        del states
        seed = config['algorithm_seed'] + state_index
        np.random.seed(seed)
        random.seed(seed)
        kwargs = dict(sdpaccuracy=config['accuracy'], solversdp=config['solver'])
        if method == 'upper':
            function, args = en.upperbip, (rho, dim.copy())
            kwargs = dict(iteramax=config['upper_iteramax'])
        elif method in ('kppt2', 'kppt3'):
            function, args = en.gekppt, (rho, dim.copy(), int(method[-1]))
        else:
            function = {'sm': en.ge_mixed_sm, 'gr': en.ge_mixed_gr,
                        'ppt': en.geppt}[method]
            args = (rho, dim.copy())
        process = psutil.Process(os.getpid())
        connection.send({'kind': 'ready', 'baseline': process.memory_info().rss})
        if connection.recv() != 'go':
            raise RuntimeError('Invalid start signal')
        start = time.perf_counter()
        result = function(*args, **kwargs)
        elapsed = time.perf_counter() - start
        end_rss = process.memory_info().rss
        value = float(result if method == 'upper' else result[0])
        if not np.isfinite(value):
            raise ValueError(f'Non-finite result: {value}')
        connection.send({'kind': 'done', 'result': value, 'time': elapsed,
                         'end_rss': end_rss, 'pid': os.getpid(),
                         'accuracy_status': None if method == 'upper' else str(result[1])})
        connection.recv()  # Keep result alive until the parent finishes sampling.
    except BaseException:
        try:
            connection.send({'kind': 'error', 'traceback': traceback.format_exc()})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def run_one(config, index, method):
    import psutil
    context = mp.get_context('spawn')
    parent, child = context.Pipe()
    process = context.Process(target=worker, args=(child, config, index, method))
    process.start()
    child.close()
    baseline = peak = None
    observed = psutil.Process(process.pid)
    try:
        while True:
            if baseline is not None:
                try:
                    peak = max(peak, observed.memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
            if parent.poll(config['sample_interval']):
                try:
                    message = parent.recv()
                except EOFError:
                    process.join()
                    raise RuntimeError(f'Worker exited without a result (exit code {process.exitcode}).')
                if message['kind'] == 'ready':
                    baseline = peak = message['baseline']
                    parent.send('go')
                elif message['kind'] == 'error':
                    raise RuntimeError(message['traceback'])
                elif message['kind'] == 'done':
                    peak = max(peak, message.pop('end_rss'))
                    message.update(peak=peak / 1024**2,
                                   memory=(peak-baseline) / 1024**2,
                                   baseline=baseline / 1024**2)
                    message.pop('kind')
                    parent.send('ack')
                    process.join(10)
                    if process.is_alive() or process.exitcode != 0:
                        raise RuntimeError('Worker did not exit cleanly.')
                    return message
            elif not process.is_alive():
                raise RuntimeError(f'Worker exited (exit code {process.exitcode}).')
    finally:
        if process.is_alive():
            process.terminate()
        process.join()
        parent.close()


def export_results(directory, prefix, methods, records):
    """Rebuild text files from committed records, in original state order."""
    for method in methods:
        rows = sorted((r for r in records if r['method'] == method),
                      key=lambda r: r['state_index'])
        for field in ('result', 'time', 'peak', 'memory', 'baseline'):
            name = prefix + method + field + '.txt'
            if method == 'upper' and field == 'result':
                name = prefix + 'upper.txt'
            atomic_text(directory / name, ''.join(f'{r[field]:.17g}\n' for r in rows))


def run_benchmark(*, dim, prefix, n=50, run_gr=True, run_k=True,
                  run_upper=False, output_dir, state_file=None,
                  seed_start=1000, algorithm_seed=2000, upper_iteramax=5000,
                  solver='MOSEK', accuracy='high', sample_interval=0.01):
    # Set before importing numerical libraries and before spawning workers.
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    import numpy as np
    import qutip
    import entcalcpy as en
    if n < 1 or sample_interval <= 0:
        raise ValueError('n and sample_interval must be positive')
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    methods = ['sm'] + (['gr'] if run_gr else []) + ['ppt', 'kppt2']
    methods += ['kppt3'] if run_k else []
    methods += ['upper'] if run_upper else []
    if run_upper and len(dim) != 2:
        raise ValueError('upperbip requires two subsystems')
    states_path = directory / (prefix + 'states.npy')
    checkpoint = directory / (prefix + 'checkpoint.json')
    versions = {}
    for package in ('numpy', 'scipy', 'qutip', 'cvxpy', 'psutil', 'Mosek'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'not installed'
    config = dict(dim=list(dim), n=n, methods=methods, prefix=prefix,
                  solver=solver, accuracy=accuracy, seed_start=seed_start,
                  algorithm_seed=algorithm_seed, upper_iteramax=upper_iteramax,
                  sample_interval=sample_interval, threads=1,
                  python=platform.python_version(), platform=platform.platform(),
                  versions=versions,
                  entcalcpy_sha256=hashlib.sha256(Path(en.__file__).read_bytes()).hexdigest(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    # Existing checkpoints are authoritative; changing the experiment needs a new folder.
    saved = json.loads(checkpoint.read_text()) if checkpoint.exists() else None
    if saved and saved['config'] != config:
        raise RuntimeError('Settings/code/environment changed. Use a new OUTPUT_DIR.')
    if saved and not states_path.exists():
        raise RuntimeError('Saved state file is missing; restore it before resuming.')
    if not states_path.exists():
        if state_file is not None:
            states = np.load(Path(state_file), allow_pickle=False)
        else:
            states = np.array([qutip.rand_dm(dim, distribution='hs',
                                           seed=seed_start+i).full() for i in range(n)])
        expected = (n, int(np.prod(dim)), int(np.prod(dim)))
        if states.shape != expected:
            raise ValueError(f'State array must have shape {expected}, got {states.shape}')
        temporary = states_path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.save(stream, states)
        os.replace(temporary, states_path)
    states = np.load(states_path, allow_pickle=False)
    if states.shape != (n, int(np.prod(dim)), int(np.prod(dim))):
        raise ValueError('State file shape does not match settings')
    state_hash = hashlib.sha256(states_path.read_bytes()).hexdigest()
    if saved and saved['states_sha256'] != state_hash:
        raise RuntimeError('Saved states have changed; refusing to mix results.')
    # Text copy of the exact matrices, including complex entries.
    text_path = directory / (prefix + 'states.txt')
    if not text_path.exists():
        with text_path.open('w', encoding='utf-8') as stream:
            for i, state in enumerate(states):
                stream.write(f'STATE {i+1}\n')
                np.savetxt(stream, state, fmt='%.16e')
                stream.write('\n')
    del states
    saved = saved or dict(config=config, states_sha256=state_hash, records=[])
    atomic_text(checkpoint, json.dumps(saved, indent=2, allow_nan=False))
    export_results(directory, prefix, methods, saved['records'])
    completed = {(r['state_index'], r['method']) for r in saved['records']}
    config = dict(config, states_path=str(states_path))
    for i in range(n):
        for method in methods:
            if (i, method) in completed:
                continue
            print(f'State {i+1}/{n}, method {method}: starting fresh process', flush=True)
            try:
                record = run_one(config, i, method)
            except BaseException:
                atomic_text(directory / (prefix + 'last_error.txt'), traceback.format_exc())
                print('Stopped. Completed results are saved; rerun to retry this calculation.', flush=True)
                raise
            record.update(state_index=i, method=method)
            saved['records'].append(record)
            atomic_text(checkpoint, json.dumps(saved, indent=2, allow_nan=False))
            export_results(directory, prefix, methods, saved['records'])
            print(f"  result={record['result']:.9g}, time={record['time']:.3f} s, "
                  f"peak={record['peak']:.2f} MiB, memory={record['memory']:.2f} MiB, "
                  f"status={record['accuracy_status']}", flush=True)
    print(f'Completed. Results: {directory}', flush=True)
