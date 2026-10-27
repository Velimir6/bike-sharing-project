"""Execute the notebook in a fresh kernel using this Python interpreter.

The temporary kernel specification stays outside the user's Jupyter settings.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os
import sys
import time
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]


def main():
    notebook = nbformat.read(ROOT / 'bike_demand.ipynb', as_version=4)
    started = time.perf_counter()
    runtime = (ROOT / '.runtime').resolve()
    assert runtime.is_relative_to(ROOT.resolve())
    runtime.mkdir(exist_ok=True)
    os.environ['IPYTHONDIR'] = str(runtime / 'ipython')
    os.environ['JUPYTER_RUNTIME_DIR'] = str(runtime / 'jupyter')
    os.environ['MPLCONFIGDIR'] = str(runtime / 'matplotlib')
    with TemporaryDirectory(prefix='kernel-', dir=runtime) as temp:
        kernels = Path(temp)
        spec = kernels / 'python3'
        spec.mkdir()
        (spec / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Python 3', 'language': 'python'}), encoding='utf-8')
        manager = KernelManager(kernel_name='python3', connection_file=str(kernels / 'connection.json'),
                                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
        client = NotebookClient(notebook, km=manager, timeout=600, startup_timeout=60,
                                resources={'metadata': {'path': str(ROOT)}})
        client.on_cell_start = lambda cell, cell_index: print(f'Checking cell {cell_index + 1}/{len(notebook.cells)}', flush=True)
        print('Starting a fresh notebook kernel...', flush=True)
        client.execute()
    nbformat.validate(notebook)
    errors = [out for cell in notebook.cells if cell.cell_type == 'code'
              for out in cell.outputs if out.output_type == 'error']
    assert not errors
    nbformat.write(notebook, ROOT / 'bike_demand.ipynb')
    exporter = HTMLExporter(template_name='lab')
    body, _ = exporter.from_notebook_node(notebook)
    (ROOT / 'bike_demand.html').write_text(body, encoding='utf-8')
    report = {'success': True, 'code_cells': sum(c.cell_type == 'code' for c in notebook.cells),
              'error_outputs': len(errors), 'wall_seconds': round(time.perf_counter() - started, 2),
              'kernel': 'fresh process using the executing Python interpreter'}
    (ROOT / 'results' / 'execution_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
