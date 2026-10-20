"""Access to the unmodified FDEdge code in third_party/fdedge.

Nothing here edits the upstream files. Changes for this study are subclasses in environment.py and agents.py.
"""
from functools import cache
import importlib
import importlib.util
import sys

from .config import UPSTREAM

COMMIT = '551988866f934b5cdb672e9122279a73eba60259'


@cache
def fdedge(name):
    """One top-level FDEdge module: 'edge_environment', 'fdsac_model', 'feedback_diffusion' or 'helpers'.

    These files import each other by bare module name, so their folder is on sys.path only while they load.
    """
    sys.path.insert(0, str(UPSTREAM))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(UPSTREAM))


@cache
def baseline(actor, kind):
    """The public SAC or DQN baseline: kind 'environment' or 'model'."""
    path = UPSTREAM / 'baselines' / actor.upper() / f'{actor}_{kind}.py'
    spec = importlib.util.spec_from_file_location(f'fdedge_{actor}_{kind}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
