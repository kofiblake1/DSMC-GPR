"""Import `Shape` from the upstream `shapes` package without its mesher deps.

`shapes.py` imports pygmsh, meshio and PIL at module level, but only uses them
in code paths this pipeline does not touch:

    shapes.py:319,351   Shape.mesh()        -> pygmsh + meshio (replaced by geo.py)
    shapes.py:604       trim_white()        -> PIL (only for generate_image)
    meshes.py:153,171   read/write_meshio_mesh

We need only Shape.generate(), which is pure numpy/math. Installing the real
packages is not merely unnecessary, it is not possible here: pygmsh 7.x
depends on the gmsh *Python* module, and the gmsh pip wheel cannot load on
this cluster (no libGLU.so.1 anywhere on the system), which is the same reason
geo.py drives the gmsh CLI as a subprocess instead.

So we register minimal placeholder modules for whatever is genuinely missing,
and only for that -- if the real packages ever become importable they are used
unchanged. Any attempt to actually *use* a stubbed module raises with an
explanation rather than failing obscurely.

$SHAPES_DIR is left as a pristine clone of jviquerat/shapes; nothing here
modifies it. `shapes_commit()` records which revision produced a dataset.
"""

import os
import subprocess
import sys
import types

_OPTIONAL = ("pygmsh", "meshio", "PIL")


class _Unavailable(types.ModuleType):
    """Placeholder that explains itself if anything tries to use it."""

    def __getattr__(self, item):
        raise RuntimeError(
            "shapes_bridge stubbed out %r because this pipeline does not use "
            "it (see module docstring). Something reached %s.%s, which means "
            "a code path relying on the upstream mesher was taken -- use "
            "geo.py instead." % (self.__name__, self.__name__, item))


def _install_stubs():
    stubbed = []
    for name in _OPTIONAL:
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except ImportError:
            sys.modules[name] = _Unavailable(name)
            stubbed.append(name)
    return stubbed


def shapes_dir():
    return os.environ.get("SHAPES_DIR", os.path.expanduser("~/shapes"))


def shapes_commit():
    """Short commit hash of $SHAPES_DIR, for provenance in run records."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=shapes_dir(), text=True, timeout=30,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return out.stdout.strip() or None
    except Exception:                              # noqa: BLE001
        return None


def load_shape_class():
    """Return (Shape, info) where info records the stubs and the commit."""
    d = shapes_dir()
    if not os.path.isdir(d):
        raise FileNotFoundError(
            "SHAPES_DIR=%r does not exist; clone jviquerat/shapes there" % d)
    stubbed = _install_stubs()
    if d not in sys.path:
        sys.path.insert(0, d)
    from shapes import Shape                       # noqa: PLC0415
    return Shape, {"shapes_dir": d, "shapes_commit": shapes_commit(),
                   "stubbed_modules": stubbed}
