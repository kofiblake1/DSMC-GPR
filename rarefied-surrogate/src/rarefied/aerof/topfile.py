"""Drive gmsh2top and verify what it produced.

gmsh2top takes a *prefix*, not a filename: it opens <prefix>.msh and writes
<prefix>.top into the current directory. It also opens the input with no
STATUS='OLD', so a missing input file is silently created empty and the run
dies on EOF instead of reporting "cannot open" -- hence the explicit existence
check here.
"""

import os
import re
import subprocess

GMSH2TOP = os.environ.get("GMSH2TOP", "/home/groups/cfarhat/bin/gmsh2top")

# "Elements <name> using FluidNodes  <count>" lines on stdout
_GROUP_RE = re.compile(r"Elements\s+(\S+)\s+using\s+FluidNodes\s+(\d+)")
_COUNTS_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s*$")


def run_gmsh2top(msh_path, log_path=None, timeout=1800):
    """Convert <prefix>.msh to <prefix>.top. Returns a summary dict."""
    if not os.path.exists(msh_path):
        raise FileNotFoundError(msh_path)
    workdir = os.path.dirname(os.path.abspath(msh_path)) or "."
    prefix = os.path.basename(msh_path)
    if not prefix.endswith(".msh"):
        raise ValueError("expected a .msh file, got %r" % msh_path)
    prefix = prefix[:-4]

    proc = subprocess.run([GMSH2TOP, prefix], cwd=workdir, text=True, timeout=timeout,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if log_path:
        with open(log_path, "w") as f:
            f.write("%s %s   (cwd %s)\n\n%s" % (GMSH2TOP, prefix, workdir, proc.stdout))
    if proc.returncode != 0:
        raise RuntimeError("gmsh2top failed (exit %d):\n%s"
                           % (proc.returncode, proc.stdout.strip()))

    top_path = os.path.join(workdir, prefix + ".top")
    if not os.path.exists(top_path):
        raise RuntimeError("gmsh2top wrote no .top file")

    groups, n_vol, n_bc = {}, None, None
    for ln in proc.stdout.splitlines():
        m = _GROUP_RE.search(ln)
        if m:
            groups[m.group(1)] = int(m.group(2))
            continue
        m = _COUNTS_RE.match(ln)
        if m and n_vol is None:
            n_vol, n_bc = int(m.group(1)), int(m.group(2))

    return {"top_file": top_path, "groups": groups,
            "n_volume_elements": n_vol, "n_boundary_elements": n_bc,
            "log": proc.stdout}


def read_top_blocks(top_path):
    """Node count and per-group element counts, read back from the .top file.

    Independent of gmsh2top's own stdout, so it catches a converter that
    reports one thing and writes another.
    """
    nodes = 0
    groups = {}
    current = None
    with open(top_path) as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            head = s.split()[0]
            if head == "Nodes":
                current = "__nodes__"
            elif head == "Elements":
                current = s.split()[1]
                groups.setdefault(current, 0)
            elif current == "__nodes__":
                nodes += 1
            elif current is not None:
                groups[current] += 1
    return {"n_nodes": nodes, "groups": groups}


def verify(mesh, summary, expected_groups):
    """Cross-check gmsh2top output against the mesh it was built from.

    `expected_groups` maps .top element-set name -> expected element count.
    Returns a list of problem strings; empty means everything matched.
    """
    problems = []
    on_disk = read_top_blocks(summary["top_file"])

    if on_disk["n_nodes"] != len(mesh.nodes):
        problems.append("top has %d nodes, mesh has %d"
                        % (on_disk["n_nodes"], len(mesh.nodes)))

    for name, want in expected_groups.items():
        got = on_disk["groups"].get(name)
        if got is None:
            problems.append("top is missing element set %r (has %s)"
                            % (name, sorted(on_disk["groups"])))
        elif got != want:
            problems.append("element set %r has %d elements, expected %d"
                            % (name, got, want))

    extra = set(on_disk["groups"]) - set(expected_groups)
    if extra:
        problems.append("top has unexpected element sets: %s" % sorted(extra))

    return problems, on_disk
