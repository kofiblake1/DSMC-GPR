"""Partition the tet mesh with mpmetis and drive sower.

The mpmetis at $FRG_BIN is NOT stock METIS. It is patched to read an AERO-F
.top file directly (programs/io.c, filtering on element type 5 = tetrahedron)
and to write sower's decomposition format itself, as `<top>.dec.<nparts>`
(io.c:491-515). So the whole partition step is:

    mpmetis mesh.top 4        ->  mesh.top.dec.4
    sower -fluid -mesh mesh.top -dec mesh.top.dec.4 -cpu 4 -output mesh

The .dec format, from sower's parser (Decomp3D.C:13-77) and mpmetis's writer:

    Decomposition for <file> in <n> parts   <- matched on first 13 chars
    <numSub>
    <n_elems_in_sub_0>
    <e>                                     <- 1-based element ids
    ...

Ids index the *tetrahedra only*: sower sets numElem = selements.size()
(FluidDomain.C:309) and keeps boundary faces separately.

Ordering: sower reads elements onto a stack and stores them at
`elements[numElem-i-1]` (FluidDomain.C:339-345). The stack pop reverses the
read order and the descending index reverses it again, so the two cancel --
elements[0] is the first tet in the .top file. mpmetis writes ids in that same
.top order, so the two agree with no reindexing. `partition_quality` checks
this empirically rather than taking it on trust.

Without -dec, sower produces a single subdomain regardless of -cpu (it warns
"number of cpus set to 1"), so this step is mandatory for a parallel run.
"""

import os
import subprocess

import numpy as np

MPMETIS = os.environ.get("MPMETIS", "/home/groups/cfarhat/bin/mpmetis")
SOWER = os.environ.get("SOWER", "/home/groups/cfarhat/bin/sower")


def dec_path_for(top_path, nparts):
    return "%s.dec.%d" % (top_path, nparts)


def write_dec(path, parts, nparts):
    """Write a .dec ourselves. Used for nparts == 1, and to build deliberate
    decompositions when testing that sower interprets the ids as we expect."""
    parts = np.asarray(parts)
    if len(parts) and (parts.min() < 0 or parts.max() >= nparts):
        raise ValueError("partition ids outside [0, %d)" % nparts)
    order = np.argsort(parts, kind="stable")
    counts = np.bincount(parts, minlength=nparts)
    with open(path, "w") as f:
        f.write("Decomposition written by rarefied.aerof.partition in %d parts\n" % nparts)
        f.write("%d\n" % nparts)
        start = 0
        for isub in range(nparts):
            n = int(counts[isub])
            f.write("%d\n" % n)
            for v in order[start:start + n] + 1:
                f.write("%d\n" % int(v))
            start += n
    return path


def read_dec(path, n_elements):
    """Parse a .dec back into a per-element 0-based partition id array."""
    # Mirror sower's parser: scan line-wise for the header, then read integers
    # from the following lines. Parsing token-wise instead would pick up the
    # count embedded in mpmetis's header text ("... in 4 parts").
    with open(path) as f:
        lines = f.read().splitlines()
    start = None
    for k, ln in enumerate(lines):
        if ln.strip()[:13] == "Decomposition":
            start = k + 1
            break
    if start is None:
        raise ValueError("%s has no 'Decomposition' header line" % path)

    nums = [int(t) for ln in lines[start:] for t in ln.split()
            if t.lstrip("-").isdigit()]
    if not nums:
        raise ValueError("%s has no decomposition data after the header" % path)
    nsub = nums[0]
    parts = np.full(n_elements, -1, dtype=np.int64)
    pos = 1
    for isub in range(nsub):
        n = nums[pos]
        pos += 1
        ids = np.asarray(nums[pos:pos + n], dtype=np.int64)
        pos += n
        parts[ids - 1] = isub
    if (parts < 0).any():
        raise ValueError("%s does not cover every element (%d unassigned)"
                         % (path, int((parts < 0).sum())))
    return parts, nsub


def run_mpmetis(top_path, nparts, gtype=None, log_path=None, timeout=3600,
                n_elements=None):
    """Partition a .top file. Returns (dec_path, parts array)."""
    if nparts < 1:
        raise ValueError("nparts must be >= 1")
    dec = dec_path_for(top_path, nparts)

    if nparts == 1:
        if n_elements is None:
            raise ValueError("n_elements is required for nparts == 1")
        write_dec(dec, np.zeros(n_elements, dtype=np.int64), 1)
        return dec, np.zeros(n_elements, dtype=np.int64)

    cmd = [MPMETIS]
    if gtype:
        cmd.append("-gtype=%s" % gtype)
    cmd += [top_path, str(nparts)]
    proc = subprocess.run(cmd, text=True, timeout=timeout,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if log_path:
        with open(log_path, "w") as f:
            f.write(" ".join(cmd) + "\n\n" + proc.stdout)
    if proc.returncode != 0 or "error" in proc.stdout.lower():
        raise RuntimeError("mpmetis failed (exit %d):\n%s"
                           % (proc.returncode, proc.stdout.strip()))
    if not os.path.exists(dec):
        raise RuntimeError("mpmetis wrote no decomposition at %s" % dec)

    parts, nsub = read_dec(dec, n_elements) if n_elements else (None, nparts)
    return dec, parts


def partition_quality(parts, tets, nodes, nparts):
    """Load balance and interface size.

    This is the check that a decomposition is spatially *coherent*, not merely
    valid. Any permutation of element ids is a legal partition and sower would
    accept it silently, so an ordering mismatch would not raise -- it would
    just produce a scrambled decomposition. A coherent partition touches few
    shared nodes and each subdomain is compact; a scrambled one has an
    interface fraction approaching 1 and subdomains spanning the whole domain.
    """
    parts = np.asarray(parts)
    tets = np.asarray(tets)
    nodes = np.asarray(nodes)
    counts = np.bincount(parts, minlength=nparts)

    # Number of subdomains touching each node, via a boolean incidence matrix.
    touch = np.zeros((len(nodes), nparts), dtype=bool)
    for col in range(4):
        touch[tets[:, col], parts] = True
    n_owners = touch.sum(axis=1)
    used = n_owners > 0
    shared = int((n_owners > 1).sum())

    cent = nodes[tets].mean(axis=1)
    diag = float(np.linalg.norm(nodes.max(axis=0) - nodes.min(axis=0)))
    spread = []
    for isub in range(nparts):
        c = cent[parts == isub]
        if len(c):
            spread.append(float(np.linalg.norm(c.max(axis=0) - c.min(axis=0)) / diag))

    return {
        "nparts": int(nparts),
        "elements_per_part": [int(v) for v in counts],
        "balance_max_over_mean": float(counts.max() / max(counts.mean(), 1e-30)),
        "interface_nodes": shared,
        "interface_node_fraction": float(shared / max(int(used.sum()), 1)),
        "subdomain_extent_over_domain": spread,
    }


def run_sower(top_path, dec_path, ncpu, out_prefix, log_path=None, timeout=3600):
    """sower -fluid: write the per-subdomain files AERO-F reads."""
    cmd = [SOWER, "-fluid", "-mesh", top_path, "-dec", dec_path,
           "-cpu", str(ncpu), "-output", out_prefix]
    proc = subprocess.run(cmd, text=True, timeout=timeout,
                          cwd=os.path.dirname(os.path.abspath(top_path)) or ".",
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if log_path:
        with open(log_path, "w") as f:
            f.write(" ".join(cmd) + "\n\n" + proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError("sower failed (exit %d):\n%s"
                           % (proc.returncode, proc.stdout.strip()))

    found, nsub, written = {}, None, []
    for ln in proc.stdout.splitlines():
        s = ln.strip()
        if s.startswith("Found "):
            head, _, tail = s[len("Found "):].rpartition("(")
            if tail:
                found[head.strip()] = int(tail.rstrip(")"))
        elif "Number of subdomain(s)" in s:
            nsub = int(s.rsplit("=", 1)[1])
        elif s.startswith("... Writing"):
            written.append(s.split("'")[1] if "'" in s else s)

    warnings = [ln.strip() for ln in proc.stdout.splitlines() if "WARNING" in ln.upper()]
    return {"bc_codes": found, "n_subdomains": nsub, "files": written,
            "warnings": warnings, "log": proc.stdout}
