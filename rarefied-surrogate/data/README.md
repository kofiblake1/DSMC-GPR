# data/

**This directory holds no data.** It holds `registry.yaml`, the manifest that
says *where* the data is (decision E3).

Everything else here is gitignored: `.gitignore` excludes `data/*` and
re-includes only this file and `registry.yaml`.

> Note the rule is `data/*`, not `data/`. A bare `data/` excludes the directory
> itself, and git cannot re-include a file whose parent directory is excluded —
> so the negations silently did nothing and `registry.yaml` was never committed
> at all. Check with `git check-ignore -v data/registry.yaml`.

## Where the data actually lives

| class | location | lifetime |
|---|---|---|
| Mesh + AERO-F run outputs | `$SCRATCH` (`/scratch/users/kofib`) | **purged after 90 days without content modification** |
| SPARTA VDF pickles | laptop `~/Documents/Meteor_Modelling/code/`; cluster path still open (CONFIRM #8) | archival |
| Long-term datasets | `$GROUP_HOME` (`/home/groups/cfarhat`), 1 TB, ~71% used | account lifetime |

There is **no `$OAK`** on this account, and `$HOME` is 15 GB at ~71% — so
neither is a staging area.

The `$SCRATCH` purge is why the registry records `regenerable: true/false`.
Mesh outputs are cheap to rebuild from `config/` + `src/`, so losing them is a
non-event. The VDF pickles are not, which is why they are described as precious
archived *inputs* rather than intermediates.
