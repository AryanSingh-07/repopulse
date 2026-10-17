# Historical dataset

This directory contains real public GitHub history metadata collected by
`scripts/collect_history.py`. `history/*.json` retains default-branch ancestry,
committer timestamps, commit SHA identifiers, hashed author email identities,
repository creation dates, collection timestamps and collected HEAD SHAs.
No repository source blobs or raw author email addresses are retained here.
`provenance.json` records source repositories and collection outcomes.

`scripts/train_model.py` produces `historical_windows.csv`, `train.csv`,
`test.csv`, and `training_provenance.json`. Every row identifies its repository,
historical prediction cutoff, complete future outcome interval and partition.
The target `inactive_183d=1` means zero observed default-branch commits in the
following 183 days. A merged pull request contributes when its resulting commit
is reachable from the collected default-branch HEAD. A PR merged solely to a
different branch is outside this operational target.

Feature windows exclude every commit after the row's cutoff. Observation
cutoffs before repository creation or before the first available historical
commit plus 90 days are excluded. Outcome intervals extending past collection
are censored and excluded. Historical snapshots occur quarterly.

The reproducible, label-independent partition rule is the first byte of
SHA-256 of the lowercase repository name modulo four: zero assigns the
repository to test; every other value assigns it to training. Training cutoffs
are no later than December 2022; test cutoffs are July 2023 or later. No
repository appears in both partitions, and the final training outcome interval
ends before the earliest test prediction date.

The 48 projects are a convenience sample of small public libraries from six
language ecosystems, including maintained and archived projects. The sample
is not representative of all GitHub. Quarterly observations from one project
are correlated, and their six-month outcome intervals overlap. The reported
window-level test metrics are descriptive measurements, not independent-trial
confidence bounds or a guarantee on arbitrary repositories.

Git history is reconstructed from current default-branch ancestry. History
rewrites, deleted commits, and historical default-branch changes cannot be
recovered. Commit timestamps may be user-edited. The model card discloses this
retrospective-data limitation.

`.git-cache/` and `.python-deps/` are local caches, excluded from version control
and deployment. Deleting the clone cache does not delete the committed dataset.
