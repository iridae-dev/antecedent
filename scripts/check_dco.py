"""Check PR sign-offs, including explicit retrospective sign-offs by the original author."""

import re
import subprocess
import sys
from pathlib import Path


def check_range(revision_range: str, cwd: Path) -> tuple[int, list[str]]:
    # Git parses the trailer block. NUL separates fields even in multiline messages.
    fmt = (
        "%H%x00%ae%x00%(trailers:key=Signed-off-by,valueonly,separator=%x0a)"
        "%x00%(trailers:key=DCO-remediation,valueonly,separator=%x0a)"
    )
    log = subprocess.check_output(
        ["git", "log", "--no-merges", "-z", f"--format={fmt}", revision_range, "--"],
        cwd=cwd,
    )
    fields = log.decode("utf-8", errors="replace").split("\0")
    unsigned = {}
    remediations = set()
    for index in range(0, len(fields) - 1, 4):
        sha, author, signers, targets = fields[index:index + 4]
        emails = set()
        for signer in signers.splitlines():
            match = re.fullmatch(r".+ <([^<>\s]+@[^<>\s]+)>", signer)
            if match:
                emails.add(match[1])
        if not emails:
            unsigned[sha] = author
        if author in emails:
            remediations.update((target, author) for target in targets.splitlines())
    return (len(fields) - 1) // 4, [
        sha for sha, author in unsigned.items() if (sha, author) not in remediations
    ]


if __name__ == "__main__":
    count, missing = check_range(sys.argv[1], Path.cwd())
    for commit in missing:
        print(f"Missing DCO on {commit}", file=sys.stderr)
    print(f"DCO: {count} non-merge commits checked, {len(missing)} missing sign-offs")
    raise SystemExit(bool(missing))
