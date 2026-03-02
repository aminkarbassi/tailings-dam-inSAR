#!/usr/bin/env python3
"""
setup_credentials.py — Store NASA Earthdata credentials in ~/.netrc
===================================================================
Run this ONCE in your terminal before using any HyP3 or ASF scripts.
Credentials are stored in ~/.netrc (readable only by you, chmod 600).

The hyp3_sdk and asf_search libraries read ~/.netrc automatically.

Usage:
    python scripts/setup_credentials.py
"""

import getpass
import stat
from pathlib import Path

MACHINE = "urs.earthdata.nasa.gov"
NETRC   = Path.home() / ".netrc"


def main():
    print("NASA Earthdata credentials setup")
    print(f"Writing to: {NETRC}")
    print(f"Register at: https://urs.earthdata.nasa.gov/\n")

    username = input("NASA Earthdata username: ").strip()
    password = getpass.getpass("NASA Earthdata password: ").strip()

    if not username or not password:
        print("ERROR: username and password cannot be empty.")
        return

    # Read existing .netrc and remove any stale urs.earthdata.nasa.gov entry
    existing = ""
    if NETRC.exists():
        lines = NETRC.read_text().splitlines()
        skip = False
        kept = []
        for line in lines:
            if line.strip().startswith("machine") and MACHINE in line:
                skip = True
            elif line.strip().startswith("machine") and skip:
                skip = False
            if not skip:
                kept.append(line)
        existing = "\n".join(kept).rstrip() + "\n"

    new_entry = f"\nmachine {MACHINE}\n  login {username}\n  password {password}\n"

    with open(NETRC, "w") as f:
        f.write(existing + new_entry)

    # chmod 600 — required by netrc spec
    NETRC.chmod(stat.S_IRUSR | stat.S_IWUSR)

    print(f"\nCredentials saved to {NETRC} (chmod 600).")
    print("You can now run the pipeline scripts — no further credential setup needed.")


if __name__ == "__main__":
    main()
