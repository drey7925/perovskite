# Security Policy

This is a best-effort game project. I will do my best to avoid introducing security issues. Please exercise standard precautions (don't run as root/admin, etc).

## Out-of-scope

At the moment, the following are out-of-scope:

* Eavesdropping on a connection and observing game state/game chat
* Active MITM allowing access to a game account
* Denial of service for the game itself, or simple CPU/memory exhausion on either clients or servers.
* In-game cheating

There are plans to introduce TLS later, but this requires additional work as well as extra configuration on the part of server owners.

Note that clients authenticate to servers using the OPAQUE protocol; issues in that protocol's integration/implementation will be fixed where known and feasible.

## Memory Safety/Security Principles

* There should be few instances where untrusted data directly goes into system interface (e.g. filenames, etc). For example, most configs live in files with hardcoded
  names (e.g. `settings.ron`), and client cache data is stored in files with a known, controlled name format (e.g. hex representation of a hash).
* 

## Supported Versions

The project is in pre-release. If a security fix is needed, it'll be fixed at the tip of the main branch. If I'm developing at head, and it's not in a releasable state, I'll
also backport the fix to the last stable point on the main branch. A new Windows client release will be built and published.

Please reach out if you need a fix cherrypicked into a different branch.

## Reporting a Vulnerability

Please report vulnerabilities through github's private vulnerability report system linked from [here](https://github.com/drey7925/perovskite/security/policy).
