# Security policy

## Supported versions

The `main` branch and the latest tagged release receive security fixes.

## Reporting a vulnerability

Please **do not open a public issue** for security problems.

Use GitHub's private vulnerability reporting: open the repository's **Security** tab and
choose **Report a vulnerability**. If that is unavailable, contact the maintainer through
the address on the GitHub profile.

Include the affected version or commit, a minimal reproduction, the impact you believe it
has, and any suggested fix. You can expect an acknowledgement within a few working days.

## Security-relevant properties

- **In-process code execution.** `stat_run_code` executes Python inside the server process.
  It is intentional and documented, but it is not a sandbox. Do not expose the server to
  untrusted clients; run it with the privileges of the user who owns the analysis.
- **Local data only.** Data tools read the path supplied by the caller. There is no path
  sandbox: run the server with the permissions appropriate to the data it may see. The
  server does not upload data anywhere.
- **No credentials stored.** The project reads no API keys and writes only figures,
  configuration and session state under the user's home directory.
- **Figures and outputs.** Figure files are written to `PINGOUIN_MCP_RESULTS_DIR`; treat that
  directory as sensitive if the analysed data is sensitive.

Issues outside this scope are still welcome as reports, but may be handled as regular bugs.
