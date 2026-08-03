# Security Policy

## Supported version

Security fixes are applied to the latest release and the default branch.

## Reporting

Do not publish exploitable vulnerabilities in a public issue. Use GitHub's private security-advisory workflow for this repository.

Include:

- affected version and component;
- reproduction steps;
- expected impact;
- suggested mitigation when available.

## Deployment guidance

- Keep the API behind authentication and network controls in non-lab environments.
- Treat uploaded KPI data as potentially sensitive operational information.
- Pin and verify remote dataset checksums for controlled builds.
- Do not mount Docker sockets or privileged devices into the service.
- Validate every action in a simulator and operator approval workflow before RAN actuation.
