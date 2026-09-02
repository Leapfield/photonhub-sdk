# Security policy

PhotonHub is in an invitation-only beta. If you believe you have found a
security vulnerability in the desktop application, the `phsolver` engine, the
`photonhub` Python SDK, the hosted beta identity service, or the cloud GPU
service, please report it privately.

## Reporting

- Email **beta@leapfield.app** with the subject line `SECURITY`.
- Include the affected component and version (**Help → About** in the desktop
  app, `photonhub.__version__` for the SDK, or the solver archive's
  `manifest.json`), reproduction steps, and the impact you observed.
- Do not include credentials, invitation codes, API keys, or session material
  in a report.

We acknowledge reports within five business days and keep you informed while
the issue is investigated and fixed. Please give us reasonable time to remediate
before public disclosure. There is no bug-bounty programme during the beta.

## Scope notes

- The SDK never contacts the network unless you call `photonhub.web`; the
  desktop application contacts only the beta identity service and, for
  cloud-entitled accounts, the cloud API.
- The desktop application contains no automatic updater; install only signed
  builds whose SHA-256 matches the value published with your invitation.
