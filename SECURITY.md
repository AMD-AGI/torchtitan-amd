# Security Policy

AMD-AGI takes the security of our software and AI/ML artifacts seriously. This
policy describes how to report a vulnerability in `torchtitan-amd` and what to
expect from the maintainers.

## Supported Versions

Security fixes are applied to the latest release and the default development
branch. Older tags may not receive patches; please upgrade to the most recent
version before reporting an issue.

| Version            | Supported          |
| ------------------ | ------------------ |
| Latest release     | :white_check_mark: |
| Default branch      | :white_check_mark: |
| Older tags/branches | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
pull requests, or discussions.**

Instead, use one of the private channels below:

1. **GitHub Private Vulnerability Reporting (preferred).** Open a private report
   from the repository's **Security** tab → **Report a vulnerability**. This
   creates a private advisory visible only to repository maintainers. See
   [GitHub's documentation](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
   for details.
2. **AMD PSIRT.** For coordinated disclosure across AMD products, contact the
   AMD Product Security Incident Response Team at **psirt@amd.com**. See the
   [AMD Product Security](https://www.amd.com/en/resources/product-security.html)
   page for AMD's disclosure process and PGP key.

When reporting, please include as much of the following as possible to help us
triage quickly:

- A description of the vulnerability and its potential impact.
- Steps to reproduce, including a minimal proof of concept where possible.
- Affected version, commit, branch, and environment (OS, ROCm/GPU, Python,
  PyTorch versions).
- Any suggested remediation or mitigation.

### AI/ML-specific concerns

Because this repository is used to train and run generative AI models, please
also report issues such as: model/checkpoint poisoning, malicious or unsafe
deserialization of weights or configs, supply-chain risks in training data or
dependencies, and adversarial inputs that can cause unsafe code execution.

## Response Process

- **Acknowledgement:** We aim to acknowledge new reports within **3 business
  days**.
- **Assessment:** We will investigate, determine severity, and keep you updated
  on remediation progress.
- **Disclosure:** We follow coordinated disclosure. We will work with you to
  agree on a public disclosure timeline once a fix or mitigation is available,
  and will credit reporters who wish to be acknowledged.

Please make a good-faith effort to avoid privacy violations, data destruction,
and service interruption while researching. Do not access or modify data that
does not belong to you.
