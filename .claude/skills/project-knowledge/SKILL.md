---
name: project-knowledge
description: |
  Public project knowledge for the Claude/Codex Telegram bot runtime:
  architecture, topic config, runtime files, operator-owned deployment and direct access,
  monitoring, recovery, testing, and release safety.

  Use when: "project architecture", "bot architecture", "topic config",
  "deployment", "install bot", "monitoring", "recovery", "direct host access",
  "release rules", "public repo rules", "how this bot works", "project knowledge"
---

# Project Knowledge

This repository is a generic open-source Telegram bot runtime for running
Claude Code or Codex from Telegram chats and forum topics.

Use this skill when you need repository-specific context. Keep all additions
public-safe: no private assistant behavior, no personal workflows, no real IDs,
no tokens, no local machine paths, and no runtime state.

## References

Read the relevant reference before making changes:

- [architecture.md](references/architecture.md) for runtime modules and control flow.
- [configuration.md](references/configuration.md) for environment variables, topic config, and runtime files.
- [deployment.md](references/deployment.md) for the operator-owned installation, update, direct-access, and monitoring model.
- [release-safety.md](references/release-safety.md) for public release, sync, staging, and leak-scan rules.
- [testing.md](references/testing.md) for expected checks and minimal test coverage.

Operator skills:

- `bot-setup` for installation, language, systemd autostart, commands, and troubleshooting.
- `topic-setup` for Telegram forum topic creation and project wiring.

## Semantic Vault Card

Human-facing project context and repository navigation: `~/projects/ai-assistant/agent-brain/vault/knowledge/products/ai-assistant/ai-assistant.md`.
