# Security

## Reporting a vulnerability

Please don't open a public issue for security problems. Report them privately through **GitHub → Security → Report a vulnerability** on this repository, or email founders@tryagentcompile.com.

We'll acknowledge your report within 3 business days and keep you updated until it's fixed.

## How the SDK handles your data

- Your model provider key never leaves your process: the SDK calls your provider itself.
- To decide, the SDK sends the conversation's messages, system prompt, tool definitions and model name to AgentCompile, with your AgentCompile key. Nothing else from the request is sent.
- If AgentCompile fails, is slow or answers something unusable, your call goes straight to your model (fail-open).
- The trail is written only to a local file (`~/.agentcompile/trail.jsonl` by default) unless you pass `trail=False`.

## Supported versions

Security fixes go into the latest release.
