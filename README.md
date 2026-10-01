# agentcompile

[![PyPI](https://img.shields.io/pypi/v/agentcompile)](https://pypi.org/project/agentcompile/) [![Python](https://img.shields.io/pypi/pyversions/agentcompile)](https://pypi.org/project/agentcompile/) [![CI](https://github.com/kbhatnagar1506/agentcompile-python/actions/workflows/ci.yml/badge.svg)](https://github.com/kbhatnagar1506/agentcompile-python/actions/workflows/ci.yml) [![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

```
pip install agentcompile
```

Wrap your agent's model client. Jobs your agent repeats run compiled, with no model call; everything else goes to your model unchanged, with your own provider key.

```python
from openai import OpenAI
import agentcompile

client = agentcompile.wrap(OpenAI(), key="ack_...")   # or AGENTCOMPILE_KEY

resp = client.chat.completions.create(
    model="gpt-5", messages=messages, tools=tools,
    conversation_id=ticket.id,                         # one id per conversation
)
```

Anthropic works the same way (`agentcompile.wrap(Anthropic())`, then `client.messages.create(...)`). So do async clients, streaming and tool calls. Instead of the keyword, you can set the id for a block of calls:

```python
with agentcompile.conversation(ticket.id):
    run_agent(client)
```

## What happens on each call

1. With a conversation id, the SDK asks AgentCompile for a decision (`/v1/decide`; it waits at most `timeout`, default 2 s).
2. **compiled**: the answer (a tool call or a reply) comes back in exactly the provider's response shape; your loop runs the tool as usual.
3. **forwarded**: your model is called as normal.
4. **fail-open**: AgentCompile errored, was slow or sent something unusable, so your model is called as normal.
5. **no conversation id**: your model is called and no decision is made.

Your provider key never leaves your process. `mode="shadow"` decides but always calls your model, so you can see what it would have done before you switch it on.

## The trail

Every call is written to `~/.agentcompile/trail.jsonl` (`trail=` sets a path, or `False` turns it off; `on_event=` gets each event).

```
agentcompile trail          # recent calls and a summary
agentcompile trail -f       # follow live
```

```
14:02:11  c1            compiled         get_order_details                         gpt-5  48 ms decide
14:02:12  c2            forwarded        no job matched                            gpt-5  51 ms decide
2 calls: compiled 1, forwarded 1. Model calls avoided: 1 (50%).
```

## Keys

Your AgentCompile key (`ack_...`) identifies your company by itself. We show it once and store only its hash; a new key revokes the old one. AgentCompile is pre-launch: to get a key, write to founders@tryagentcompile.com.

## Releasing

Commit with conventional messages (`fix: ...` is a patch, `feat: ...` a minor version, `feat!: ...` a major one). A release PR stays open with the next version and its changelog; merging it tags the release and publishes it to PyPI. The version always comes from the git tag.

## Development

```
pip install -e ".[dev]"
pytest tests
```
