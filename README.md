# agentcompile (Python SDK)

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

`agent-compiler endpoint-key --company <name>` issues a key of the form `ack_<company>.<secret>`. It's shown once, and only its hash is stored. The key alone identifies the company; issuing a new one revokes the old.

## Development

```
pip install -e "sdk/python[dev]"
pytest sdk/python/tests
```
