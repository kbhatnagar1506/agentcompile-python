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

1. With a conversation id, the SDK asks AgentCompile for a decision (`/v1/decide`). The whole call takes at most `timeout` (default 2 s), and AgentCompile is told that, so it answers in time.
2. **compiled**: the answer (a tool call or a reply) comes back in exactly the provider's response shape; your loop runs the tool as usual. A tool call is only ever one of the `tools` you passed.
3. **forwarded**: your model is called as normal.
4. **fail-open**: AgentCompile errored, was slow or sent something unusable, so your model is called as normal. After 5 failures in a row the SDK stops asking for 30 s, then tries one call.
5. **unsupported**: the request asks for something a compiled answer couldn't honour (`n` > 1, a forced `tool_choice`, a `response_format`), so your model is called without asking.
6. **no conversation id**: your model is called and no decision is made.

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

## Outcomes and customers

Tell AgentCompile how a conversation ended, so it learns only from the ones that went well:

```python
agentcompile.outcome(ticket.id, "resolved")  # or escalated, unresolved, abandoned, reopened, complaint
```

Pass a stable customer id so repeat jobs are counted per customer (scrubbed like everything
else when capture is on):

```python
with agentcompile.conversation(ticket.id, customer=ticket.customer_id):
    run_agent()
```

Streamed answers are captured too: your agent reads the stream as always, and the whole answer
is captured once it ends.

## The Responses API and any framework

A wrapped OpenAI client captures `client.responses.create(...)` as well (streamed or not). Those
calls always go to your model: compiled answers come in Chat Completions' and Anthropic's shapes.

When a framework builds its own client (LangChain, LiteLLM, CrewAI, Pydantic AI, the OpenAI
Agents SDK), give it a capturing HTTP client instead of wrapping anything:

```python
http = agentcompile.http_client(key="ack_...")         # async: agentcompile.async_http_client()

ChatOpenAI(model="gpt-5", http_client=http)            # LangChain
litellm.client_session = http                          # LiteLLM, CrewAI
OpenAIProvider(http_client=agentcompile.async_http_client())   # Pydantic AI
set_default_openai_client(AsyncOpenAI(http_client=agentcompile.async_http_client()))  # Agents SDK

with agentcompile.conversation(ticket.id):
    run_agent()
```

It captures the POSTs to `/chat/completions`, `/responses` and `/messages` that pass through
it, scrubbed the same way, and leaves everything else alone. It only captures: to have known
jobs answered compiled, use `wrap`. An SDK that takes only `httpx2` clients (newer `anthropic`
releases) takes `agentcompile.http_client(lib="httpx2")`.
`agentcompile.transport()` gives the bare transport, to wrap one of your own.

## Privacy

With `capture=True`, personal data is scrubbed on your machine before anything is sent:
emails, payment cards, phone numbers and account numbers become keyed tokens
(`<email:3f9a1c2e>`), the same value always giving the same token, so AgentCompile can still
match values across a conversation without seeing them. The key stays with you:
`AGENTCOMPILE_SCRUB_KEY` (set the same one on all your servers), else one created once in
`~/.agentcompile/scrub.key`. `scrub=False` turns it off. Live decisions (`/v1/decide`) need
real values to act on a customer's request; they are used in memory and stored scrubbed.
