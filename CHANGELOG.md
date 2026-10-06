# Changelog

## [0.4.0](https://github.com/kbhatnagar1506/agentcompile-python/compare/v0.3.0...v0.4.0) (2026-10-06)


### Features

* a bounded decision, a breaker, fail open on what a compiled answer can't honour ([#12](https://github.com/kbhatnagar1506/agentcompile-python/issues/12)) ([3f997bf](https://github.com/kbhatnagar1506/agentcompile-python/commit/3f997bf51afc8e5961dc85c4b539b6c62d8a8731))
* capture streamed answers whole; customer ids; outcome() ([#11](https://github.com/kbhatnagar1506/agentcompile-python/issues/11)) ([8af130e](https://github.com/kbhatnagar1506/agentcompile-python/commit/8af130ec92e8ae56c2a0ec5dd38045ee2cf20bd3))
* capture the Responses API, and any framework through a capturing HTTP transport ([#14](https://github.com/kbhatnagar1506/agentcompile-python/issues/14)) ([ad1c326](https://github.com/kbhatnagar1506/agentcompile-python/commit/ad1c32631fb515c8a4a349897635e3574753802f))
* scrub captured calls on this machine before they are sent ([#9](https://github.com/kbhatnagar1506/agentcompile-python/issues/9)) ([02b5678](https://github.com/kbhatnagar1506/agentcompile-python/commit/02b56783a81aea7b61ac09e00e34fefeb8840c57))


### Bug Fixes

* a batch put back on a full capture queue drops the oldest, counted ([#13](https://github.com/kbhatnagar1506/agentcompile-python/issues/13)) ([17440ba](https://github.com/kbhatnagar1506/agentcompile-python/commit/17440ba28ad2690ceac7dd0666ef8b07a9b3a572))

## [0.3.0](https://github.com/kbhatnagar1506/agentcompile-python/compare/v0.2.1...v0.3.0) (2026-10-01)


### Features

* capture (opt-in) sends each call's request and answer in the background ([#7](https://github.com/kbhatnagar1506/agentcompile-python/issues/7)) ([58f564d](https://github.com/kbhatnagar1506/agentcompile-python/commit/58f564da708c31dbf6d4c11c377dab2e315994de))

## [0.2.1](https://github.com/kbhatnagar1506/agentcompile-python/compare/v0.2.0...v0.2.1) (2026-10-01)


### Bug Fixes

* keep AgentCompile's decision events in the trail ([#5](https://github.com/kbhatnagar1506/agentcompile-python/issues/5)) ([944c178](https://github.com/kbhatnagar1506/agentcompile-python/commit/944c178e5121f3389041335e36cee06a7f80e441))

## [0.2.0](https://github.com/kbhatnagar1506/agentcompile-python/compare/agentcompile-v0.1.0...agentcompile-v0.2.0) (2026-10-01)


### Features

* agentcompile --version ([c2fea67](https://github.com/kbhatnagar1506/agentcompile-python/commit/c2fea67a3bc9b23457a41545f19f94f8005583ce))
