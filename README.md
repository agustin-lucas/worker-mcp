# implementation-worker-mcp

A local **stdio MCP server** that lets Codex or Claude Code drive persistent
DeepSeek Harness coding agents as asynchronous implementation workers.

The intended workflow: the lead agent (Sol/Opus) investigates the repository and
decides the architecture, writes a detailed Markdown implementation plan, creates
a dedicated git worktree, and then delegates implementation to a DeepSeek worker
through this MCP. The worker runs in the background while the lead keeps
investigating, and is steered afterwards with `worker_message` /
`worker_interrupt`.

```
                        Codex / Claude Code
                               |
                               | MCP stdio
                               v
                    +-----------------------+
                    | implementation-worker |
                    |      MCP server       |
                    +-----------+-----------+
                                |
                  disk-backed worker registry
                                |
                    launches detached runner
                                |
                                v
                    +-----------------------+
                    |      turn-runner      |
                    | holds worker lock     |
                    +-----------+-----------+
                                |
                                | stdin task
                                v
                    +-----------------------+
                    | DeepSeek Harness      |
                    | --profile headless    |
                    | --json                |
                    | --session-id ...      |
                    +-----------+-----------+
                                |
                                v
                        git worktree
```

## Design in one page

* **The MCP server is not a daemon.** Codex or Claude Code launches it over
  stdio; there is no port, no service, no container, no database.
* **Turns outlive the MCP process.** `worker_start` and `worker_message` spawn a
  *detached* turn-runner (`start_new_session=True`) and return as soon as it owns
  the worker lock. Closing the MCP client does not stop DeepSeek.
* **State lives on disk**, not in memory, so independent Codex and Claude MCP
  processes see and operate on the same logical workers.
* **One turn per worker.** Mutual exclusion is a per-worker `flock`; the lock is
  handed to the runner through the inherited descriptor, so there is no window in
  which two turns can start.
* **BUSY is a result, not a queue.** `worker_message` returns `BUSY` immediately
  when a turn is active. It never queues.
* **Interrupting does not revert code.** `worker_interrupt` kills the DeepSeek
  process tree (SIGTERM, then SIGKILL after a grace period) and blocks until the
  worker can accept another message. Files already written stay written.
* **Sessions are continuous.** After an interruption the next turn resumes the
  same persisted DeepSeek session in the same working directory.
* **Crash-safe.** The OS releases the lock when a runner dies, and a stale run is
  reconciled on the next `worker_status`/`worker_result`. A dead runner can never
  leave a worker permanently BUSY.

## Requirements

| Component | Notes |
|---|---|
| Linux remote development server | POSIX `flock` and process groups; developed and tested on Linux/macOS |
| Python | **>= 3.10** |
| [`uv`](https://docs.astral.sh/uv/) | Recommended; plain `pip`/`venv` works too |
| Node.js | >= 24.2 for the Harness CLI; see the version note below |
| DeepSeek Harness | `@deepseek-ai/dsh` (`dsh` on `PATH`, or an absolute path) |
| DeepSeek credentials | `DEEPSEEK_API_KEY`, or credentials already stored in `DSH_HOME` |

> **Node version note.** The published `dsh` entry point dispatches itself with
> `import.meta.main`. On Node releases that do not define it (verified on
> Node 24.1.0) every `dsh` invocation exits 0 and prints nothing. Confirm your
> Node has it before relying on the CLI:
>
> ```bash
> node --input-type=module -e "console.log(import.meta.main)"
> ```
>
> `import.meta.main` landed in **Node.js 24.2.0**
> ([release notes](https://nodejs.org/en/blog/release/v24.2.0)); if the command
> above prints `undefined`, install Node >= 24.2 or use the small launcher shown
> in [Troubleshooting](#dsh-exits-silently).

## Installation

```bash
git clone <this-repo> ~/tools/implementation-worker-mcp
cd ~/tools/implementation-worker-mcp
uv sync                    # creates .venv and installs mcp v2 + dev tools
```

Absolute entry points (use these in MCP configuration; never rely on `PATH` or
the caller's working directory):

```text
~/tools/implementation-worker-mcp/.venv/bin/implementation-worker-mcp
~/tools/implementation-worker-mcp/.venv/bin/implementation-worker-runner
```

Install the DeepSeek Harness:

```bash
npm install -g @deepseek-ai/dsh
dsh --version
dsh --profile headless --help     # must list --json and --session-id
```

### Credentials

The detached runner inherits the environment of the MCP server process, so the
Harness must be able to find credentials **on the remote server**:

```bash
export DEEPSEEK_API_KEY=...        # add to ~/.bashrc / ~/.profile
export DSH_HOME=~/.dsh             # only if you keep it outside the default
```

`DEEPSEEK_API_KEY` is never read, copied, or written by this project; only the
Harness sees it. Worker state and run logs never contain credentials.

> The `deepseek-official` provider route requires an actual API key. A
> `~/.dsh/.credentials.yaml` device token alone is not enough — the Harness
> reports `MISSING_CREDENTIAL` and the run is recorded as `failed`.

## Configuration

All configuration is environment variables, read once when the server starts.

| Variable | Default | Meaning |
|---|---|---|
| `WORKER_MCP_STATE_DIR` | `$XDG_STATE_HOME/implementation-worker-mcp`, else `~/.local/state/implementation-worker-mcp` | Where the worker registry lives |
| `WORKER_MCP_DSH_COMMAND_JSON` | `["dsh"]` | JSON argv array for the Harness executable |
| `WORKER_MCP_INTERRUPT_GRACE_SECONDS` | `10` | SIGTERM→SIGKILL grace period |
| `WORKER_MCP_MAX_PLAN_BYTES` | `262144` | Reject larger plan files |
| `WORKER_MCP_MAX_FINAL_TEXT_BYTES` | `262144` | Cap on `final_text` stored in `run.json` |
| `WORKER_MCP_LOG_LEVEL` | `INFO` | Server/runner log level (stderr and `runner.log`) |

`WORKER_MCP_DSH_COMMAND_JSON` is parsed as **JSON**, not shell syntax, so there
is no quoting or word-splitting layer to get wrong:

```bash
# globally installed
export WORKER_MCP_DSH_COMMAND_JSON='["dsh"]'
# absolute install
export WORKER_MCP_DSH_COMMAND_JSON='["/home/user/.local/bin/dsh"]'
# ad-hoc via npx
export WORKER_MCP_DSH_COMMAND_JSON='["npx","-y","@deepseek-ai/dsh"]'
```

### State layout

```text
~/.local/state/implementation-worker-mcp/
└── workers/
    └── auth-refactor/
        ├── worker.json          # name, backend, fixed cwd, session id, active/last run
        ├── lock                 # the flock that serializes turns
        └── runs/
            └── <run-id>/
                ├── run.json     # status, pids, exit code, final text, error
                ├── prompt.txt   # exactly what was sent on stdin
                ├── events.jsonl # raw Harness --json stream, verbatim
                ├── stderr.log   # Harness stderr
                ├── runner.log   # turn-runner diagnostics
                └── interrupt.request   # present iff this run was interrupted
```

The state directory is created with `0700` permissions and the worker name is
validated against `^[A-Za-z0-9._-]{1,100}$`, so no name can escape the tree.

## Manual smoke test

Run the server directly and drive it by hand (it speaks MCP on stdout, so any
output you see is protocol traffic):

```bash
export WORKER_MCP_STATE_DIR=/tmp/worker-mcp-state
~/tools/implementation-worker-mcp/.venv/bin/implementation-worker-mcp
```

With [MCP Inspector](https://github.com/modelcontextprotocol/inspector):

```bash
npx @modelcontextprotocol/inspector \
  ~/tools/implementation-worker-mcp/.venv/bin/implementation-worker-mcp
```

Or use the repository's own tests as a smoke test:

```bash
uv run pytest -q                          # full suite; no credentials needed
uv run pytest -m real_binary -q           # drives the real dsh binary
uv run pytest -m real_harness -q          # needs an API key; disposable repo
```

## VS Code Remote SSH

Everything runs on the remote host. Nothing is forwarded, and no inbound port is
opened:

```text
Laptop
  |
  | VS Code Remote SSH
  v
remote Linux server
  |
  +-- Codex extension / Claude Code extension
  +-- Codex or Claude stdio MCP process
  +-- detached implementation-worker-runner
  +-- DeepSeek Harness (dsh)
  +-- source repository / git worktree
```

Practical consequences:

* The MCP server, the runners, the Harness and the repository all share the
  **remote** filesystem and environment. `DEEPSEEK_API_KEY`, `PATH` and
  `DSH_HOME` must be set on the remote host.
* Register the server with an **absolute** executable path. A relative path or a
  bare command name resolves against whatever working directory the extension
  host happens to have.
* If you use `direnv`, `asdf`, `nvm` or a virtualenv activated from `~/.bashrc`,
  remember that a non-interactive GUI/extension process may not source it. Set
  `WORKER_MCP_DSH_COMMAND_JSON` to an absolute path instead of relying on `PATH`.
* Codex CLI and the Codex IDE extension read the same `~/.codex/config.toml`, so
  configuring the CLI on the remote host configures the extension for that host.

## Codex setup

Codex reads user-level MCP servers from `~/.codex/config.toml`. The installed CLI
(v0.137) provides:

```bash
codex mcp add implementation-worker -- \
  /home/user/tools/implementation-worker-mcp/.venv/bin/implementation-worker-mcp
```

Add environment variables at the same time when needed:

```bash
codex mcp add implementation-worker \
  --env WORKER_MCP_STATE_DIR=/home/user/.local/state/implementation-worker-mcp \
  --env 'WORKER_MCP_DSH_COMMAND_JSON=["/home/user/.local/bin/dsh"]' \
  -- /home/user/tools/implementation-worker-mcp/.venv/bin/implementation-worker-mcp
```

Verify:

```bash
codex mcp list
codex mcp get implementation-worker
```

The equivalent TOML, if you prefer to edit the file directly:

```toml
[mcp_servers.implementation-worker]
command = "/home/user/tools/implementation-worker-mcp/.venv/bin/implementation-worker-mcp"
args = []
env = { WORKER_MCP_DSH_COMMAND_JSON = '["/home/user/.local/bin/dsh"]' }
```

## Claude Code setup

Register at **user scope** so every project on the remote host can use it:

```bash
claude mcp add --scope user implementation-worker -- \
  /home/user/tools/implementation-worker-mcp/.venv/bin/implementation-worker-mcp
```

Pass environment variables with `-e` when needed:

```bash
claude mcp add --scope user implementation-worker \
  -e WORKER_MCP_DSH_COMMAND_JSON='["/home/user/.local/bin/dsh"]' \
  -- /home/user/tools/implementation-worker-mcp/.venv/bin/implementation-worker-mcp
```

Verify:

```bash
claude mcp list
```

and inside a session use `/mcp` to inspect the connection and the worker tools.
(A user-scope entry is stored in `~/.claude.json`; a project-scope `.mcp.json`
also works but only covers that project.)

## MCP surface

| Tool | Inputs | Purpose |
|---|---|---|
| `worker_start` | `name`, `cwd`, `plan_path`, `additional_instructions?` | Create a worker bound to one worktree and start its first turn from a Markdown plan. **Returns before implementation finishes.** |
| `worker_message` | `name`, `message` | Send the next turn to the persisted session. Returns `BUSY` (never queues) if a turn is active. |
| `worker_status` | `name` | Cheap status: state, cwd, session availability, active/last run. Repairs stale state. |
| `worker_result` | `name`, `run_id?` | Run outcome: status, timings, exit code, final text, error. Defaults to the most recent run. |
| `worker_interrupt` | `name` | Stop the active turn, keep filesystem changes, and wait until the worker can accept another message. |
| `worker_list` | – | Every known worker, shared across MCP clients. |

### Result codes

Expected operational outcomes come back as normal tool results with a stable
`code` and `ok: false` (not as protocol errors), so a controller never has to
parse English to tell `BUSY` from a broken server:

```text
STARTED  BUSY  IDLE  INTERRUPTED  ALREADY_FINISHED  NOT_FOUND
ALREADY_EXISTS  CWD_IN_USE  NO_SESSION  INVALID_ARGUMENT  FAILED
```

`worker_status` and `worker_result` use the upper-cased worker/run state as their
success code (`IDLE`, `RUNNING`, `COMPLETED`, `FAILED`, `INTERRUPTED`).
`CWD_IN_USE` is an addition to the plan's list: it distinguishes "two workers
would share one worktree" from "that name is taken".

A genuine internal crash still surfaces as an MCP tool error (`is_error: true`)
with no traceback text, and the details are logged server-side.

## Example controller workflow

```text
1. Controller creates a worktree:
   git worktree add ../worktrees/auth-refactor -b auth-refactor

2. Controller writes the plan:
   .agent/plans/auth-refactor.md

3. Controller starts the worker (returns immediately):

   worker_start(
       name="auth-refactor",
       cwd="/home/user/worktrees/auth-refactor",
       plan_path="/home/user/repo/.agent/plans/auth-refactor.md"
   )
   -> {"ok": true, "code": "STARTED", "run_id": "..."}

4. Controller keeps investigating or reviewing other work.

5. Controller checks progress:

   worker_status("auth-refactor")
   -> {"state": "running", "session_id_available": true, ...}

6. A correction turns out to be urgent:

   worker_message("auth-refactor", "Do not modify schema.py")
   -> {"ok": false, "code": "BUSY", "active_run_id": "..."}

7. Controller interrupts and waits for the worker to be free:

   worker_interrupt("auth-refactor")
   -> {"ok": true, "code": "INTERRUPTED", ...}

8. Immediately afterwards (no BUSY, no race):

   worker_message(
       "auth-refactor",
       "Important correction: do not modify schema.py. Revert any changes "
       "already made there, preserve the existing API, and continue."
   )
   -> {"ok": true, "code": "STARTED"}

9. DeepSeek resumes the same persisted session, in the same worktree.
   Files it had already written are still there.

10. When the worker goes idle, review its work:

    worker_result("auth-refactor")
    git -C ../worktrees/auth-refactor diff

11. Send review feedback as another worker_message to the same worker.
```

## Suggested agent policy

Optional text for `AGENTS.md` or `CLAUDE.md` — a *usage* policy, not something the
MCP enforces:

```text
For substantial implementation work, act primarily as architect and reviewer.

After determining the design, write a detailed Markdown implementation plan
and delegate implementation through the implementation-worker MCP.

Workers execute asynchronously and only one turn may run at a time.

If worker_message returns BUSY:
- normally allow the worker to finish and check its status later;
- if immediate steering is important, call worker_interrupt, then send the
  corrective message after interruption succeeds.

Review all worker changes yourself. Do not assume worker output is correct.
Use the same worker for review/fix iterations so it retains implementation
context.
```

## Testing

```bash
uv run pytest -q                     # 129 tests; fake Harness, no credentials
uv run pytest -m "not real_harness" -q
uv run pytest -m real_binary -q      # drives the real dsh process (no model call)
WORKER_MCP_REAL_HARNESS=1 uv run pytest -m real_harness -q -s   # real DeepSeek runs
```

`uv run` does not load `.env` automatically in every version, so export the key
first (`set -a; . ./.env; set +a`) or pass `--env-file .env`.

`-m real_binary` drives the genuine `dsh` process through the whole stack without
making a model call (it points `WORKER_MCP_TEST_DSH_COMMAND_JSON` at your Harness,
so it needs no credentials). The main suite never calls DeepSeek at all: it runs
the real lock, the real detached runner, and a fake Harness executable (`tests/fake_harness.py`) that speaks the
same CLI and JSON contract as the installed Harness, and it asserts behaviour —
not just return codes:

* async start (returns in milliseconds, before the Harness finishes);
* session id captured *during* an active turn;
* `BUSY` while running, with **no second Harness process** spawned;
* session id and cwd reused on resume;
* interruption kills the Harness **and its children**, releases the lock, keeps
  partial file changes, and lets the next `worker_message` start immediately;
* SIGTERM-ignoring backends are escalated to SIGKILL;
* a SIGKILLed turn-runner does not leave the worker BUSY, and stale recorded PIDs
  are never signalled;
* `run.json` is never observed half-written;
* the real stdio MCP transport lists tools, returns structured results, and
  keeps stdout free of anything but JSON-RPC.

`tests/test_real_harness.py` is the only suite that spends API credits; it runs
in a disposable git repository and is skipped unless `WORKER_MCP_REAL_HARNESS=1`
and credentials are present.

See [`docs/dsh-integration-notes.md`](docs/dsh-integration-notes.md) for the
DeepSeek Harness CLI/JSON contract this was built and verified against, and for
the differences from the original implementation plan.

> `.dsh-cli/` holds a local `npm install` of `@deepseek-ai/dsh` used by the
> `real_binary` tests. It is gitignored; delete it if you do not want it, or
> recreate it with
> `npm install --prefix .dsh-cli @deepseek-ai/dsh@0.2.0-rc.2`.

## Security notes

* `shell=True` is never used; prompts are delivered on stdin, never interpolated
  into a command line.
* Worker names are validated and state paths are never derived from user input
  beyond that validated name.
* API keys are never persisted; only the Harness sees them.
* State is created with `0700`/`0600` permissions.
* Nothing is printed to MCP stdout except protocol traffic.
* Interruption and cleanup never run `git reset`, `git checkout`, `git restore`
  or any other destructive command.
* **Logs may contain repository-sensitive information.** `events.jsonl` holds
  model output and tool activity (file contents, diffs, command output), and
  `stderr.log` holds Harness diagnostics. Treat the state directory as sensitive,
  and remember that any process running as your user can read it.

## Troubleshooting

### `dsh` exits silently

If `dsh --help` prints nothing and exits 0, your Node does not define
`import.meta.main` (the published `bin.js` guard never fires). Upgrade Node, or
use a launcher that dispatches explicitly:

```js
// /home/user/tools/dsh-launch.mjs
const mod = await import('@deepseek-ai/dsh/lib/bin.js');
await mod.runCli();
```

```bash
export WORKER_MCP_DSH_COMMAND_JSON='["node","/home/user/tools/dsh-launch.mjs"]'
```

### Runs fail with `MISSING_CREDENTIAL`

The Harness could not find an API key for the provider route. Export
`DEEPSEEK_API_KEY` in the environment the MCP server (and therefore the runner)
inherits, then inspect the run:

```bash
cat ~/.local/state/implementation-worker-mcp/workers/<name>/runs/<run-id>/run.json
cat ~/.local/state/implementation-worker-mcp/workers/<name>/runs/<run-id>/stderr.log
```

### Everything returns `BUSY`

A turn is genuinely running. `worker_status` shows `active_run_id`; if the runner
died, the next `worker_status` reconciles the stale run automatically. As a last
resort `worker_interrupt` stops whatever holds the lock.

### `worker_message` returns `NO_SESSION`

The worker is idle but never recorded a session id — usually because its very
first turn failed before the Harness emitted its opening `session` event (for
example a missing Harness executable or missing credentials). Check
`run.json`/`stderr.log` for that run, fix the cause, and start a fresh worker.

### `CWD_IN_USE`

Another logical worker already owns that canonical directory. One worktree is
meant to map to one worker; create a separate `git worktree` for the new worker.

## License

MIT.
