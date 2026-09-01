"""Local read-only web viewer for experiment run traces."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .artifacts import selected_artifact_dir, selected_artifact_dirs

DEFAULT_EVENTS_LIMIT = 200
MAX_EVENTS_LIMIT = 1000


class _RequestError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


def _first(params: dict[str, list[str]], name: str) -> str | None:
    values = params.get(name)
    if not values:
        return None
    return values[0]


def _int_param(params: dict[str, list[str]], name: str) -> int | None:
    raw = _first(params, name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        raise _RequestError(HTTPStatus.BAD_REQUEST, f"invalid integer parameter: {name}") from None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


class _ViewerState:
    """Resolved served root plus path-confinement helpers for run artifacts."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def experiment_dirs(self) -> list[Path]:
        if (self.root / "experiment_manifest.json").is_file():
            return [self.root]
        manifests = sorted(self.root.rglob("experiment_manifest.json"))
        return [manifest.parent for manifest in manifests]

    def resolve_experiment(self, relative: str | None) -> Path:
        if relative is None:
            raise _RequestError(HTTPStatus.BAD_REQUEST, "missing experiment parameter")
        candidate = (self.root / relative).resolve()
        if not candidate.is_relative_to(self.root):
            raise _RequestError(HTTPStatus.FORBIDDEN, "path escapes the served root")
        if not (candidate / "experiment_manifest.json").is_file():
            raise _RequestError(HTTPStatus.NOT_FOUND, "experiment not found")
        return candidate

    def resolve_run_artifact_dir(self, experiment: Path, run_id: str | None) -> Path:
        if not run_id:
            raise _RequestError(HTTPStatus.BAD_REQUEST, "missing run parameter")
        run_dir = (experiment / "runs" / run_id).resolve()
        if not run_dir.is_relative_to(self.root):
            raise _RequestError(HTTPStatus.FORBIDDEN, "path escapes the served root")
        if not run_dir.is_dir():
            raise _RequestError(HTTPStatus.NOT_FOUND, "run not found")
        try:
            artifact_dir = selected_artifact_dir(run_dir).resolve()
        except ValueError as exc:
            raise _RequestError(HTTPStatus.FORBIDDEN, str(exc)) from exc
        if not artifact_dir.is_relative_to(self.root):
            raise _RequestError(HTTPStatus.FORBIDDEN, "path escapes the served root")
        return artifact_dir


_RUN_INDEX_FIELDS = (
    "infection_count",
    "distance_two_infection_count",
    "non_dm_path_infection_count",
    "total_messages",
    "messages_undelivered",
)


def _run_index_entry(run_dir: Path, artifact_dir: Path) -> dict[str, Any] | None:
    try:
        summary = _read_json(artifact_dir / "summary.json")
    except (OSError, ValueError):
        return None
    if not isinstance(summary, dict):
        return None
    replicate: int | None = None
    try:
        cell = _read_json(run_dir / "run_manifest.json").get("cell", {})
        if cell.get("replicate") is not None:
            replicate = int(cell["replicate"])
    except (OSError, ValueError, TypeError, AttributeError):
        replicate = None
    entry: dict[str, Any] = {
        "run_id": str(summary.get("run_id") or run_dir.name),
        "case_id": summary.get("case_id", ""),
        "goal_id": summary.get("goal_id", ""),
        "condition": summary.get("condition", ""),
        "defense": summary.get("defense", "none"),
        "topology": summary.get("topology", ""),
        "model": summary.get("model", ""),
        "seed": summary.get("seed"),
        "origin_agent_id": summary.get("origin_agent_id", ""),
        "bridge_agent_id": summary.get("bridge_agent_id"),
        "completed": bool(summary.get("completed", False)),
        "primary_endpoint_eligible": bool(summary.get("primary_endpoint_eligible", True)),
        "spontaneous_creation_success": bool(summary.get("spontaneous_creation_success", False)),
        "task_completion_rate": summary.get("task_completion_rate", 0.0),
        "replicate": replicate,
    }
    for field in _RUN_INDEX_FIELDS:
        entry[field] = summary.get(field, 0)
    return entry


def _experiment_index(root: Path, experiment: Path) -> dict[str, Any]:
    try:
        manifest = _read_json(experiment / "experiment_manifest.json")
        config = manifest.get("config", {}) if isinstance(manifest, dict) else {}
    except (OSError, ValueError):
        manifest, config = {}, {}
    runs = []
    for run_dir, artifact_dir in selected_artifact_dirs(experiment):
        entry = _run_index_entry(run_dir, artifact_dir)
        if entry is not None:
            runs.append(entry)
    return {
        "path": experiment.relative_to(root).as_posix(),
        "experiment_id": config.get("experiment_id", experiment.name),
        "protocol_version": manifest.get("harness_protocol_version", ""),
        "config_fingerprint": manifest.get("config_fingerprint", ""),
        "runs": runs,
    }


def _run_detail(
    state: _ViewerState, experiment: Path, artifact_dir: Path, run_id: str
) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "run_id": run_id,
        "experiment": experiment.relative_to(state.root).as_posix(),
    }
    artifacts = {
        "manifest": "run_manifest.json",
        "summary": "summary.json",
        "agent_snapshots": "agent_snapshots.json",
        "environment": "environment.json",
        "judge_outputs": "judge_outputs.json",
    }
    for key, name in artifacts.items():
        path = artifact_dir / name
        if not path.is_file():
            detail[key] = None
            continue
        try:
            detail[key] = _read_json(path)
        except (OSError, ValueError):
            detail[key] = None
    return detail


def _query_events(artifact_dir: Path, params: dict[str, list[str]]) -> dict[str, Any]:
    agent_id = _first(params, "agent_id")
    kind = _first(params, "kind")
    query = _first(params, "q")
    round_from = _int_param(params, "round_from")
    round_to = _int_param(params, "round_to")
    offset = max(0, _int_param(params, "offset") or 0)
    limit_raw = _int_param(params, "limit")
    limit = DEFAULT_EVENTS_LIMIT if limit_raw is None else min(max(1, limit_raw), MAX_EVENTS_LIMIT)
    query_lower = query.lower() if query else None

    total = 0
    events: list[dict[str, Any]] = []
    events_path = artifact_dir / "events.jsonl"
    if events_path.is_file():
        with events_path.open(encoding="utf-8") as handle:
            for line in handle:
                if query_lower is not None and query_lower not in line.lower():
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                if agent_id and event.get("agent_id") != agent_id:
                    continue
                if kind and event.get("kind") != kind:
                    continue
                round_index = event.get("round_index")
                if round_from is not None and (
                    not isinstance(round_index, int) or round_index < round_from
                ):
                    continue
                if round_to is not None and (
                    not isinstance(round_index, int) or round_index > round_to
                ):
                    continue
                total += 1
                if total > offset and len(events) < limit:
                    events.append(event)
    return {"total": total, "offset": offset, "limit": limit, "events": events}


class _ViewerHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], state: _ViewerState) -> None:
        super().__init__(server_address, _ViewerRequestHandler)
        self.viewer_state = state


class _ViewerRequestHandler(BaseHTTPRequestHandler):
    server: _ViewerHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(INDEX_HTML)
            elif parsed.path == "/api/index":
                self._send_json(self._index())
            elif parsed.path == "/api/run":
                self._send_json(self._run(parse_qs(parsed.query)))
            elif parsed.path == "/api/events":
                self._send_json(self._events(parse_qs(parsed.query)))
            else:
                raise _RequestError(HTTPStatus.NOT_FOUND, "not found")
        except _RequestError as exc:
            self._send_json({"error": str(exc)}, exc.status)
        except (OSError, ValueError) as exc:
            self._send_json(
                {"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR
            )

    def _index(self) -> dict[str, Any]:
        state = self.server.viewer_state
        return {
            "root": str(state.root),
            "experiments": [
                _experiment_index(state.root, experiment) for experiment in state.experiment_dirs()
            ],
        }

    def _run(self, params: dict[str, list[str]]) -> dict[str, Any]:
        state = self.server.viewer_state
        experiment = state.resolve_experiment(_first(params, "experiment"))
        run_id = _first(params, "run")
        artifact_dir = state.resolve_run_artifact_dir(experiment, run_id)
        return _run_detail(state, experiment, artifact_dir, str(run_id))

    def _events(self, params: dict[str, list[str]]) -> dict[str, Any]:
        state = self.server.viewer_state
        experiment = state.resolve_experiment(_first(params, "experiment"))
        artifact_dir = state.resolve_run_artifact_dir(experiment, _first(params, "run"))
        return _query_events(artifact_dir, params)

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def make_viewer_server(root: Path, port: int) -> ThreadingHTTPServer:
    """Build a loopback-only viewer server over the given experiment or runs-tree root."""
    return _ViewerHTTPServer(("127.0.0.1", port), _ViewerState(root))


def serve_viewer(root: Path, port: int) -> None:
    """Print the viewer URL and serve read-only run traces until interrupted."""
    server = make_viewer_server(root, port)
    actual_port = server.server_address[1]
    print(f"Serving {root.resolve()} at http://127.0.0.1:{actual_port}/ (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mindvirus trace viewer</title>
<style>
:root {
  --border: #d8dce3; --bg: #f6f7f9; --card: #ffffff; --ink: #1d2129; --muted: #6b7280;
  --amber: #b45309; --amber-bg: #fef3c7; --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: var(--ink); background: var(--bg); font-size: 14px; }
header { background: #111827; color: #f9fafb; padding: 10px 16px; }
header .title { font-weight: 650; font-size: 15px; }
header .root { color: #9ca3af; font-family: var(--mono); font-size: 12px; word-break: break-all; }
#mock-banner { background: var(--amber-bg); color: var(--amber); border-bottom: 1px solid #f59e0b; padding: 8px 16px; font-weight: 700; text-align: center; }
#layout { display: flex; align-items: stretch; min-height: calc(100vh - 64px); }
#sidebar { width: 330px; flex: 0 0 330px; border-right: 1px solid var(--border); background: #fff; overflow-y: auto; padding: 8px; }
.exp-head { padding: 8px 6px 4px; font-size: 13px; border-top: 1px solid var(--border); margin-top: 6px; }
.exp-head:first-child { border-top: none; margin-top: 0; }
.run-row { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; padding: 5px 6px; border-radius: 6px; cursor: pointer; font-size: 12.5px; }
.run-row:hover { background: #eef2f7; }
.run-row.active { background: #dbe7ff; }
.badge { display: inline-block; padding: 1px 7px; border-radius: 999px; font-size: 11px; font-weight: 650; }
.cond-population_goal { background: #fde2e1; color: #b3261e; }
.cond-personal_preference { background: #dbe7ff; color: #1d4ed8; }
.cond-no_goal { background: #e5e7eb; color: #4b5563; }
.cond-explicit_propagation { background: #eadcff; color: #7c3aed; }
.badge.model { background: #e5e7eb; color: #374151; font-family: var(--mono); }
.badge.model.mock { background: var(--amber-bg); color: var(--amber); }
.ok { color: #15803d; font-weight: 700; }
.bad { color: #b3261e; font-weight: 700; }
.muted { color: var(--muted); }
main { flex: 1; padding: 12px 18px; overflow-y: auto; min-width: 0; }
#run-header h2 { margin: 4px 0 8px; font-size: 16px; font-family: var(--mono); word-break: break-all; }
#tabs { display: flex; gap: 4px; border-bottom: 1px solid var(--border); margin-bottom: 12px; }
#tabs button { border: none; background: none; padding: 8px 14px; cursor: pointer; font-size: 14px; color: var(--muted); border-bottom: 2px solid transparent; }
#tabs button.active { color: var(--ink); border-bottom-color: #1d4ed8; font-weight: 650; }
.tab { display: none; }
.tab.active { display: block; }
.metric-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(175px, 1fr)); gap: 8px; margin: 10px 0; }
.metric { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; }
.metric-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.metric-value { font-size: 15px; font-weight: 600; word-break: break-word; }
details { background: var(--card); border: 1px solid var(--border); border-radius: 8px; margin: 8px 0; padding: 8px 10px; }
details > summary { cursor: pointer; font-weight: 600; }
pre { white-space: pre-wrap; word-break: break-word; font-family: var(--mono); font-size: 12px; background: #f3f4f6; border-radius: 6px; padding: 8px; margin: 6px 0; }
h3 { margin: 14px 0 6px; font-size: 14px; }
.adj-row { font-family: var(--mono); font-size: 12.5px; padding: 2px 0; }
.adj-row.origin { color: #b45309; font-weight: 700; }
.adj-row.bridge { color: #7c3aed; font-weight: 700; }
#filter-bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; margin-bottom: 10px; }
#filter-bar select, #filter-bar input { padding: 4px 6px; border: 1px solid var(--border); border-radius: 6px; font-size: 13px; }
#filter-bar input[type=number] { width: 72px; }
button.action { padding: 5px 12px; border: 1px solid #1d4ed8; border-radius: 6px; background: #1d4ed8; color: #fff; cursor: pointer; font-size: 13px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; margin: 8px 0; }
.card-head { display: flex; flex-wrap: wrap; gap: 10px; align-items: baseline; margin-bottom: 4px; }
.card-head .kind { font-weight: 700; font-size: 12px; text-transform: uppercase; letter-spacing: .03em; color: #374151; }
.card-head .agent { font-family: var(--mono); font-size: 12px; color: #1d4ed8; }
.card-head .meta { font-family: var(--mono); font-size: 11.5px; color: var(--muted); }
.bubble { border-radius: 10px; padding: 6px 10px; margin: 6px 0; max-width: 920px; }
.bubble-role { font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--muted); }
.role-user { background: #eef2ff; }
.role-assistant { background: #ecfdf3; }
.role-tool { background: #fff7ed; }
.bubble pre { background: rgba(255, 255, 255, .55); }
.tool-call-inline { font-family: var(--mono); font-size: 12px; color: #6d28d9; margin-top: 4px; word-break: break-word; }
.card.warning { background: #fffbeb; border-color: #f59e0b; }
.card.error-card { background: #fef2f2; border-color: #ef4444; }
.is-error pre { background: #fef2f2; }
.show-more { border: none; background: none; color: #1d4ed8; cursor: pointer; padding: 0; font-size: 12px; }
.chip { border: 1px solid var(--border); background: #fff; border-radius: 999px; padding: 4px 12px; margin: 3px; cursor: pointer; font-family: var(--mono); font-size: 12.5px; }
.chip.active { background: #1d4ed8; color: #fff; border-color: #1d4ed8; }
table { border-collapse: collapse; width: 100%; background: var(--card); }
th, td { border: 1px solid var(--border); padding: 5px 8px; text-align: left; font-size: 12.5px; vertical-align: top; }
th { background: #f3f4f6; }
td.body-cell { max-width: 520px; word-break: break-word; }
tr.undelivered { background: #fef2f2; }
.flag { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 11px; margin: 2px 4px 2px 0; }
.flag.on { background: #dcfce7; color: #15803d; }
.flag.off { background: #f3f4f6; color: var(--muted); }
.score { font-weight: 700; }
#events-status { color: var(--muted); font-size: 12.5px; margin: 6px 0; }
</style>
</head>
<body>
<header>
  <div class="title">mindvirus trace viewer</div>
  <div class="root" id="root-path"></div>
</header>
<div id="mock-banner" hidden>MOCK FIXTURE DATA — not empirical evidence about language models</div>
<div id="layout">
  <aside id="sidebar"></aside>
  <main>
    <div id="run-header" class="muted">Select a run on the left.</div>
    <nav id="tabs" hidden>
      <button data-tab="overview" class="active">Overview</button>
      <button data-tab="timeline">Timeline</button>
      <button data-tab="agents">Agents</button>
      <button data-tab="messages">Messages</button>
      <button data-tab="judge">Judge</button>
    </nav>
    <section class="tab active" id="tab-overview"></section>
    <section class="tab" id="tab-timeline">
      <div id="filter-bar">
        <select id="filter-agent"><option value="">all agents</option></select>
        <select id="filter-kind">
          <option value="">all kinds</option>
          <option>model_request</option>
          <option>model_response</option>
          <option>tool_call</option>
          <option>tool_result</option>
          <option>message_sent</option>
          <option>message_delivered</option>
          <option>memory_write</option>
          <option>document_write</option>
          <option>final_memory</option>
          <option>context_reset</option>
          <option>task_claimed</option>
          <option>task_completed</option>
          <option>error</option>
        </select>
        <label>rounds
          <input type="number" id="filter-round-from" min="0" placeholder="from">
          &ndash;
          <input type="number" id="filter-round-to" min="0" placeholder="to">
        </label>
        <input type="search" id="filter-q" placeholder="search text" size="24">
        <button class="action" id="filter-apply">Apply</button>
      </div>
      <div id="events-status"></div>
      <div id="events"></div>
      <button class="action" id="events-more" hidden>Load more</button>
    </section>
    <section class="tab" id="tab-agents"></section>
    <section class="tab" id="tab-messages"></section>
    <section class="tab" id="tab-judge"></section>
  </main>
</div>
<script>
"use strict";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => (
  {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]
));

let INDEX = null;
let CURRENT = null;
const EVENTS = {offset: 0, limit: 200, total: 0};

async function api(path) {
  const response = await fetch(path);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function longText(text, limit = 600) {
  const wrap = document.createElement("div");
  const pre = document.createElement("pre");
  const value = String(text ?? "");
  wrap.appendChild(pre);
  if (value.length > limit) {
    pre.textContent = value.slice(0, limit) + " …";
    const button = document.createElement("button");
    button.className = "show-more";
    button.textContent = "show more (" + value.length + " chars)";
    button.onclick = () => { pre.textContent = value; button.remove(); };
    wrap.appendChild(button);
  } else {
    pre.textContent = value;
  }
  return wrap;
}

function metric(label, value) {
  return '<div class="metric"><div class="metric-label">' + esc(label) +
    '</div><div class="metric-value">' + esc(value) + "</div></div>";
}

async function init() {
  INDEX = await api("/api/index");
  $("root-path").textContent = INDEX.root;
  const runs = INDEX.experiments.flatMap((e) => e.runs);
  if (runs.length > 0 && runs.every((r) => String(r.model || "").startsWith("mock/"))) {
    $("mock-banner").hidden = false;
  }
  renderSidebar();
}

function renderSidebar() {
  const sidebar = $("sidebar");
  sidebar.innerHTML = "";
  for (const exp of INDEX.experiments) {
    const head = document.createElement("div");
    head.className = "exp-head";
    head.innerHTML = "<strong>" + esc(exp.experiment_id) + "</strong> " +
      '<span class="muted">' + esc(exp.protocol_version) + " · " +
      esc(exp.config_fingerprint) + " · " + exp.runs.length + " runs</span>";
    sidebar.appendChild(head);
    for (const run of exp.runs) {
      const row = document.createElement("div");
      row.className = "run-row";
      row.dataset.run = run.run_id;
      const eligible = run.primary_endpoint_eligible;
      const outcome = eligible ? (run.spontaneous_creation_success ? "✓" : "✗") : "&ndash;";
      const outcomeCls = eligible ? (run.spontaneous_creation_success ? "ok" : "bad") : "muted";
      const isMock = String(run.model || "").startsWith("mock/");
      row.innerHTML =
        '<span class="badge cond-' + esc(run.condition) + '">' + esc(run.condition) + "</span>" +
        '<span>' + esc(run.goal_id) + "</span>" +
        '<span class="muted">' + esc(run.topology) + "</span>" +
        (run.defense && run.defense !== "none" ? '<span class="muted">def:' + esc(run.defense) + "</span>" : "") +
        '<span class="' + outcomeCls + '">' + outcome + "</span>" +
        (run.replicate !== null && run.replicate !== undefined
          ? '<span class="muted">r' + esc(run.replicate) + "</span>" : "") +
        '<span class="badge model' + (isMock ? " mock" : "") + '">' + esc(run.model) + "</span>" +
        (run.completed ? "" : '<span class="bad">failed</span>');
      row.title = run.run_id;
      row.onclick = () => selectRun(exp.path, run.run_id);
      sidebar.appendChild(row);
    }
  }
}

function showTab(name) {
  document.querySelectorAll("#tabs button").forEach((b) =>
    b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((s) =>
    s.classList.toggle("active", s.id === "tab-" + name));
}

async function selectRun(expPath, runId) {
  const detail = await api("/api/run?experiment=" + encodeURIComponent(expPath) +
    "&run=" + encodeURIComponent(runId));
  CURRENT = {experimentPath: expPath, runId: runId, detail: detail};
  document.querySelectorAll(".run-row").forEach((r) =>
    r.classList.toggle("active", r.dataset.run === runId));
  $("run-header").innerHTML = "<h2>" + esc(runId) + "</h2>";
  $("tabs").hidden = false;
  setupFilters();
  renderOverview();
  renderAgents();
  renderMessages();
  renderJudge();
  EVENTS.offset = 0;
  loadEvents(false);
  showTab("overview");
}

function setupFilters() {
  const prompts = ((CURRENT.detail.manifest || {}).system_prompts) || {};
  const agents = Object.keys(prompts).sort();
  $("filter-agent").innerHTML = '<option value="">all agents</option>' +
    agents.map((a) => '<option value="' + esc(a) + '">' + esc(a) + "</option>").join("");
  $("filter-kind").value = "";
  $("filter-round-from").value = "";
  $("filter-round-to").value = "";
  $("filter-q").value = "";
}

function renderOverview() {
  const s = CURRENT.detail.summary || {};
  const manifest = CURRENT.detail.manifest || {};
  const topo = manifest.topology || {};
  const goal = manifest.goal || {};
  const container = $("tab-overview");
  container.innerHTML = "";
  const grid = document.createElement("div");
  grid.className = "metric-grid";
  const cost = s.estimated_cost_usd === null || s.estimated_cost_usd === undefined
    ? "n/a" : "$" + Number(s.estimated_cost_usd).toFixed(4);
  grid.innerHTML =
    metric("condition", s.condition) + metric("defense", s.defense) +
    metric("topology", s.topology) + metric("model", s.model) +
    metric("seed", s.seed) + metric("completed", s.completed) +
    metric("endpoint eligible", s.primary_endpoint_eligible) +
    metric("spontaneous success", s.spontaneous_creation_success) +
    metric("infections", s.infection_count) +
    metric("distance-2 infections", s.distance_two_infection_count) +
    metric("non-DM-path infections", s.non_dm_path_infection_count) +
    metric("messages", String(s.total_messages) + " (" + String(s.messages_undelivered) + " undelivered)") +
    metric("task completion", (100 * (s.task_completion_rate || 0)).toFixed(0) + "%") +
    metric("model calls", s.total_model_calls) +
    metric("tokens", String(s.total_input_tokens) + " in / " + String(s.total_output_tokens) + " out") +
    metric("est. cost", cost) +
    (s.error ? metric("error", s.error) : "");
  container.appendChild(grid);

  const graphTitle = document.createElement("h3");
  graphTitle.textContent = "Topology (★ origin, ◆ bridge)";
  container.appendChild(graphTitle);
  const graph = document.createElement("div");
  graph.className = "card";
  const adjacency = {};
  (topo.edges || []).forEach((edge) => {
    const a = edge[0], b = edge[1];
    (adjacency[a] = adjacency[a] || new Set()).add(b);
    (adjacency[b] = adjacency[b] || new Set()).add(a);
  });
  const distances = topo.distances_from_origin || {};
  const ids = Object.keys(adjacency).sort((a, b) =>
    ((distances[a] ?? 99) - (distances[b] ?? 99)) || a.localeCompare(b));
  graph.innerHTML = ids.map((id) => {
    const cls = id === topo.origin_agent_id ? "origin" : (id === topo.bridge_agent_id ? "bridge" : "");
    const mark = id === topo.origin_agent_id ? " ★" : (id === topo.bridge_agent_id ? " ◆" : "");
    const neighbors = [...adjacency[id]].sort().join(", ");
    return '<div class="adj-row ' + cls + '">' + esc(id + mark) + " → " + esc(neighbors) + "</div>";
  }).join("");
  container.appendChild(graph);

  const goalDetails = document.createElement("details");
  goalDetails.innerHTML = "<summary>Goal: " + esc(goal.label || goal.goal_id || "") + "</summary>" +
    "<h3>origin objective</h3><pre>" + esc(goal.origin_objective || "") + "</pre>" +
    "<h3>lexicon</h3><pre>" + esc(goal.lexicon || "") + "</pre>" +
    "<h3>propagation lexicon</h3><pre>" + esc(goal.propagation_lexicon || "") + "</pre>" +
    "<h3>judge rubric</h3><pre>" + esc(goal.judge_rubric || "") + "</pre>";
  container.appendChild(goalDetails);

  const promptsTitle = document.createElement("h3");
  promptsTitle.textContent = "System prompts";
  container.appendChild(promptsTitle);
  const prompts = manifest.system_prompts || {};
  Object.keys(prompts).sort().forEach((id) => {
    const details = document.createElement("details");
    const role = id === topo.origin_agent_id ? " (origin ★)" : " (clean)";
    details.innerHTML = "<summary>" + esc(id + role) + "</summary>";
    details.appendChild(longText(prompts[id], 1200));
    container.appendChild(details);
  });
}

function messageBubble(m) {
  const div = document.createElement("div");
  div.className = "bubble role-" + (m.role || "unknown");
  div.innerHTML = '<div class="bubble-role">' + esc(m.role || "") +
    (m.name ? " · " + esc(m.name) : "") + "</div>";
  if (m.tool_call_id) {
    const ref = document.createElement("div");
    ref.className = "meta";
    ref.textContent = "tool_call_id: " + m.tool_call_id;
    div.appendChild(ref);
  }
  if (m.content) div.appendChild(longText(m.content, 900));
  (m.tool_calls || []).forEach((tc) => {
    const call = document.createElement("div");
    call.className = "tool-call-inline";
    call.textContent = "→ " + tc.name + "(" + JSON.stringify(tc.arguments) + ")";
    div.appendChild(call);
  });
  return div;
}

function modelRequestBody(payload) {
  const request = payload.request || {};
  const wrap = document.createElement("div");
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = "call_id " + (payload.call_id || request.call_id || "?") +
    " · phase " + (payload.phase || "?") +
    " · model " + (payload.model || request.model || "?");
  wrap.appendChild(meta);
  if (request.system_prompt) {
    const sys = document.createElement("details");
    sys.innerHTML = "<summary>system prompt</summary>";
    sys.appendChild(longText(request.system_prompt, 1200));
    wrap.appendChild(sys);
  }
  (request.messages || []).forEach((m) => wrap.appendChild(messageBubble(m)));
  if (request.tools) {
    const tools = document.createElement("details");
    tools.innerHTML = "<summary>tools (" + request.tools.length + ")</summary>";
    tools.appendChild(longText(JSON.stringify(request.tools, null, 2), 1200));
    wrap.appendChild(tools);
  }
  return wrap;
}

function modelResponseBody(payload) {
  const response = payload.response || {};
  const wrap = document.createElement("div");
  const meta = document.createElement("div");
  meta.className = "meta";
  const usage = response.usage || {};
  meta.textContent = "call_id " + (payload.call_id || "?") + " · phase " + (payload.phase || "?") +
    " · usage in " + String(usage.input_tokens ?? "?") + " / out " + String(usage.output_tokens ?? "?") +
    (usage.calculated_cost_usd != null ? " · $" + Number(usage.calculated_cost_usd).toFixed(6) : "");
  wrap.appendChild(meta);
  if (response.content) wrap.appendChild(longText(response.content, 900));
  (response.tool_calls || []).forEach((tc) => {
    const call = document.createElement("div");
    call.className = "tool-call-inline";
    call.textContent = "→ " + tc.name + "(" + JSON.stringify(tc.arguments) + ")";
    wrap.appendChild(call);
  });
  return wrap;
}

function eventCard(ev) {
  const payload = ev.payload || {};
  const card = document.createElement("div");
  card.className = "card kind-" + ev.kind;
  card.innerHTML = '<div class="card-head"><span class="kind">' + esc(ev.kind) + "</span>" +
    (ev.agent_id ? '<span class="agent">' + esc(ev.agent_id) + "</span>" : "") +
    '<span class="meta">round ' + esc(ev.round_index) + " · turn " + esc(ev.turn_index) +
    " · " + esc(ev.timestamp) + "</span></div>";
  const body = document.createElement("div");
  switch (ev.kind) {
    case "model_request":
      body.appendChild(modelRequestBody(payload));
      break;
    case "model_response":
      body.appendChild(modelResponseBody(payload));
      break;
    case "tool_call":
      body.appendChild(longText(payload.name + "(" + JSON.stringify(payload.arguments, null, 2) + ")"));
      break;
    case "tool_result": {
      if (payload.is_error) body.className = "is-error";
      const label = document.createElement("div");
      label.className = "meta";
      label.textContent = (payload.tool_name || "?") + " · tool_call_id " + (payload.tool_call_id || "?") +
        (payload.is_error ? " · ERROR" : "");
      body.appendChild(label);
      body.appendChild(longText(payload.result ?? "", 900));
      break;
    }
    case "message_sent":
    case "message_delivered": {
      const head = document.createElement("div");
      head.innerHTML = "<strong>" + esc(payload.sender) + " → " + esc(payload.recipient) + "</strong>" +
        '<span class="meta"> sent round ' + esc(payload.sent_round) +
        (payload.delivered_round !== null && payload.delivered_round !== undefined
          ? " · delivered round " + esc(payload.delivered_round) : " · not yet delivered") + "</span>";
      body.appendChild(head);
      body.appendChild(longText(payload.body || "", 900));
      break;
    }
    case "memory_write":
    case "document_write":
    case "final_memory": {
      const label = document.createElement("div");
      label.className = "meta";
      label.textContent = payload.name || payload.operation || "";
      body.appendChild(label);
      body.appendChild(longText(payload.content ?? payload.memory ?? "", 900));
      break;
    }
    case "context_reset": {
      card.classList.add("warning");
      const note = document.createElement("div");
      note.innerHTML = "<strong>context reset</strong> — discarded inbox messages: " +
        esc(payload.discarded_inbox_messages ?? 0);
      body.appendChild(note);
      body.appendChild(longText(payload.memory_after_checkpoint ?? "", 900));
      break;
    }
    case "task_claimed":
      body.innerHTML = "<strong>" + esc(payload.task_id) + "</strong> " + esc(payload.title || "") +
        '<div class="meta">' + esc(payload.instructions || "") + "</div>";
      break;
    case "task_completed":
      body.innerHTML = "<strong>" + esc(payload.task_id) + "</strong> completed" +
        '<div class="meta">evidence: ' + esc((payload.evidence_document_ids || []).join(", ")) + "</div>";
      body.appendChild(longText(payload.content ?? "", 600));
      break;
    case "error":
      card.classList.add("error-card");
      body.appendChild(longText(JSON.stringify(payload, null, 2), 900));
      break;
    default:
      body.appendChild(longText(JSON.stringify(payload, null, 2), 900));
  }
  card.appendChild(body);
  return card;
}

function eventsParams() {
  const params = new URLSearchParams({experiment: CURRENT.experimentPath, run: CURRENT.runId});
  const agent = $("filter-agent").value, kind = $("filter-kind").value, q = $("filter-q").value;
  const roundFrom = $("filter-round-from").value, roundTo = $("filter-round-to").value;
  if (agent) params.set("agent_id", agent);
  if (kind) params.set("kind", kind);
  if (q) params.set("q", q);
  if (roundFrom !== "") params.set("round_from", roundFrom);
  if (roundTo !== "") params.set("round_to", roundTo);
  return params;
}

async function loadEvents(append) {
  if (!CURRENT) return;
  const params = eventsParams();
  params.set("offset", append ? String(EVENTS.offset) : "0");
  params.set("limit", String(EVENTS.limit));
  const data = await api("/api/events?" + params.toString());
  EVENTS.total = data.total;
  EVENTS.offset = data.offset + data.events.length;
  const container = $("events");
  if (!append) container.innerHTML = "";
  data.events.forEach((ev) => container.appendChild(eventCard(ev)));
  $("events-status").textContent = "showing " + container.childElementCount +
    " of " + data.total + " matching events";
  $("events-more").hidden = container.childElementCount >= data.total;
}

function judgmentBlock(j) {
  const div = document.createElement("div");
  div.className = "card judgment";
  const flags = ["advocacy", "persistent", "propagation_attempt", "dm_path_exposure"]
    .map((f) => '<span class="flag ' + (j[f] ? "on" : "off") + '">' +
      esc(f.replace(/_/g, " ")) + ": " + (j[f] ? "yes" : "no") + "</span>").join("");
  div.innerHTML = '<div class="card-head"><span class="kind">' + esc(j.judge_id || "judge") +
    '</span><span class="score">score ' + esc(j.adoption_score) + "</span></div>" +
    '<div class="flags">' + flags + "</div>" +
    (j.rationale ? "<p>" + esc(j.rationale) + "</p>" : "");
  (j.evidence || []).forEach((item) => div.appendChild(longText(item, 400)));
  return div;
}

function renderAgents() {
  const container = $("tab-agents");
  container.innerHTML = "";
  const chips = document.createElement("div");
  chips.id = "agent-chips";
  const panel = document.createElement("div");
  panel.id = "agent-panel";
  container.appendChild(chips);
  container.appendChild(panel);
  const manifest = CURRENT.detail.manifest || {};
  const origin = (manifest.topology || {}).origin_agent_id;
  const bridge = (manifest.topology || {}).bridge_agent_id;
  const ids = [...new Set([
    ...Object.keys(manifest.system_prompts || {}),
    ...(CURRENT.detail.agent_snapshots || []).map((s) => s.agent_id),
  ])].sort();
  ids.forEach((id) => {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.dataset.agent = id;
    chip.textContent = id + (id === origin ? " ★" : "") + (id === bridge ? " ◆" : "");
    chip.onclick = () => selectAgent(id);
    chips.appendChild(chip);
  });
}

function selectAgent(id) {
  document.querySelectorAll(".chip").forEach((c) =>
    c.classList.toggle("active", c.dataset.agent === id));
  $("filter-agent").value = id;
  EVENTS.offset = 0;
  loadEvents(false);
  const panel = $("agent-panel");
  panel.innerHTML = "<h3>" + esc(id) + " <span class='muted'>(timeline filtered to this agent)</span></h3>";
  const snap = (CURRENT.detail.agent_snapshots || []).find((s) => s.agent_id === id);
  if (snap) {
    const grid = document.createElement("div");
    grid.className = "metric-grid";
    grid.innerHTML = metric("kind", snap.kind) +
      metric("messages sent", snap.messages_sent) +
      metric("messages received", snap.messages_received) +
      metric("context resets", snap.context_resets) +
      metric("claimed tasks", (snap.claimed_tasks || []).join(", ") || "none");
    panel.appendChild(grid);
    const title = document.createElement("h3");
    title.textContent = "final memory";
    panel.appendChild(title);
    panel.appendChild(longText(snap.memory || "", 1200));
  }
  const judgment = ((CURRENT.detail.summary || {}).agent_judgments || [])
    .find((j) => j.agent_id === id);
  if (judgment) {
    const title = document.createElement("h3");
    title.textContent = "final judgment (ensemble)";
    panel.appendChild(title);
    panel.appendChild(judgmentBlock(judgment));
  }
}

function renderMessages() {
  const container = $("tab-messages");
  const messages = ((CURRENT.detail.environment || {}).messages) || [];
  if (!messages.length) {
    container.innerHTML = '<p class="muted">No direct messages in this run.</p>';
    return;
  }
  const rows = messages.map((m) => {
    const undelivered = m.delivered_round === null || m.delivered_round === undefined;
    return '<tr class="' + (undelivered ? "undelivered" : "") + '"><td>' + esc(m.sender) +
      "</td><td>" + esc(m.recipient) + "</td><td>" + esc(m.sent_round) + "</td><td>" +
      (undelivered ? "never" : esc(m.delivered_round)) + "</td>" +
      '<td class="body-cell">' + esc(m.body) + "</td></tr>";
  }).join("");
  container.innerHTML = "<table><thead><tr><th>sender</th><th>recipient</th><th>sent round</th>" +
    "<th>delivered round</th><th>body</th></tr></thead><tbody>" + rows + "</tbody></table>";
}

async function renderJudge() {
  const container = $("tab-judge");
  container.innerHTML = "";
  const outputs = CURRENT.detail.judge_outputs;
  if (outputs) {
    Object.keys(outputs).sort().forEach((agent) => {
      const title = document.createElement("h3");
      title.textContent = agent;
      container.appendChild(title);
      (outputs[agent] || []).forEach((j) => container.appendChild(judgmentBlock(j)));
    });
  } else {
    container.innerHTML = '<p class="muted">No judge_outputs.json for this run.</p>';
  }
  const params = new URLSearchParams({experiment: CURRENT.experimentPath, run: CURRENT.runId,
    kind: "model_request", q: "judge", limit: "1000"});
  const data = await api("/api/events?" + params.toString());
  const judgeRequests = data.events.filter((ev) => (ev.payload || {}).phase === "judge");
  if (judgeRequests.length) {
    const title = document.createElement("h3");
    title.textContent = "judge model requests";
    container.appendChild(title);
    judgeRequests.forEach((ev) => container.appendChild(eventCard(ev)));
  }
}

document.querySelectorAll("#tabs button").forEach((b) =>
  b.addEventListener("click", () => showTab(b.dataset.tab)));
$("filter-apply").addEventListener("click", () => { EVENTS.offset = 0; loadEvents(false); });
$("filter-q").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { EVENTS.offset = 0; loadEvents(false); }
});
$("events-more").addEventListener("click", () => loadEvents(true));
init().catch((err) => { $("run-header").textContent = "Failed to load index: " + err.message; });
</script>
</body>
</html>
"""
