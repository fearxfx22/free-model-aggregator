// main.ts — the free-model aggregator server.
//
// One server, written from the task spec. It reads the brief (ten simulated
// providers, the sealed quality table, the workload manifest), probes every
// provider's catalog in that provider's own wire, registers with the env,
// and serves the client's twenty requests one at a time.
//
// Routing is greedy and deterministic: for each request, every free model
// that meets the request's requirements is ranked by its sealed rating, ties
// broken by provider id then model id, and tried in that order until one
// answers. A refusal that says a provider's budget is spent (or that it is
// out) retires that provider for the rest of the trial; any other refusal
// just moves on to the next candidate. No model is ever tried twice for one
// request, and no paid model is ever routed to. JSON-mode requests from an
// OpenAI client never go to an Anthropic-wire provider, because that wire
// cannot carry the JSON-mode flag and its echo would lose the request.
//
// The reply is always the provider's own answer, rendered in the client's
// wire, with the provider's usage numbers copied exactly.

import http from "node:http";
import {
  buildCatalog,
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
} from "./lib/providers.ts";
import type {
  Brief,
  CatalogReport,
  Capabilities,
  ProviderDirectoryEntry,
  Requirement,
} from "./lib/providers.ts";
import {
  callProvider,
  freeModelIds,
  fromAnthropic,
  fromOpenAI,
  isExhaustedRefusal,
  renderAnthropicJson,
  renderAnthropicSse,
  renderModelList,
  renderOpenAIJson,
  renderOpenAISse,
} from "./lib/wire.ts";
import type { CanonicalRequest } from "./lib/wire.ts";

// ---------------------------------------------------------------------------
// Trial clock: the harness hands us a fixed now; outside it, use the epoch
// the simulated providers use, so every timestamp we emit is deterministic.
// ---------------------------------------------------------------------------

function trialNowMs(): number {
  const raw = process.env.MAKEFUN_TRIAL_NOW ?? "";
  if (/^\d+$/.test(raw)) return Number(raw);
  const parsed = Date.parse(raw);
  if (Number.isFinite(parsed)) return parsed;
  return 1767225600000;
}

const NOW_MS = trialNowMs();
const ENV_URL = process.env.MAKEFUN_ENV_URL ?? "";

// ---------------------------------------------------------------------------
// Brief and catalog
// ---------------------------------------------------------------------------

interface Candidate {
  provider: ProviderDirectoryEntry;
  model: string;
  rating: number;
  caps: Capabilities;
}

interface ServerState {
  brief: Brief;
  catalog: CatalogReport;
  /** Providers retired for this trial because their budget ran out. */
  exhausted: Set<string>;
  /** Free models, ranked once, reused for every request. */
  ranked: Candidate[];
  /** How many client requests have arrived (indexes the manifest). */
  arrivals: number;
}

/** Read the brief, or an empty one when there is no prompt file. */
function loadBrief(): Brief {
  const path = process.env.MAKEFUN_PROMPT_FILE ?? "";
  if (path === "") {
    return {
      directory: { schema: "", providers: [] },
      ratings: null,
      workload: null,
      promptPath: "",
    };
  }
  try {
    return readBrief(path);
  } catch {
    return {
      directory: { schema: "", providers: [] },
      ratings: null,
      workload: null,
      promptPath: path,
    };
  }
}

/** Probe every provider and build the registered catalog. */
async function loadCatalog(brief: Brief): Promise<CatalogReport> {
  const probes = await Promise.all(
    brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider)),
  );
  return buildCatalog(ENV_URL, probes, brief.ratings);
}

/**
 * Rank every free model once: catalog (what the provider actually lists)
 * crossed with the sealed ratings (what it is worth and what it can do).
 * A model with no rating row cannot be ranked, so it is left out.
 */
function rankCandidates(brief: Brief, catalog: CatalogReport): Candidate[] {
  const byProvider = new Map<string, { id: string; free: boolean; context_length: number }[]>();
  for (const p of catalog.providers) byProvider.set(p.id, p.models);
  const out: Candidate[] = [];
  for (const provider of brief.directory.providers) {
    const listed = byProvider.get(provider.id) ?? [];
    for (const model of listed) {
      if (!model.free) continue;
      const row = ratingOfModel(brief.ratings, provider.id, model.id);
      if (!row) continue;
      if (!modelFree(model, row)) continue;
      out.push({
        provider,
        model: model.id,
        rating: row.rating_milli,
        caps: capabilitiesOf(model, row),
      });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
  });
  return out;
}

/** Requirements for a request: the manifest's entry, or a best-effort read. */
function requirementsFor(state: ServerState, req: CanonicalRequest, index: number): Requirement {
  const entry = state.brief.workload?.manifest[index];
  if (entry && entry.requires) return entry.requires;
  // Fallback when the manifest is missing: read what the body shows.
  return {
    tools: req.tools.length > 0,
    vision: req.images.length > 0,
    json: req.jsonMode,
    min_context: 0,
  };
}

/** The candidates for one request, in the order they will be tried. */
function candidatesFor(state: ServerState, req: CanonicalRequest, requires: Requirement): Candidate[] {
  return state.ranked.filter((c) => {
    if (state.exhausted.has(c.provider.id)) return false;
    if (!meetsRequirements(requires, c.caps)) return false;
    // An Anthropic-wire provider cannot echo JSON mode, so a JSON-mode
    // OpenAI request must never be handed to one.
    if (req.jsonMode && c.provider.wire === "anthropic") return false;
    return true;
  });
}

// ---------------------------------------------------------------------------
// Serving one request: walk the ranked candidates until one answers
// ---------------------------------------------------------------------------

interface Served {
  model: string;
  providerId: string;
  text: string;
  inputTokens: number;
  outputTokens: number;
}

async function serveRequest(
  state: ServerState,
  req: CanonicalRequest,
): Promise<Served | null> {
  const index = state.arrivals;
  state.arrivals += 1;
  const requires = requirementsFor(state, req, index);
  const candidates = candidatesFor(state, req, requires);
  for (const cand of candidates) {
    if (state.exhausted.has(cand.provider.id)) continue;
    const res = await callProvider(ENV_URL, cand.provider, cand.model, req);
    if (res.ok) {
      return {
        model: cand.model,
        providerId: cand.provider.id,
        text: res.answer.text,
        inputTokens: res.answer.inputTokens,
        outputTokens: res.answer.outputTokens,
      };
    }
    if (isExhaustedRefusal(res.message)) {
      state.exhausted.add(cand.provider.id);
    }
    // Any other refusal is transient for this call: the next candidate
    // (another model, or another provider) may still answer.
  }
  return null;
}

// ---------------------------------------------------------------------------
// Talking to the env: register and the work poll
// ---------------------------------------------------------------------------

async function envJson(
  path: string,
  init?: { method?: string; body?: unknown },
): Promise<{ status: number; body: string }> {
  const res = await fetch(`${ENV_URL}${path}`, {
    method: init?.method ?? "GET",
    headers: { "content-type": "application/json" },
    body: init?.body === undefined ? undefined : JSON.stringify(init.body),
  });
  return { status: res.status, body: await res.text() };
}

async function register(state: ServerState, port: number): Promise<number> {
  const { status, body } = await envJson("/register", {
    method: "POST",
    body: { protocol_version: 1, port, catalog: state.catalog },
  });
  if (status < 200 || status >= 300) {
    throw new Error(`register failed (${status}): ${body.slice(0, 200)}`);
  }
  let count = 0;
  try {
    const parsed: unknown = JSON.parse(body);
    if (parsed && typeof parsed === "object" && typeof (parsed as { requests?: unknown }).requests === "number") {
      count = (parsed as { requests: number }).requests;
    }
  } catch {
    // The env answered ok without a count; the poll below is authoritative.
  }
  return count;
}

/** Poll /work every 100 ms until the env says the client is done. */
async function waitForDone(): Promise<void> {
  for (;;) {
    try {
      const { status, body } = await envJson("/work");
      if (status >= 200 && status < 300) {
        const parsed: unknown = JSON.parse(body);
        if (
          parsed &&
          typeof parsed === "object" &&
          (parsed as { done?: unknown }).done === true
        ) {
          return;
        }
      }
    } catch {
      // The env is briefly unreachable; keep polling.
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
}

// ---------------------------------------------------------------------------
// HTTP server: the three routes the client uses
// ---------------------------------------------------------------------------

const MAX_BODY_BYTES = 2 * 1024 * 1024;

function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(new Error("body too large"));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

function sendJson(res: http.ServerResponse, status: number, payload: unknown): void {
  const text = JSON.stringify(payload);
  res.writeHead(status, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(text),
  });
  res.end(text);
}

function sendSse(res: http.ServerResponse, text: string): void {
  res.writeHead(200, {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    "content-length": Buffer.byteLength(text),
  });
  res.end(text);
}

/** The wire a GET /v1/models caller speaks, from its auth header. */
function modelsWire(req: http.IncomingMessage): "openai" | "anthropic" {
  const key = req.headers["x-api-key"];
  if (typeof key === "string" && key.length > 0) return "anthropic";
  return "openai";
}

function makeHandler(state: ServerState): http.RequestListener {
  return (req, res) => {
    void (async () => {
      try {
        const url = new URL(req.url ?? "/", "http://127.0.0.1");
        if (req.method === "GET" && url.pathname === "/v1/models") {
          const ids = freeModelIds(state.catalog);
          sendJson(res, 200, renderModelList(ids, modelsWire(req), NOW_MS));
          return;
        }
        if (req.method === "POST" && url.pathname === "/v1/chat/completions") {
          const raw = await readBody(req);
          let body: unknown;
          try {
            body = JSON.parse(raw);
          } catch {
            sendJson(res, 400, { error: { message: "body must be a JSON object", type: "invalid_request_error" } });
            return;
          }
          if (!body || typeof body !== "object" || Array.isArray(body)) {
            sendJson(res, 400, { error: { message: "body must be a JSON object", type: "invalid_request_error" } });
            return;
          }
          const canon = fromOpenAI(body as Record<string, unknown>);
          const served = await serveRequest(state, canon);
          if (!served) {
            sendJson(res, 502, { error: { message: "no provider could serve this request", type: "server_error" } });
            return;
          }
          const answer = { text: served.text, inputTokens: served.inputTokens, outputTokens: served.outputTokens };
          if (canon.stream) sendSse(res, renderOpenAISse(answer, served.model, NOW_MS));
          else sendJson(res, 200, renderOpenAIJson(answer, served.model, NOW_MS));
          return;
        }
        if (req.method === "POST" && url.pathname === "/v1/messages") {
          const raw = await readBody(req);
          let body: unknown;
          try {
            body = JSON.parse(raw);
          } catch {
            sendJson(res, 400, { error: { type: "invalid_request_error", message: "body must be a JSON object" } });
            return;
          }
          if (!body || typeof body !== "object" || Array.isArray(body)) {
            sendJson(res, 400, { error: { type: "invalid_request_error", message: "body must be a JSON object" } });
            return;
          }
          const canon = fromAnthropic(body as Record<string, unknown>);
          const served = await serveRequest(state, canon);
          if (!served) {
            sendJson(res, 502, { error: { type: "overloaded_error", message: "no provider could serve this request" } });
            return;
          }
          const answer = { text: served.text, inputTokens: served.inputTokens, outputTokens: served.outputTokens };
          if (canon.stream) sendSse(res, renderAnthropicSse(answer, served.model));
          else sendJson(res, 200, renderAnthropicJson(answer, served.model));
          return;
        }
        sendJson(res, 404, { error: { message: "not found", type: "invalid_request_error" } });
      } catch (err) {
        console.error(`handler error for ${req.method} ${req.url}:`, err instanceof Error ? err.message : String(err));
        if (!res.headersSent) {
          sendJson(res, 500, {
            error: { message: err instanceof Error ? err.message : String(err), type: "server_error" },
          });
        } else {
          res.destroy();
        }
      }
    })();
  };
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

async function main(): Promise<number> {
  const args = process.argv.slice(2);
  const brief = loadBrief();
  const catalog = await loadCatalog(brief);
  if (args.includes("--list")) {
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return 0;
  }
  if (ENV_URL === "") {
    process.stderr.write("MAKEFUN_ENV_URL is not set\n");
    return 2;
  }
  const state: ServerState = {
    brief,
    catalog,
    exhausted: new Set<string>(),
    ranked: rankCandidates(brief, catalog),
    arrivals: 0,
  };
  const server = http.createServer(makeHandler(state));
  await new Promise<void>((resolve) => server.listen(0, "0.0.0.0", resolve));
  const address = server.address();
  const port = typeof address === "object" && address !== null ? address.port : 0;
  const expected = await register(state, port);
  process.stdout.write(`aggregator listening on ${port}; env expects ${expected} requests\n`);
  await waitForDone();
  server.close();
  process.stdout.write(`done after ${state.arrivals} requests\n`);
  return 0;
}

process.exitCode = await main();
