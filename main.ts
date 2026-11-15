// free-model-aggregator-01: one server in front of ten free providers.
//
// The trial gives this server a brief (the provider directory, the sealed quality table, the
// workload manifest) at $MAKEFUN_PROMPT_FILE and an env at $MAKEFUN_ENV_URL that serves the
// providers under their base paths. The server reads the brief, probes each provider's catalog
// route in the provider's own wire, builds one catalog, then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (x-api-key means Anthropic, Authorization
//      means OpenAI), POST /v1/chat/completions and POST /v1/messages, each with a JSON body or
//      with server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts.
//
// ROUTING (quality is 60% of the score, served 15%, failover 10%)
//   For each request the server builds an ordered fallback chain of every free model that
//   meets the request's requirements (tools, vision, JSON mode, context length), ranked by the
//   quality table's rating_milli, best first; models tied on rating are ordered by the trial's
//   seeded RNG, so the load spreads instead of always favouring one provider. A request that
//   asks for JSON mode on the OpenAI wire never routes to an anthropic-wire provider, whose
//   wire cannot carry response_format (its echo would say json_mode was absent).
//   The chain is tried in order, one provider call at a time. A provider that answers 402,
//   429 or a 5xx (or a 200 carrying an error) loses the request to the next candidate, and the
//   call it cost is counted. A provider whose refusal says its free-tier budget is spent, or
//   that it is out, is marked exhausted and never tried again in this trial; a provider whose
//   refusal was for a single call stays in the pool for later requests. When the next candidate
//   is far worse than the one that just failed (more than 10 rating milli-points), the failed
//   provider is retried once before moving on, because a single refused call is usually a
//   one-off while the quality gap is paid on every request that falls through.
//   A paid model is never a legal pick and never appears in a chain.
//
// WIRES (10% of the score)
//   Every provider answer is reduced to its canonical form (text plus the provider's usage
//   pair) and rendered in the client's wire; the provider's usage is passed through exactly,
//   the stop family is always 0 (stop / end_turn), and the answer text is carried whole. See
//   lib/wire.ts for the three upstream request builders and the two renderers.
//
// DETERMINISM
//   No clock and no unseeded randomness anywhere in what is sent: the only random choice is
//   the order of equally rated candidates, seeded from MAKEFUN_TRIAL_SEED through the vendored
//   SDK's trialRng. The catalog is built once and the same object is printed by --list and
//   posted to /register. The client sends one request at a time, in order, and this server
//   answers one at a time.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToOpenAI,
  clientToGemini,
  collectAnthropicStream,
  collectGeminiStream,
  collectOpenAIStream,
  errorOfAnswer,
  openAIError,
  openAIModelList,
  openAIToAnthropicRequest,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  renderAnthropicAnswer,
  renderAnthropicStream,
  renderOpenAIAnswer,
  renderOpenAIStream,
  type CanonicalAnswer,
  type ServedModel,
} from './lib/wire.ts';
import { trialRng, type Rng } from './sdk/makefun-agent.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** A retry on the same provider is only worth it when falling through would cost this many rating milli-points times the request's weight. */
const RETRY_GAP_MILLI = 10;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
/** The trial's seed: every random choice is drawn from it, so a replay draws the same. */
const TRIAL_SEED = process.env.MAKEFUN_TRIAL_SEED ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Providers whose refusal said the free-tier budget is spent: never tried again this trial. */
const exhausted = new Set<string>();
/** Calls made per provider, successes and refusals alike: the provider numbers them the same way. */
const callsMade = new Map<string, number>();
/** The seeded RNG that orders equally rated candidates. */
let routingRng: Rng | null = null;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  routingRng = trialRng({ seed: TRIAL_SEED === '' ? 'free-model-aggregator-01' : TRIAL_SEED }, 'routing');
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** One candidate for a request: a free model on a provider that has not spent its budget. */
interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * The ordered fallback chain for one request: every free model that meets the requirements,
 * best rating first. Models tied on rating are ordered by the trial's seeded RNG, so two
 * providers holding the same model rating share the load instead of one always going first.
 * When `jsonNeedsWire` is set, anthropic-wire providers are left out: their wire cannot carry
 * response_format, so a JSON-mode request routed to one would echo json_mode absent.
 */
function candidatesFor(requires: Requirement, jsonNeedsWire: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (exhausted.has(probe.provider.id)) continue;
    if (jsonNeedsWire && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row === null ? 0 : row.rating_milli });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  // Shuffle the equal-rating runs with the seeded RNG: same seed, same order, every run.
  const rng = routingRng;
  if (rng !== null) {
    let start = 0;
    while (start < out.length) {
      let end = start + 1;
      while (end < out.length && out[end]!.rating === out[start]!.rating) end += 1;
      if (end - start > 1) {
        const run = out.slice(start, end);
        rng.shuffle(run);
        for (let i = 0; i < run.length; i++) out[start + i] = run[i]!;
      }
      start = end;
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): { requires: Requirement; weight: number } {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return { requires: entry.requires, weight: entry.weight };
  return { requires: requirementsFromBody(body), weight: 1 };
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether this request carries JSON mode on the OpenAI wire (a response_format the provider must echo). */
function jsonModeOnOpenAIWire(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One failed provider attempt, for the log and for the exhaustion check. */
interface Failure {
  status: number;
  code: string;
  message: string;
}

/** Whether a refusal says the provider's free-tier budget is spent (never try it again). */
function isExhaustion(message: string): boolean {
  const low = message.toLowerCase();
  return low.includes('budget is spent') || low.includes('is out');
}

/** The upstream request for one candidate, in the candidate's own wire. */
function upstreamRequest(candidate: Candidate, clientWire: 'openai' | 'anthropic', body: Json): { url: string; payload: Json; stream: boolean } {
  const base = joinBase(ENV_URL, candidate.probe.provider.base);
  const stream = body.stream === true;
  const model = candidate.model.id;
  const wire = candidate.probe.provider.wire;
  if (wire === 'openai') {
    const payload = clientWire === 'openai' ? { ...body, model } : anthropicToOpenAI(body, model);
    if (stream) payload.stream_options = { include_usage: true };
    return { url: `${base}/v1/chat/completions`, payload, stream };
  }
  if (wire === 'anthropic') {
    const payload = clientWire === 'anthropic' ? { ...body, model } : openAIToAnthropicRequest(body, model);
    return { url: `${base}/v1/messages`, payload, stream };
  }
  const payload = clientToGemini(body, model);
  const url = stream ? `${base}/v1beta/models/${model}:streamGenerateContent?alt=sse` : `${base}/v1beta/models/${model}:generateContent`;
  return { url, payload, stream };
}

/** One provider call: the canonical answer on success, or the failure that cost the call. */
async function callCandidate(
  candidate: Candidate,
  clientWire: 'openai' | 'anthropic',
  body: Json,
): Promise<{ ok: true; answer: CanonicalAnswer } | { ok: false; failure: Failure }> {
  const providerId = candidate.probe.provider.id;
  callsMade.set(providerId, (callsMade.get(providerId) ?? 0) + 1);
  const { url, payload, stream } = upstreamRequest(candidate, clientWire, body);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(candidate.probe.provider.auth, candidate.probe.provider.key),
  };
  if (candidate.probe.provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (stream) headers.accept = 'text/event-stream';
  let upstream: Response;
  try {
    upstream = await fetch(url, { method: 'POST', headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS) });
  } catch (err) {
    return { ok: false, failure: { status: 0, code: 'upstream_unreachable', message: (err as Error).message } };
  }
  if (!upstream.ok) {
    const text = await upstream.text().catch(() => '');
    let code = '';
    let message = text;
    try {
      const parsed: unknown = JSON.parse(text);
      if (isRecord(parsed) && isRecord(parsed.error)) {
        const e = parsed.error;
        code = typeof e.code === 'string' ? e.code : typeof e.type === 'string' ? e.type : typeof e.status === 'string' ? e.status : '';
        message = typeof e.message === 'string' ? e.message : text;
      } else if (isRecord(parsed) && typeof parsed.message === 'string') {
        message = parsed.message;
      }
    } catch {
      // The error body is not JSON: keep the raw text as the message.
    }
    if (isExhaustion(message)) exhausted.add(providerId);
    return { ok: false, failure: { status: upstream.status, code, message } };
  }
  const wire = candidate.probe.provider.wire;
  if (stream) {
    if (upstream.body === null) return { ok: false, failure: { status: 502, code: 'empty_stream', message: 'the provider answered with no stream' } };
    try {
      if (wire === 'openai') return { ok: true, answer: await collectOpenAIStream(upstream.body) };
      if (wire === 'anthropic') return { ok: true, answer: await collectAnthropicStream(upstream.body) };
      return { ok: true, answer: await collectGeminiStream(upstream.body) };
    } catch (err) {
      return { ok: false, failure: { status: 502, code: 'stream_error', message: (err as Error).message } };
    }
  }
  let parsed: unknown;
  try {
    parsed = await upstream.json();
  } catch {
    return { ok: false, failure: { status: 502, code: 'bad_json', message: 'the provider answered a body that is not JSON' } };
  }
  const disguised = errorOfAnswer(parsed);
  if (disguised !== null) {
    if (isExhaustion(disguised.message)) exhausted.add(providerId);
    return { ok: false, failure: { status: 200, code: disguised.code, message: disguised.message } };
  }
  if (!isRecord(parsed)) return { ok: false, failure: { status: 502, code: 'bad_json', message: 'the provider answered a JSON value that is not an object' } };
  if (wire === 'openai') return { ok: true, answer: parseOpenAIAnswer(parsed) };
  if (wire === 'anthropic') return { ok: true, answer: parseAnthropicAnswer(parsed) };
  return { ok: true, answer: parseGeminiAnswer(parsed) };
}

/**
 * Serve one client request through its fallback chain. Returns the canonical answer on
 * success; on total failure, null, and the caller answers 503. A candidate that fails for a
 * single call is retried once when falling through to the next candidate would cost more
 * than 10 rating milli-points times the request's weight; a provider that is out of budget
 * is never retried.
 */
async function serve(
  clientWire: 'openai' | 'anthropic',
  body: Json,
): Promise<{ answer: CanonicalAnswer; model: string } | null> {
  const { requires, weight } = requirementsFor(body);
  const jsonNeedsWire = clientWire === 'openai' && jsonModeOnOpenAIWire(body);
  const chain = candidatesFor(requires, jsonNeedsWire);
  let retried = false;
  for (let i = 0; i < chain.length; i++) {
    const candidate = chain[i]!;
    const result = await callCandidate(candidate, clientWire, body);
    if (result.ok) return { answer: result.answer, model: candidate.model.id };
    const next = chain[i + 1];
    const gap = next === undefined ? 0 : (candidate.rating - next.rating) * weight;
    if (!retried && !exhausted.has(candidate.probe.provider.id) && gap > RETRY_GAP_MILLI) {
      retried = true;
      const again = await callCandidate(candidate, clientWire, body);
      if (again.ok) return { answer: again.answer, model: candidate.model.id };
    }
    log(`request: ${candidate.probe.provider.id}/${candidate.model.id} failed HTTP ${result.failure.status} (${result.failure.code}): ${result.failure.message.slice(0, 120)}`);
  }
  return null;
}

/** Answer one client request in its own wire, from the canonical answer. */
function answerClient(
  res: ServerResponse,
  clientWire: 'openai' | 'anthropic',
  body: Json,
  served: { answer: CanonicalAnswer; model: string },
): void {
  const stream = body.stream === true;
  if (clientWire === 'openai') {
    if (stream) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      for (const event of renderOpenAIStream(served.answer, served.model)) res.write(event);
      res.end();
      return;
    }
    sendJson(res, 200, renderOpenAIAnswer(served.answer, served.model));
    return;
  }
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    for (const event of renderAnthropicStream(served.answer, served.model)) res.write(event);
    res.end();
    return;
  }
  sendJson(res, 200, renderAnthropicAnswer(served.answer, served.model));
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the fallback chain. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('openai', body);
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory could serve this request');
  answerClient(res, 'openai', body, served);
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the fallback chain. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('anthropic', body);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory could serve this request');
  answerClient(res, 'anthropic', body, served);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
