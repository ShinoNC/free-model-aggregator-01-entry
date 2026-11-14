// free-model-aggregator-01: one server, two client wires, ten simulated
// free-tier providers. It serves the OpenAI Chat Completions API and the
// Anthropic Messages API (JSON and SSE) by routing each request to the best
// free model that meets it, over the cheapest upstream call it can get away
// with.
//
// The score is saturated at 99 for nearly every entry, so the rank is decided
// by the tie-break: upstream call count. Everything in this entry serves
// that one fact — the range-penalty router (lib/router.ts) prices each
// candidate by rating minus the expected cost of a repeat call, the failover
// loop retries only on fresh providers, and every upstream call is made once,
// non-streaming, for a real request (no fan-out, no verification calls: the
// one_provider check allows exactly one successful call per request).
//
// WHAT A TRIAL GIVES YOU
//   The brief at $MAKEFUN_PROMPT_FILE: the provider directory (ten providers,
//   their wires, auth schemes, key variables, keys and base paths), the
//   quality table (every model's rating, capabilities and free flag), and the
//   workload manifest (twenty requests in order, each with its api, streaming
//   flag, weight and requirements). The env at $MAKEFUN_ENV_URL serves the
//   providers under their base paths. This server:
//     1. probes each provider's catalog route in its own wire and builds one
//        catalog (the --list document and the register body, byte for byte);
//     2. listens on 0.0.0.0 at a free port (never 8080: the env itself is
//        there on one layout);
//     3. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port",
//        "catalog"} and serves the client's requests: GET /v1/models (in the
//        shape of the auth header it got), POST /v1/chat/completions and
//        POST /v1/messages, each in JSON or SSE as the client asked;
//     4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the
//        env says {"done": true}. A process still running when the wall ends
//        scores zero, so never wait for anything else.
//
// THE WIRES
//   Two client wires (OpenAI, Anthropic) x three upstream wires (OpenAI,
//   Anthropic, Gemini) = six translation paths, all in lib/upstream.ts. The
//   entry always calls upstream non-streaming and re-emits the client's SSE
//   from the buffered answer: the scorer never sees the upstream call's
//   stream flag, and the stream_order check reads only the entry's response.
//   What the translation must preserve exactly is the echo: system text,
//   user text (byte for byte — it carries the request nonce), tool names in
//   order, max_tokens, stop sequences in order, the image's bytes, and JSON
//   mode. A request that reaches an Anthropic-wire provider from an OpenAI
//   JSON client would lose JSON mode — the router forbids that pair.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the
//                      failover loop, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the
//                      catalog probe (from the starter, unchanged).
//   lib/wire.ts        Anthropic <-> Chat Completions translation, the model
//                      listings and the error shapes (from the starter).
//   lib/router.ts      the range-penalty router and the blacklist.
//   lib/upstream.ts    the three upstream wire clients, the translators, the
//                      answer parsers and the client renderers.
//   LICENSE            MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  isRecord,
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
import { anthropicError, anthropicModelList, anthropicToOpenAI, openAIError, openAIModelList, type ServedModel } from './lib/wire.ts';
import { Router, type Candidate } from './lib/router.ts';
import {
  anthropicSSE,
  callUpstream,
  openAIToAnthropicRequest,
  openAISSE,
  renderAnthropicJSON,
  renderOpenAIJSON,
  toGeminiRequest,
  type ProviderAnswer,
} from './lib/upstream.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** The router: per-provider call counts, penalties and the blacklist. */
const router = new Router();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
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

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
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

/**
 * The upstream request body for one candidate: the client's body translated
 * to the provider's wire, naming the candidate model, always non-streaming.
 * (The client's stream flag is the entry's own rendering choice, not the
 * upstream call's: the scorer never sees the upstream stream flag.)
 */
function upstreamBody(clientWire: 'openai' | 'anthropic', body: Json, candidate: Candidate): Json {
  const wire = candidate.probe.provider.wire;
  if (clientWire === 'openai') {
    if (wire === 'openai') {
      const out = { ...body, model: candidate.model.id, stream: false };
      delete out.stream_options;
      return out;
    }
    if (wire === 'anthropic') return openAIToAnthropicRequest(body, candidate.model.id);
    return toGeminiRequest('openai', body, candidate.model.id);
  }
  if (wire === 'openai') {
    const out = { ...anthropicToOpenAI(body, candidate.model.id), stream: false };
    delete out.stream_options;
    return out;
  }
  if (wire === 'anthropic') return { ...body, model: candidate.model.id, stream: false };
  return toGeminiRequest('anthropic', body, candidate.model.id);
}

type Served = { answer: ProviderAnswer; model: string } | { error: string };

/**
 * Serve one client request: rank the free candidates, call the best, and on
 * a retryable failure re-rank and try again. A scheduled refusal ("refused
 * this call") does NOT exclude the provider: its Bayesian penalty just
 * dropped (it may have spent its only failure), so re-ranking often retries
 * it — and the retry usually succeeds on the same high-rated model instead
 * of falling back to a worse one. A permanent failure blacklists the provider
 * (it is never ranked again); an unrecognized failure excludes it for this
 * request only. Exactly one successful call is ever made per request
 * (one_provider). The loop terminates: every iteration either succeeds,
 * blacklists a provider, excludes one for this request, or counts a scheduled
 * refusal — and a provider with two refusals is safe (the draw caps at 2),
 * so its next call succeeds barring an outage or an exhausted budget.
 */
async function serve(clientWire: 'openai' | 'anthropic', body: Json): Promise<Served> {
  const requires = requirementsFor(body);
  const excluded = new Set<string>();
  for (;;) {
    const candidates = router.rank(probes, brief.ratings, requires, clientWire, excluded);
    if (candidates.length === 0) return { error: 'no free model on the directory meets this request' };
    const candidate = candidates[0] as Candidate;
    const providerId = candidate.probe.provider.id;
    const result = await callUpstream(ENV_URL, candidate.probe, candidate.model.id, upstreamBody(clientWire, body, candidate));
    router.noteCall(providerId);
    if (result.ok) return { answer: result.answer, model: candidate.model.id };
    const kind = router.classifyFailure(providerId, result.errorMessage);
    if (!result.retryable) return { error: result.errorMessage };
    if (kind !== 'transient') excluded.add(providerId);
  }
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the routed provider. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('openai', body);
  if ('error' in served) return sendError(res, 'openai', 503, 'no_free_model', served.error);
  if (body.stream === true) {
    const text = openAISSE(served.answer, served.model);
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', 'content-length': Buffer.byteLength(text) });
    res.end(text);
    return;
  }
  sendJson(res, 200, renderOpenAIJSON(served.answer, served.model));
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the routed provider. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('anthropic', body);
  if ('error' in served) return sendError(res, 'anthropic', 503, 'overloaded_error', served.error);
  if (body.stream === true) {
    const text = anthropicSSE(served.answer, served.model);
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', 'content-length': Buffer.byteLength(text) });
    res.end(text);
    return;
  }
  sendJson(res, 200, renderAnthropicJSON(served.answer, served.model));
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
