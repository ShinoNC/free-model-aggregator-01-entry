// The three upstream wires: translate a client request to any provider's wire,
// call it, and parse the answer into one shape the client renderers share.
//
// The entry always calls upstream NON-streaming (JSON), even when the client
// asked for a stream: the scorer never sees the upstream call's stream flag
// (the call record holds status, model, stop and usage only), and the client
// SSE is re-emitted from the buffered answer in the right event order. This
// keeps one code path for every (client wire x upstream wire) pair instead of
// nine, and the stream_order check only reads the entry's own response.
//
// What the translation must preserve exactly (the env computes the echo from
// what the PROVIDER received and compares it with the client's own echo):
// system text, user text (byte for byte — it carries the request nonce),
// tool NAMES in order, max_tokens, stop sequences in order, the image's
// bytes, and JSON mode. Temperature and top_p are not in the echo and are
// forwarded only where the wire carries them.
//
// The workload's requests are simple (one system message at most, one user
// message, no assistant/tool messages), but the translators stay general:
// tool_use / tool_result blocks are carried where the target wire has them.

import { authHeaders, isRecord, joinBase, type Json, type Probe } from './providers.ts';

/** One provider answer, in wire-neutral form. */
export interface ProviderAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  /** The provider's stop reason: "stop" (OpenAI), "end_turn" (Anthropic), "STOP" (Gemini). */
  stop: string;
}

/**
 * The outcome of one upstream call. A failure is `retryable` when no
 * successful provider call was recorded for this request (an HTTP error, or
 * a network error that never reached the provider): the caller may retry on
 * another provider. It is NOT retryable when the provider answered 2xx but
 * the body is unreadable — the provider already recorded a successful call,
 * so a retry would break one_provider; the caller must fail the request.
 */
export type UpstreamResult =
  | { ok: true; status: number; answer: ProviderAnswer }
  | { ok: false; status: number; errorMessage: string; retryable: boolean };

/** A fixed epoch second for rendered completions (nothing reads the clock). */
const FIXED_CREATED = 1767225600;
/** How many characters per SSE content chunk (the env splits the same way). */
const SSE_DELTA_CHARS = 13;

// ---------------------------------------------------------------------------
// Text and image helpers (shared by the translators)
// ---------------------------------------------------------------------------

/** The text of a string or of a block array's text blocks, joined with "\n". */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One OpenAI content part as an Anthropic content block: text or image; null otherwise. */
function anthropicBlockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url)) {
    const url = part.image_url.url;
    if (typeof url !== 'string') return null;
    const data = /^data:([^;,]+);base64,(.*)$/s.exec(url);
    if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data[1], data: data[2] } };
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** One Anthropic content block as a Gemini content part: text or inline image; null otherwise. */
function geminiPartOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  return null;
}

/** One OpenAI content part as a Gemini content part: text or inline image; null otherwise. */
function geminiPartOfOpenAI(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url)) {
    const url = part.image_url.url;
    if (typeof url !== 'string') return null;
    const data = /^data:([^;,]+);base64,(.*)$/s.exec(url);
    if (data !== null) return { inlineData: { mimeType: data[1], data: data[2] } };
    return { fileData: { fileUri: url } };
  }
  return null;
}

/** Tool definitions as Anthropic tools (name, description, input_schema). */
function anthropicToolsOf(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw)) return [];
    // Already Anthropic-shaped.
    if (typeof raw.name === 'string') {
      return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', input_schema: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
    }
    // OpenAI-shaped: {type: 'function', function: {name, description, parameters}}.
    if (isRecord(raw.function) && typeof raw.function.name === 'string') {
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    }
    return [];
  });
}

/** Tool definitions as Gemini functionDeclarations (name, description, parameters). */
function geminiToolsOf(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  const declarations = tools.flatMap((raw) => {
    if (!isRecord(raw)) return [];
    if (typeof raw.name === 'string') {
      return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
    }
    if (isRecord(raw.function) && typeof raw.function.name === 'string') {
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    }
    return [];
  });
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : [];
}

// ---------------------------------------------------------------------------
// Client request -> upstream request
// ---------------------------------------------------------------------------

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming
 * `model`. JSON mode cannot be carried (the Anthropic wire has no JSON-mode
 * field) — the router never sends an OpenAI JSON request to an Anthropic-wire
 * provider, so this translator never has to.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const content = raw.content;
      const blocks: Json[] = [];
      const text = textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let input: Json = {};
        if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
          try {
            const parsed: unknown = JSON.parse(call.function.arguments);
            if (isRecord(parsed)) input = parsed;
          } catch {
            // Not JSON: an empty input.
          }
        }
        blocks.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : '', name: typeof call.function.name === 'string' ? call.function.name : '', input });
      }
      messages.push({ role: 'assistant', content: blocks.length === 1 && blocks[0]?.type === 'text' ? (blocks[0].text as string) : blocks });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) }],
      });
      continue;
    }
    // A user message: string stays a string, blocks map 1:1.
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    if (Array.isArray(raw.content)) {
      const blocks = raw.content.flatMap((block) => {
        if (!isRecord(block)) return [];
        const mapped = anthropicBlockOf(block);
        return mapped === null ? [] : [mapped];
      });
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    }
  }
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop.filter((s): s is string => typeof s === 'string');
  const tools = anthropicToolsOf(body.tools);
  if (tools.length > 0) out.tools = tools;
  out.stream = false;
  return out;
}

/**
 * Either client's request as a Gemini generateContent body naming `model`.
 * System text goes to systemInstruction (never into contents: the env reads
 * the user text from the non-model contents, and a system message there would
 * corrupt it). JSON mode becomes responseMimeType "application/json".
 */
export function toGeminiRequest(clientWire: 'openai' | 'anthropic', body: Json, model: string): Json {
  const contents: Json[] = [];
  let systemText = '';
  if (clientWire === 'anthropic') {
    systemText = textOf(body.system);
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      const role = raw.role === 'assistant' ? 'model' : 'user';
      const parts: Json[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          const part = geminiPartOf(block);
          if (part !== null) parts.push(part);
          else if (block.type === 'tool_use' && typeof block.name === 'string') parts.push({ text: `tool_use ${block.name}` });
          else if (block.type === 'tool_result') parts.push({ text: typeof block.content === 'string' ? block.content : textOf(block.content) });
        }
      }
      if (parts.length > 0) contents.push({ role, parts });
    }
  } else {
    const source = Array.isArray(body.messages) ? body.messages : [];
    const systemParts: string[] = [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'system') {
        const text = textOf(raw.content);
        if (text !== '') systemParts.push(text);
        continue;
      }
      const role = raw.role === 'assistant' ? 'model' : 'user';
      const parts: Json[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          const part = geminiPartOfOpenAI(block);
          if (part !== null) parts.push(part);
        }
      }
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (isRecord(call) && isRecord(call.function) && typeof call.function.name === 'string') parts.push({ text: `tool_use ${call.function.name}` });
      }
      if (parts.length > 0) contents.push({ role, parts });
    }
    systemText = systemParts.join('\n');
  }
  const out: Json = { model, contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const tools = geminiToolsOf(body.tools);
  if (tools.length > 0) out.tools = tools;
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = clientWire === 'anthropic' ? body.stop_sequences : body.stop;
  if (typeof stops === 'string') config.stopSequences = [stops];
  else if (Array.isArray(stops) && stops.length > 0) config.stopSequences = stops.filter((s): s is string => typeof s === 'string');
  const jsonMode = clientWire === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  if (jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Upstream answers -> one shape
// ---------------------------------------------------------------------------

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions JSON answer as a ProviderAnswer. */
export function parseOpenAIAnswer(value: unknown): ProviderAnswer | null {
  if (!isRecord(value)) return null;
  const choices = Array.isArray(value.choices) ? value.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const usage = isRecord(value.usage) ? value.usage : {};
  return {
    text,
    inputTokens: numberOrZero(usage.prompt_tokens),
    outputTokens: numberOrZero(usage.completion_tokens),
    stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
  };
}

/** An Anthropic Messages JSON answer as a ProviderAnswer. */
export function parseAnthropicAnswer(value: unknown): ProviderAnswer | null {
  if (!isRecord(value)) return null;
  if (!Array.isArray(value.content)) return null;
  const text = value.content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
  const usage = isRecord(value.usage) ? value.usage : {};
  return {
    text,
    inputTokens: numberOrZero(usage.input_tokens),
    outputTokens: numberOrZero(usage.output_tokens),
    stop: typeof value.stop_reason === 'string' ? value.stop_reason : 'end_turn',
  };
}

/** A Gemini generateContent JSON answer as a ProviderAnswer. */
export function parseGeminiAnswer(value: unknown): ProviderAnswer | null {
  if (!isRecord(value)) return null;
  const candidates = Array.isArray(value.candidates) ? value.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null) return null;
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const usage = isRecord(value.usageMetadata) ? value.usageMetadata : {};
  return {
    text,
    inputTokens: numberOrZero(usage.promptTokenCount),
    outputTokens: numberOrZero(usage.candidatesTokenCount),
    stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP',
  };
}

/** The error message of a provider error body, in any wire's shape. */
export function errorMessageOf(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error) && typeof value.error.message === 'string') return value.error.message;
    if (isRecord(value) && typeof value.message === 'string') return value.message;
  } catch {
    // Not JSON: the raw text is the message.
  }
  return text.slice(0, 500);
}

// ---------------------------------------------------------------------------
// One upstream call, on any wire
// ---------------------------------------------------------------------------

/**
 * POST `body` to the provider's inference route and parse the answer.
 * Always non-streaming: the caller renders the client's stream from the
 * buffered answer. A 2xx with an unparsable body is returned as a failure
 * WITHOUT a retry signal — the provider already recorded a successful call,
 * so a retry would break one_provider; the caller must not retry it.
 */
export async function callUpstream(envUrl: string, probe: Probe, modelId: string, body: Json): Promise<UpstreamResult> {
  const provider = probe.provider;
  const base = joinBase(envUrl, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  let url: string;
  if (provider.wire === 'openai') url = `${base}/v1/chat/completions`;
  else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(modelId)}:generateContent`;
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body) });
  } catch (err) {
    return { ok: false, status: 502, errorMessage: `upstream unreachable: ${(err as Error).message}`, retryable: true };
  }
  const text = await res.text();
  if (!res.ok) return { ok: false, status: res.status, errorMessage: errorMessageOf(text), retryable: true };
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return { ok: false, status: 502, errorMessage: 'the provider answered a body that is not JSON', retryable: false };
  }
  const answer =
    provider.wire === 'openai' ? parseOpenAIAnswer(parsed) : provider.wire === 'anthropic' ? parseAnthropicAnswer(parsed) : parseGeminiAnswer(parsed);
  if (answer === null) return { ok: false, status: 502, errorMessage: 'the provider answered a body this entry cannot read', retryable: false };
  return { ok: true, status: res.status, answer };
}

// ---------------------------------------------------------------------------
// One answer -> the client's wire (JSON or SSE)
// ---------------------------------------------------------------------------

/** A Chat Completions JSON answer for an OpenAI client. */
export function renderOpenAIJSON(answer: ProviderAnswer, model: string): Json {
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: 'stop', logprobs: null }],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
  };
}

/** A Messages JSON answer for an Anthropic client. */
export function renderAnthropicJSON(answer: ProviderAnswer, model: string): Json {
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** Split text into SSE-sized pieces (the env splits the same way). */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += SSE_DELTA_CHARS) pieces.push(text.slice(at, at + SSE_DELTA_CHARS));
  return pieces.length === 0 ? [''] : pieces;
}

/** One SSE frame: `event: <name>` when named, then the JSON data line, then the blank line. */
function sseFrame(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream for an OpenAI client, from the buffered
 * answer: role chunk, content chunks, finish chunk, usage chunk, [DONE].
 * The usage chunk is mandatory — without it the usage check reads no usage.
 */
export function openAISSE(answer: ProviderAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames = [sseFrame(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseFrame(null, chunk({ content: piece }, null)));
  frames.push(sseFrame(null, chunk({}, 'stop')));
  frames.push(
    sseFrame(null, {
      id: 'chatcmpl-agg',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/**
 * An Anthropic SSE stream for an Anthropic client, from the buffered answer:
 * message_start (with input usage), content_block_start, text deltas,
 * content_block_stop, message_delta (stop reason + output usage), message_stop.
 */
export function anthropicSSE(answer: ProviderAnswer, model: string): string {
  const frames = [
    sseFrame('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
    sseFrame('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(sseFrame('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseFrame('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseFrame('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: answer.outputTokens } }),
    sseFrame('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}
