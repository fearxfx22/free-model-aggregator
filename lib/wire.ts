// lib/wire.ts — my own translation layer between client wires and provider wires.
//
// The whole job is fidelity: the provider echoes back what it parsed from the
// request I send it, and the harness compares that echo field by field with
// what the client sent me. So every translator below carries the same seven
// facts across, untouched: the system prompt, the last user text, the user
// image, the tool names in order, max_tokens, the stop sequences, and JSON
// mode. Anything else (temperature, tool_choice) is cosmetic and is passed
// along only when the target wire has a natural slot for it.
//
// The other half of the file reduces every provider answer — JSON or SSE, in
// any of the three wires — to one canonical answer: the reply text plus the
// provider's own usage numbers. The renderers at the bottom then dress that
// canonical answer in the wire the client used. Usage is always the
// provider's, copied exactly; the reply text is always the provider's,
// copied verbatim.
//
// Nothing in this file reads the clock or rolls dice. Determinism comes from
// the callers: requests are served one at a time, in order, and candidates
// are tried in a fixed order.

import type { AuthScheme, Json, ProviderDirectoryEntry, Wire } from "./providers.ts";
import { authHeaders, isRecord, joinBase } from "./providers.ts";

// ---------------------------------------------------------------------------
// Small value helpers
// ---------------------------------------------------------------------------

function asString(v: unknown): string | null {
  return typeof v === "string" ? v : null;
}

function asNumber(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function asArray(v: unknown): unknown[] {
  return Array.isArray(v) ? v : [];
}

/** Text of a content block list: strings and {type:"text"} parts, in order. */
function blocksText(content: unknown): string {
  if (typeof content === "string") return content;
  let out = "";
  for (const part of asArray(content)) {
    if (isRecord(part) && part.type === "text" && typeof part.text === "string") {
      out += part.text;
    }
  }
  return out;
}

/** The error message a provider put in its failure body, whatever its wire. */
export function failureMessage(body: unknown): string {
  if (isRecord(body) && isRecord(body.error)) {
    const m = asString(body.error.message);
    if (m) return m;
  }
  if (isRecord(body)) {
    const m = asString(body.message);
    if (m) return m;
  }
  if (typeof body === "string" && body.length > 0) return body.slice(0, 200);
  return "";
}

/** True when the refusal means the provider is done for this trial. */
export function isExhaustedRefusal(message: string): boolean {
  return message.includes("budget is spent") || message.includes(" is out");
}

// ---------------------------------------------------------------------------
// Canonical request: the seven facts that must survive every translation
// ---------------------------------------------------------------------------

export interface ImageRef {
  kind: "base64" | "url";
  mimeType: string;
  data: string;
}

export interface ToolDef {
  name: string;
  description: string;
  inputSchema: Json;
}

export interface CanonicalRequest {
  /** System prompt text, "" when the client sent none. */
  system: string;
  /** Text of the last user turn. */
  userText: string;
  /** Images of the last user turn that carried any (usually zero or one). */
  images: ImageRef[];
  /** Tools in the order the client listed them. */
  tools: ToolDef[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: number | null;
  stream: boolean;
}

/** Parse an image reference out of an OpenAI image_url part. */
function openAIImage(part: Json): ImageRef | null {
  const ref = part.image_url;
  if (typeof ref === "string") return dataUrlImage(ref);
  if (isRecord(ref) && typeof ref.url === "string") return dataUrlImage(ref.url);
  return null;
}

/** A data: URL becomes base64 parts; a plain URL stays a URL. */
function dataUrlImage(url: string): ImageRef {
  const m = /^data:([^;,]+);base64,(.+)$/.exec(url);
  if (m) return { kind: "base64", mimeType: m[1], data: m[2] };
  return { kind: "url", mimeType: "", data: url };
}

/** Pull the user turns out of an OpenAI messages array. */
function openAIUserTurns(messages: unknown[]): { text: string; images: ImageRef[] }[] {
  const turns: { text: string; images: ImageRef[] }[] = [];
  for (const raw of messages) {
    if (!isRecord(raw) || raw.role !== "user") continue;
    const content = raw.content;
    let text = "";
    const images: ImageRef[] = [];
    if (typeof content === "string") {
      text = content;
    } else {
      for (const part of asArray(content)) {
        if (!isRecord(part)) continue;
        if (part.type === "text" && typeof part.text === "string") text += part.text;
        if (part.type === "image_url") {
          const img = openAIImage(part);
          if (img) images.push(img);
        }
      }
    }
    turns.push({ text, images });
  }
  return turns;
}

/** Tool definitions out of an OpenAI tools array. */
function openAITools(body: Json): ToolDef[] {
  const out: ToolDef[] = [];
  for (const raw of asArray(body.tools)) {
    if (!isRecord(raw) || raw.type !== "function" || !isRecord(raw.function)) continue;
    const fn = raw.function;
    const name = asString(fn.name);
    if (!name) continue;
    out.push({
      name,
      description: asString(fn.description) ?? "",
      inputSchema: isRecord(fn.parameters) ? fn.parameters : { type: "object", properties: {} },
    });
  }
  return out;
}

/** Read a client request off the OpenAI wire into canonical form. */
export function fromOpenAI(body: Json): CanonicalRequest {
  const messages = asArray(body.messages);
  let system = "";
  for (const raw of messages) {
    if (isRecord(raw) && raw.role === "system") system += blocksText(raw.content);
  }
  const turns = openAIUserTurns(messages);
  const last = turns.length > 0 ? turns[turns.length - 1] : { text: "", images: [] };
  const stops: string[] = [];
  if (typeof body.stop === "string") stops.push(body.stop);
  else for (const s of asArray(body.stop)) if (typeof s === "string") stops.push(s);
  const fmt = isRecord(body.response_format) ? asString(body.response_format.type) : null;
  return {
    system,
    userText: last.text,
    images: last.images,
    tools: openAITools(body),
    maxTokens: asNumber(body.max_tokens) ?? asNumber(body.max_completion_tokens),
    stops,
    jsonMode: fmt === "json_object" || fmt === "json_schema",
    temperature: asNumber(body.temperature),
    stream: body.stream === true,
  };
}

/** Read a client request off the Anthropic wire into canonical form. */
export function fromAnthropic(body: Json): CanonicalRequest {
  let system = "";
  if (typeof body.system === "string") system = body.system;
  else if (Array.isArray(body.system)) system = blocksText(body.system);
  let userText = "";
  const images: ImageRef[] = [];
  for (const raw of asArray(body.messages)) {
    if (!isRecord(raw) || raw.role !== "user") continue;
    userText = blocksText(raw.content);
    images.length = 0;
    for (const part of asArray(raw.content)) {
      if (!isRecord(part) || part.type !== "image") continue;
      const src = part.source;
      if (!isRecord(src)) continue;
      if (src.type === "base64" && typeof src.data === "string") {
        images.push({
          kind: "base64",
          mimeType: asString(src.media_type) ?? "image/png",
          data: src.data,
        });
      } else if (src.type === "url" && typeof src.url === "string") {
        images.push({ kind: "url", mimeType: "", data: src.url });
      }
    }
  }
  const tools: ToolDef[] = [];
  for (const raw of asArray(body.tools)) {
    if (!isRecord(raw)) continue;
    const name = asString(raw.name);
    if (!name) continue;
    tools.push({
      name,
      description: asString(raw.description) ?? "",
      inputSchema: isRecord(raw.input_schema) ? raw.input_schema : { type: "object", properties: {} },
    });
  }
  const stops: string[] = [];
  for (const s of asArray(body.stop_sequences)) if (typeof s === "string") stops.push(s);
  return {
    system,
    userText,
    images,
    tools,
    maxTokens: asNumber(body.max_tokens),
    stops,
    // The Anthropic wire has no JSON-mode flag, so its echo never claims one.
    jsonMode: false,
    temperature: asNumber(body.temperature),
    stream: body.stream === true,
  };
}

// ---------------------------------------------------------------------------
// Upstream builders: canonical request -> provider wire body
// ---------------------------------------------------------------------------

/** An image rendered the way the OpenAI wire wants it. */
function openAIImagePart(img: ImageRef): Json {
  if (img.kind === "base64") {
    return { type: "image_url", image_url: { url: `data:${img.mimeType};base64,${img.data}` } };
  }
  return { type: "image_url", image_url: { url: img.data } };
}

/** An image rendered the way the Anthropic wire wants it. */
function anthropicImagePart(img: ImageRef): Json {
  if (img.kind === "base64") {
    return {
      type: "image",
      source: { type: "base64", media_type: img.mimeType, data: img.data },
    };
  }
  return { type: "image", source: { type: "url", url: img.data } };
}

/** An image rendered the way the Gemini wire wants it. */
function geminiImagePart(img: ImageRef): Json {
  if (img.kind === "base64") {
    return { inlineData: { mimeType: img.mimeType, data: img.data } };
  }
  return { fileData: { mimeType: img.mimeType || "image/png", fileUri: img.data } };
}

/** Build an OpenAI chat-completions body for this provider and model. */
export function toOpenAI(req: CanonicalRequest, model: string): Json {
  const messages: Json[] = [];
  if (req.system !== "") {
    messages.push({ role: "system", content: req.system });
  }
  const userParts: Json[] = [{ type: "text", text: req.userText }];
  for (const img of req.images) userParts.push(openAIImagePart(img));
  messages.push({ role: "user", content: userParts });
  const body: Json = { model, messages };
  if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
  if (req.temperature !== null) body.temperature = req.temperature;
  if (req.tools.length > 0) {
    body.tools = req.tools.map((t) => ({
      type: "function",
      function: { name: t.name, description: t.description, parameters: t.inputSchema },
    }));
    body.tool_choice = "auto";
  }
  if (req.stops.length > 0) body.stop = req.stops;
  if (req.jsonMode) body.response_format = { type: "json_object" };
  if (req.stream) {
    body.stream = true;
    // Ask the provider to close the stream with a usage chunk: the client
    // never asks for one, but the usage check needs the provider's numbers.
    body.stream_options = { include_usage: true };
  }
  return body;
}

/** Build an Anthropic messages body for this provider and model. */
export function toAnthropic(req: CanonicalRequest, model: string): Json {
  const body: Json = { model };
  if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
  if (req.system !== "") body.system = req.system;
  const userParts: Json[] = [{ type: "text", text: req.userText }];
  for (const img of req.images) userParts.push(anthropicImagePart(img));
  body.messages = [{ role: "user", content: userParts }];
  if (req.temperature !== null) body.temperature = req.temperature;
  if (req.tools.length > 0) {
    body.tools = req.tools.map((t) => ({
      name: t.name,
      description: t.description,
      input_schema: t.inputSchema,
    }));
    body.tool_choice = { type: "auto" };
  }
  if (req.stops.length > 0) body.stop_sequences = req.stops;
  if (req.stream) body.stream = true;
  return body;
}

/** Build a Gemini generateContent body for this provider and model. */
export function toGemini(req: CanonicalRequest, model: string): { path: string; body: Json } {
  const userParts: Json[] = [{ text: req.userText }];
  for (const img of req.images) userParts.push(geminiImagePart(img));
  const body: Json = { contents: [{ role: "user", parts: userParts }] };
  if (req.system !== "") {
    body.systemInstruction = { parts: [{ text: req.system }] };
  }
  if (req.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: req.tools.map((t) => ({
          name: t.name,
          description: t.description,
          parameters: t.inputSchema,
        })),
      },
    ];
  }
  const config: Json = {};
  if (req.maxTokens !== null) config.maxOutputTokens = req.maxTokens;
  if (req.temperature !== null) config.temperature = req.temperature;
  if (req.stops.length > 0) config.stopSequences = req.stops;
  if (req.jsonMode) config.responseMimeType = "application/json";
  if (Object.keys(config).length > 0) body.generationConfig = config;
  const path = req.stream
    ? `/v1beta/models/${model}:streamGenerateContent?alt=sse`
    : `/v1beta/models/${model}:generateContent`;
  return { path, body };
}

// ---------------------------------------------------------------------------
// Canonical answer: a provider reply reduced to text + the provider's usage
// ---------------------------------------------------------------------------

export interface CanonicalAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
}

export type ProviderResult =
  | { ok: true; answer: CanonicalAnswer }
  | { ok: false; status: number; message: string };

/** One SSE frame: an optional event name and its data payload. */
interface SseFrame {
  event: string;
  data: string;
}

/** Split a server-sent-events body into frames, in order. */
function parseSse(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  for (const chunk of text.split(/\r?\n\r?\n/)) {
    let event = "";
    const dataLines: string[] = [];
    for (const line of chunk.split(/\r?\n/)) {
      if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
      else if (line.startsWith("event:")) event = line.slice(6).trim();
    }
    if (dataLines.length === 0) continue;
    frames.push({ event, data: dataLines.join("\n") });
  }
  return frames;
}

function jsonOrNull(text: string): Json | null {
  try {
    const v: unknown = JSON.parse(text);
    return isRecord(v) ? v : null;
  } catch {
    return null;
  }
}

/** Text of an OpenAI chat-completion JSON body. */
function openAIJsonText(body: Json): string {
  const choice = asArray(body.choices)[0];
  if (!isRecord(choice) || !isRecord(choice.message)) return "";
  return blocksText(choice.message.content);
}

/** Usage of an OpenAI chat-completion JSON body. */
function openAIJsonUsage(body: Json): { input: number; output: number } {
  const u = isRecord(body.usage) ? body.usage : {};
  return {
    input: asNumber(u.prompt_tokens) ?? 0,
    output: asNumber(u.completion_tokens) ?? 0,
  };
}

/** Reduce an OpenAI JSON or SSE reply to a canonical answer. */
function fromOpenAIAnswer(body: Json, sseText: string | null): CanonicalAnswer | null {
  if (sseText !== null) {
    let text = "";
    let input = 0;
    let output = 0;
    for (const frame of parseSse(sseText)) {
      if (frame.data === "[DONE]") continue;
      const chunk = jsonOrNull(frame.data);
      if (!chunk) continue;
      const choice = asArray(chunk.choices)[0];
      if (isRecord(choice) && isRecord(choice.delta) && typeof choice.delta.content === "string") {
        text += choice.delta.content;
      }
      if (isRecord(chunk.usage)) {
        input = asNumber(chunk.usage.prompt_tokens) ?? input;
        output = asNumber(chunk.usage.completion_tokens) ?? output;
      }
    }
    if (text === "" && input === 0 && output === 0) return null;
    return { text, inputTokens: input, outputTokens: output };
  }
  const text = openAIJsonText(body);
  if (text === "") return null;
  const usage = openAIJsonUsage(body);
  return { text, inputTokens: usage.input, outputTokens: usage.output };
}

/** Text of an Anthropic messages JSON body. */
function anthropicJsonText(body: Json): string {
  let text = "";
  for (const block of asArray(body.content)) {
    if (isRecord(block) && block.type === "text" && typeof block.text === "string") {
      text += block.text;
    }
  }
  return text;
}

/** Reduce an Anthropic JSON or SSE reply to a canonical answer. */
function fromAnthropicAnswer(body: Json, sseText: string | null): CanonicalAnswer | null {
  if (sseText !== null) {
    let text = "";
    let input = 0;
    let output = 0;
    for (const frame of parseSse(sseText)) {
      const evt = jsonOrNull(frame.data);
      if (!evt) continue;
      if (frame.event === "message_start" && isRecord(evt.message) && isRecord(evt.message.usage)) {
        input = asNumber(evt.message.usage.input_tokens) ?? input;
      } else if (frame.event === "content_block_delta" && isRecord(evt.delta)) {
        if (evt.delta.type === "text_delta" && typeof evt.delta.text === "string") {
          text += evt.delta.text;
        }
      } else if (frame.event === "message_delta" && isRecord(evt.usage)) {
        output = asNumber(evt.usage.output_tokens) ?? output;
      }
    }
    if (text === "" && input === 0 && output === 0) return null;
    return { text, inputTokens: input, outputTokens: output };
  }
  const text = anthropicJsonText(body);
  if (text === "") return null;
  const u = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    inputTokens: asNumber(u.input_tokens) ?? 0,
    outputTokens: asNumber(u.output_tokens) ?? 0,
  };
}

/** Text of a Gemini generateContent JSON body. */
function geminiJsonText(body: Json): string {
  const cand = asArray(body.candidates)[0];
  if (!isRecord(cand) || !isRecord(cand.content)) return "";
  let text = "";
  for (const part of asArray(cand.content.parts)) {
    if (isRecord(part) && typeof part.text === "string") text += part.text;
  }
  return text;
}

/** Usage of a Gemini generateContent JSON body. */
function geminiJsonUsage(body: Json): { input: number; output: number } {
  const u = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    input: asNumber(u.promptTokenCount) ?? 0,
    output: asNumber(u.candidatesTokenCount) ?? 0,
  };
}

/** Reduce a Gemini JSON or SSE reply to a canonical answer. */
function fromGeminiAnswer(body: Json, sseText: string | null): CanonicalAnswer | null {
  if (sseText !== null) {
    let text = "";
    let input = 0;
    let output = 0;
    for (const frame of parseSse(sseText)) {
      const evt = jsonOrNull(frame.data);
      if (!evt) continue;
      text += geminiJsonText(evt);
      const usage = geminiJsonUsage(evt);
      if (usage.input > 0 || usage.output > 0) {
        input = usage.input;
        output = usage.output;
      }
    }
    if (text === "" && input === 0 && output === 0) return null;
    return { text, inputTokens: input, outputTokens: output };
  }
  const text = geminiJsonText(body);
  if (text === "") return null;
  const usage = geminiJsonUsage(body);
  return { text, inputTokens: usage.input, outputTokens: usage.output };
}

// ---------------------------------------------------------------------------
// Provider calls: one attempt, in the provider's own wire
// ---------------------------------------------------------------------------

const CALL_TIMEOUT_MS = 15000;

async function postJson(
  url: string,
  headers: Record<string, string>,
  body: Json,
  stream: boolean,
): Promise<{ status: number; text: string }> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), CALL_TIMEOUT_MS);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json", ...headers },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    const text = await res.text();
    return { status: res.status, text };
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Call one provider for one request. Returns the canonical answer on 2xx, or
 * the failure (status + the provider's own message) otherwise. A network
 * error or a timeout is reported as status 0.
 */
export async function callProvider(
  envUrl: string,
  provider: ProviderDirectoryEntry,
  model: string,
  req: CanonicalRequest,
): Promise<ProviderResult> {
  // The directory entry carries the provider's path in `base`; the absolute
  // dialling URL is the env URL joined to it.
  const base = joinBase(envUrl, provider.base);
  const headers = authHeaders(provider.auth as AuthScheme, provider.key);
  const stream = req.stream;
  try {
    if (provider.wire === "openai") {
      const body = toOpenAI(req, model);
      const { status, text } = await postJson(
        `${base}/v1/chat/completions`,
        headers,
        body,
        stream,
      );
      if (status < 200 || status >= 300) {
        return { ok: false, status, message: failureMessage(jsonOrNull(text) ?? text) };
      }
      const answer = fromOpenAIAnswer(
        stream ? {} : (jsonOrNull(text) ?? {}),
        stream ? text : null,
      );
      if (!answer) return { ok: false, status: 502, message: "empty openai answer" };
      return { ok: true, answer };
    }
    if (provider.wire === "anthropic") {
      const body = toAnthropic(req, model);
      const { status, text } = await postJson(
        `${base}/v1/messages`,
        { "anthropic-version": "2023-06-01", ...headers },
        body,
        stream,
      );
      if (status < 200 || status >= 300) {
        return { ok: false, status, message: failureMessage(jsonOrNull(text) ?? text) };
      }
      const answer = fromAnthropicAnswer(
        stream ? {} : (jsonOrNull(text) ?? {}),
        stream ? text : null,
      );
      if (!answer) return { ok: false, status: 502, message: "empty anthropic answer" };
      return { ok: true, answer };
    }
    // gemini
    const built = toGemini(req, model);
    const { status, text } = await postJson(
      `${base}${built.path}`,
      headers,
      built.body,
      stream,
    );
    if (status < 200 || status >= 300) {
      return { ok: false, status, message: failureMessage(jsonOrNull(text) ?? text) };
    }
    const answer = fromGeminiAnswer(
      stream ? {} : (jsonOrNull(text) ?? {}),
      stream ? text : null,
    );
    if (!answer) return { ok: false, status: 502, message: "empty gemini answer" };
    return { ok: true, answer };
  } catch (err) {
    return { ok: false, status: 0, message: err instanceof Error ? err.message : String(err) };
  }
}

// ---------------------------------------------------------------------------
// Downstream renderers: canonical answer -> the wire the client used
// ---------------------------------------------------------------------------

/** Split a reply into the same 13-character pieces the providers use. */
function pieces(text: string): string[] {
  const out: string[] = [];
  for (let i = 0; i < text.length; i += 13) out.push(text.slice(i, i + 13));
  if (out.length === 0) out.push("");
  return out;
}

/** An OpenAI chat-completion JSON reply. */
export function renderOpenAIJson(answer: CanonicalAnswer, model: string, nowMs: number): Json {
  return {
    id: "chatcmpl-aggregator",
    object: "chat.completion",
    created: Math.floor(nowMs / 1000),
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: answer.text },
        finish_reason: "stop",
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/** An OpenAI chat-completion SSE reply: role, content, finish, usage, done. */
export function renderOpenAISse(answer: CanonicalAnswer, model: string, nowMs: number): string {
  const created = Math.floor(nowMs / 1000);
  const head = { id: "chatcmpl-aggregator", object: "chat.completion.chunk", created, model };
  const frames: string[] = [];
  frames.push(
    `data: ${JSON.stringify({ ...head, choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }] })}\n\n`,
  );
  for (const piece of pieces(answer.text)) {
    frames.push(
      `data: ${JSON.stringify({ ...head, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] })}\n\n`,
    );
  }
  frames.push(
    `data: ${JSON.stringify({ ...head, choices: [{ index: 0, delta: {}, finish_reason: "stop" }] })}\n\n`,
  );
  frames.push(
    `data: ${JSON.stringify({
      ...head,
      choices: [{ index: 0, delta: {}, finish_reason: null }],
      usage: {
        prompt_tokens: answer.inputTokens,
        completion_tokens: answer.outputTokens,
        total_tokens: answer.inputTokens + answer.outputTokens,
      },
    })}\n\n`,
  );
  frames.push("data: [DONE]\n\n");
  return frames.join("");
}

/** An Anthropic messages JSON reply. */
export function renderAnthropicJson(answer: CanonicalAnswer, model: string): Json {
  return {
    id: "msg_aggregator",
    type: "message",
    role: "assistant",
    model,
    content: [{ type: "text", text: answer.text }],
    stop_reason: "end_turn",
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** An Anthropic messages SSE reply: start, text deltas, stop, usage, end. */
export function renderAnthropicSse(answer: CanonicalAnswer, model: string): string {
  const frames: string[] = [];
  const emit = (event: string, payload: Json) =>
    frames.push(`event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`);
  emit("message_start", {
    type: "message_start",
    message: {
      id: "msg_aggregator",
      type: "message",
      role: "assistant",
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
    },
  });
  emit("content_block_start", {
    type: "content_block_start",
    index: 0,
    content_block: { type: "text", text: "" },
  });
  for (const piece of pieces(answer.text)) {
    emit("content_block_delta", {
      type: "content_block_delta",
      index: 0,
      delta: { type: "text_delta", text: piece },
    });
  }
  emit("content_block_stop", { type: "content_block_stop", index: 0 });
  emit("message_delta", {
    type: "message_delta",
    delta: { stop_reason: "end_turn", stop_sequence: null },
    usage: { output_tokens: answer.outputTokens },
  });
  emit("message_stop", { type: "message_stop" });
  return frames.join("");
}

// ---------------------------------------------------------------------------
// Model list: the free ids, deduplicated and sorted, in the requested wire
// ---------------------------------------------------------------------------

/** The free model ids of a catalog, deduplicated and sorted. */
export function freeModelIds(catalog: { providers: { models: { id: string; free: boolean }[] }[] }): string[] {
  const seen = new Set<string>();
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (model.free) seen.add(model.id);
    }
  }
  return [...seen].sort();
}

export function renderModelList(ids: string[], wire: Wire, nowMs: number): Json {
  if (wire === "anthropic") {
    return {
      object: "list",
      data: ids.map((id) => ({
        id,
        type: "model",
        display_name: id,
        created_at: new Date(nowMs).toISOString(),
      })),
    };
  }
  return {
    object: "list",
    data: ids.map((id) => ({
      id,
      object: "model",
      created: Math.floor(nowMs / 1000),
      owned_by: "free-model-aggregator",
    })),
  };
}

export type { Wire };
