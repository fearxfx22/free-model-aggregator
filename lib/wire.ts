// The client wires and the three upstream wires this aggregator speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an
// Anthropic one (POST /v1/messages), with a JSON body or with server-sent events. The providers
// behind it speak one of three wires (openai, anthropic, gemini). This file is the translation
// layer between them, written around one idea: every provider answer, whatever its wire, is
// plain text with a stop family of 0 and a usage pair, so each answer is first reduced to a
// canonical form {text, inputTokens, outputTokens} and then rendered in the client's wire.
// That keeps the six client x provider combinations down to three parsers and two renderers,
// and it keeps the usage the client sees exactly the usage the provider reported.
//
// Request translation goes the other way, per wire:
//   - openai: the client's body, with the routed model's id; stream_options.include_usage is
//     always set on a stream so the usage chunk comes back.
//   - anthropic: system becomes a top-level system field, messages become Anthropic blocks
//     (text, image, tool_use, tool_result), tools become Anthropic tools, stop becomes
//     stop_sequences. Anthropic has no response_format, so a JSON-mode request never routes
//     to an anthropic-wire provider (main.ts enforces that).
//   - gemini: messages become generateContent contents (systemInstruction for the system
//     prompt, inlineData/fileData for images, functionCall/functionResponse for tool traffic),
//     tools become functionDeclarations, and generationConfig carries maxOutputTokens,
//     stopSequences, responseMimeType for JSON mode, temperature and topP.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same
// body and the same provider answer give the same bytes on a replay. The model listings use
// fixed timestamps for the same reason.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** The id stamped on every Chat Completions answer this entry renders. */
const OPENAI_ANSWER_ID = 'chatcmpl-aggregator';
/** The id stamped on every Messages answer this entry renders. */
const ANTHROPIC_ANSWER_ID = 'msg_aggregator';
/** The id stamped on every streamed Messages answer this entry renders. */
const ANTHROPIC_STREAM_ID = 'msg_stream';
/** How wide the text pieces are in a rendered stream: the providers answer in 13-char pieces. */
const STREAM_PIECE = 13;

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One OpenAI content part as an Anthropic content block: text or an inline image; null for the rest. */
function anthropicBlockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const dataUrl = /^data:([^;,]+);base64,(.+)$/.exec(url);
    if (dataUrl !== null) {
      return { type: 'image', source: { type: 'base64', media_type: dataUrl[1] ?? 'image/png', data: dataUrl[2] ?? '' } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`: system
 * messages become the top-level system field, user parts become Anthropic blocks (text and image),
 * assistant tool_calls become tool_use blocks, tool messages become tool_result blocks, tools
 * become Anthropic tools, tool_choice is rewritten, and the output cap, temperature, top_p, stop
 * sequences and streaming pass through. An OpenAI response_format has no Anthropic spelling and is
 * not carried: a JSON-mode request never routes to an anthropic-wire provider (see main.ts).
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    const content = raw.content;
    if (role === 'system') {
      const text = textOf(content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (role === 'tool') {
      const text = typeof content === 'string' ? content : textOf(content);
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: text }],
      });
      continue;
    }
    // Any other role is a user turn.
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      const block = anthropicBlockOf(part);
      if (block !== null) blocks.push(block);
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else out.max_tokens = 1024;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  else if (typeof body.stop === 'string' && body.stop !== '') out.stop_sequences = [body.stop];
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One OpenAI content part as a Gemini part: text or an inline image; null for the rest. */
function geminiPartOfOpenAI(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const dataUrl = /^data:([^;,]+);base64,(.+)$/.exec(url);
    if (dataUrl !== null) return { inlineData: { mimeType: dataUrl[1] ?? 'image/png', data: dataUrl[2] ?? '' } };
    return { fileData: { fileUri: url } };
  }
  return null;
}

/** One Anthropic content block as a Gemini part: text, image or tool_result; null for the rest. */
function geminiPartOfAnthropic(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  if (block.type === 'tool_result') {
    return {
      functionResponse: {
        name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
        response: { content: toolResultText(block.content) },
      },
    };
  }
  return null;
}

/**
 * Either client wire's request as a Gemini generateContent body naming `model`: the system prompt
 * becomes systemInstruction, messages become contents (user/model roles; images become inlineData
 * or fileData; tool calls become functionCall parts and tool results become functionResponse
 * parts), tools become functionDeclarations, and generationConfig carries the output cap, stop
 * sequences, JSON mode, temperature and top_p. An Anthropic client has no JSON-mode signal, so
 * no responseMimeType is set for it.
 */
export function clientToGemini(body: Json, model: string): Json {
  const anthropicClient = typeof body.system === 'string' || Array.isArray(body.system);
  const systemTexts: string[] = [];
  const topSystem = textOf(body.system);
  if (topSystem !== '') systemTexts.push(topSystem);
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    const content = raw.content;
    if (role === 'system') {
      // An OpenAI client's system prompt is a message, not a top-level field: it still
      // becomes systemInstruction, joined the way the OpenAI wire joins system texts.
      const text = textOf(content);
      if (text !== '') systemTexts.push(text);
      continue;
    }
    if (anthropicClient) {
      if (role === 'assistant') {
        const parts: Json[] = [];
        if (Array.isArray(content)) {
          for (const block of content) {
            if (!isRecord(block)) continue;
            if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
            else if (block.type === 'tool_use') {
              parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
            }
          }
        } else if (typeof content === 'string') {
          parts.push({ text: content });
        }
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      const parts: Json[] = [];
      if (typeof content === 'string') {
        parts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          const part = geminiPartOfAnthropic(block);
          if (part !== null) parts.push(part);
        }
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
      continue;
    }
    // OpenAI client.
    if (role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (role === 'tool') {
      const text = typeof content === 'string' ? content : textOf(content);
      contents.push({
        role: 'user',
        parts: [{ functionResponse: { name: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', response: { content: text } } }],
      });
      continue;
    }
    const parts: Json[] = [];
    if (typeof content === 'string') {
      parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        const geminiPart = geminiPartOfOpenAI(part);
        if (geminiPart !== null) parts.push(geminiPart);
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }

  const out: Json = { model, contents };
  const systemText = systemTexts.join('\n');
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const tools = Array.isArray(body.tools) ? body.tools : [];
  if (tools.length > 0) {
    const declarations = tools.flatMap((raw) => {
      if (!isRecord(raw)) return [];
      if (anthropicClient) {
        if (typeof raw.name !== 'string') return [];
        return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
      }
      if (!isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : null;
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  const stops = Array.isArray(body.stop_sequences) ? body.stop_sequences : Array.isArray(body.stop) ? body.stop : typeof body.stop === 'string' ? [body.stop] : [];
  if (stops.length > 0) config.stopSequences = stops;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** One provider answer, reduced to the text and usage every wire carries. */
export interface CanonicalAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
}

/** A Chat Completions answer (JSON) reduced to the canonical form. */
export function parseOpenAIAnswer(body: Json): CanonicalAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const usage = isRecord(body.usage) ? body.usage : {};
  return { text, inputTokens: numberOrZero(usage.prompt_tokens), outputTokens: numberOrZero(usage.completion_tokens) };
}

/** A Messages answer (JSON) reduced to the canonical form. */
export function parseAnthropicAnswer(body: Json): CanonicalAnswer {
  const blocks = Array.isArray(body.content) ? body.content : [];
  const text = blocks
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
  const usage = isRecord(body.usage) ? body.usage : {};
  return { text, inputTokens: numberOrZero(usage.input_tokens), outputTokens: numberOrZero(usage.output_tokens) };
}

/** A Gemini generateContent answer (JSON) reduced to the canonical form. */
export function parseGeminiAnswer(body: Json): CanonicalAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { text, inputTokens: numberOrZero(usage.promptTokenCount), outputTokens: numberOrZero(usage.candidatesTokenCount) };
}

/**
 * The parsed JSON payloads of a server-sent event stream, in order, skipping [DONE] and lines
 * that are not data. The stream is consumed whole; the caller renders the collected answer.
 */
export async function* sseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<unknown> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const framesOf = (text: string): unknown[] => {
    const out: unknown[] = [];
    for (const frame of text.split('\n\n')) {
      for (const line of frame.split('\n')) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data:')) continue;
        const payload = trimmed.slice(5).trim();
        if (payload === '' || payload === '[DONE]') continue;
        try {
          out.push(JSON.parse(payload));
        } catch {
          // A line that is not JSON is not a frame.
        }
      }
    }
    return out;
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        for (const frame of framesOf(buffer.slice(0, at))) yield frame;
        buffer = buffer.slice(at + 2);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const frame of framesOf(buffer)) yield frame;
  } finally {
    reader.releaseLock();
  }
}

/** A Chat Completions SSE stream reduced to the canonical form. */
export async function collectOpenAIStream(stream: ReadableStream<Uint8Array>): Promise<CanonicalAnswer> {
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  for await (const frame of sseFrames(stream)) {
    if (!isRecord(frame)) continue;
    if (isRecord(frame.usage)) {
      inputTokens = numberOrZero(frame.usage.prompt_tokens);
      outputTokens = numberOrZero(frame.usage.completion_tokens);
    }
    const choices = Array.isArray(frame.choices) ? frame.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
  }
  return { text, inputTokens, outputTokens };
}

/** A Messages SSE stream reduced to the canonical form. */
export async function collectAnthropicStream(stream: ReadableStream<Uint8Array>): Promise<CanonicalAnswer> {
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  for await (const frame of sseFrames(stream)) {
    if (!isRecord(frame)) continue;
    const type = frame.type;
    if (type === 'message_start' && isRecord(frame.message) && isRecord(frame.message.usage)) {
      inputTokens = numberOrZero(frame.message.usage.input_tokens);
    }
    if (type === 'content_block_delta' && isRecord(frame.delta) && frame.delta.type === 'text_delta' && typeof frame.delta.text === 'string') {
      text += frame.delta.text;
    }
    if (type === 'message_delta' && isRecord(frame.usage)) {
      outputTokens = numberOrZero(frame.usage.output_tokens);
    }
  }
  return { text, inputTokens, outputTokens };
}

/** A Gemini streamGenerateContent SSE stream reduced to the canonical form. */
export async function collectGeminiStream(stream: ReadableStream<Uint8Array>): Promise<CanonicalAnswer> {
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  for await (const frame of sseFrames(stream)) {
    if (!isRecord(frame)) continue;
    const candidates = Array.isArray(frame.candidates) ? frame.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null && isRecord(candidate.content) && Array.isArray(candidate.content.parts)) {
      for (const part of candidate.content.parts) {
        if (isRecord(part) && typeof part.text === 'string') text += part.text;
      }
    }
    if (isRecord(frame.usageMetadata)) {
      inputTokens = numberOrZero(frame.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(frame.usageMetadata.candidatesTokenCount);
    }
  }
  return { text, inputTokens, outputTokens };
}

/**
 * A 200 answer that is an error in disguise: an error envelope, or a 200 whose message text
 * carries a billing/exhaustion message. A provider that answers 200 with an error never counts
 * as a success: the router must fail over, the way it would on a 402 or 429.
 */
export function errorOfAnswer(body: unknown): { message: string; code: string } | null {
  if (!isRecord(body)) return null;
  if (isRecord(body.error)) {
    const e = body.error;
    const message = typeof e.message === 'string' ? e.message : 'the provider answered an error';
    const code = typeof e.code === 'string' ? e.code : typeof e.code === 'number' ? String(e.code) : typeof e.status === 'string' ? e.status : typeof e.type === 'string' ? e.type : '';
    return { message, code };
  }
  if (body.type === 'error') return { message: 'the provider answered an error', code: '' };
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const first = isRecord(choices[0]) ? choices[0] : null;
  const message = first !== null && isRecord(first.message) ? first.message : null;
  const content = message !== null && typeof message.content === 'string' ? message.content : '';
  if (content !== '') {
    const low = content.toLowerCase();
    if (low.includes('insufficient_quota') || (low.includes('credit') && low.includes('exhaust')) || low.includes('billing_error')) {
      return { message: content.slice(0, 200), code: 'error_text' };
    }
  }
  return null;
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split a text into pieces of STREAM_PIECE chars, the way a provider's stream pieces it. */
function piecesOf(text: string): string[] {
  const out: string[] = [];
  for (let i = 0; i < text.length; i += STREAM_PIECE) out.push(text.slice(i, i + STREAM_PIECE));
  return out;
}

/** A canonical answer as a Chat Completions JSON object. */
export function renderOpenAIAnswer(answer: CanonicalAnswer, model: string): Json {
  return {
    id: OPENAI_ANSWER_ID,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: 'stop' }],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
  };
}

/** A canonical answer as a Chat Completions SSE event sequence. */
export function renderOpenAIStream(answer: CanonicalAnswer, model: string): string[] {
  const head: Json = { id: OPENAI_ANSWER_ID, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model };
  const events: string[] = [];
  events.push(sseEvent(null, { ...head, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] }));
  for (const piece of piecesOf(answer.text)) {
    events.push(sseEvent(null, { ...head, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] }));
  }
  events.push(sseEvent(null, { ...head, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }));
  events.push(sseEvent(null, { ...head, choices: [], usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens } }));
  events.push('data: [DONE]\n\n');
  return events;
}

/** A canonical answer as a Messages JSON object. */
export function renderAnthropicAnswer(answer: CanonicalAnswer, model: string): Json {
  return {
    id: ANTHROPIC_ANSWER_ID,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** A canonical answer as a Messages SSE event sequence. */
export function renderAnthropicStream(answer: CanonicalAnswer, model: string): string[] {
  const events: string[] = [];
  events.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: ANTHROPIC_STREAM_ID,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
  );
  events.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
  for (const piece of piecesOf(answer.text)) {
    events.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  events.push(
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: answer.outputTokens } }),
  );
  events.push(sseEvent('message_stop', { type: 'message_stop' }));
  return events;
}
