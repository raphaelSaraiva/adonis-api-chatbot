'use strict';

const MetricRAGService = use('App/Services/LangChain/MetricRAGService');
const WebRAGService = use('App/Services/LangChain/WebRAGService');

const path = use('path');
const fs = use('fs');

const OpenAI = use('openai');

// fetch global (Node 18+) ou fallback para node-fetch
let _fetch = globalThis.fetch || global.fetch;
if (typeof _fetch !== 'function') {
  try {
    // eslint-disable-next-line global-require
    _fetch = require('node-fetch');
    console.log('[MetricRAGChatService] node-fetch carregado como fallback de fetch');
  } catch (e) {
    console.error('[MetricRAGChatService] ERRO carregando node-fetch:', e.message);
  }
}
if (typeof _fetch !== 'function') {
  console.error('[MetricRAGChatService] ATENÇÃO: _fetch indisponível; WebRAG/Ollama/LM Studio podem falhar.');
}

// ===================================================================
// ✅ ENV helper
// ===================================================================

function envGet(key, fallback = '') {
  try {
    const Env = use('Env');
    const v = Env.get(key);
    return (v !== undefined && v !== null ? String(v) : String(fallback)).trim();
  } catch (e) {
    const v = process.env[key];
    return (v !== undefined && v !== null ? String(v) : String(fallback)).trim();
  }
}

function envBool(key, fallback = false) {
  const v = envGet(key, fallback ? '1' : '0').toLowerCase();
  return v === '1' || v === 'true' || v === 'yes' || v === 'on';
}

function envInt(key, fallback) {
  const n = parseInt(envGet(key, String(fallback)), 10);
  return Number.isFinite(n) ? n : fallback;
}

function envFloat(key, fallback) {
  const n = parseFloat(envGet(key, String(fallback)));
  return Number.isFinite(n) ? n : fallback;
}

// ===================================================================
// ---------- OpenAI (lazy init) ----------
// ===================================================================

let _openaiClient = null;
let _openaiKeyCached = null;

function getOpenAIKey() {
  return envGet('OPENAI_API_KEY', '');
}
function getOpenAIModelDefault() {
  return envGet('OPENAI_MODEL', 'gpt-4.1-mini');
}
function getOpenAIClient() {
  const key = getOpenAIKey();
  if (!key) return null;

  if (!_openaiClient || _openaiKeyCached !== key) {
    _openaiClient = new OpenAI({ apiKey: key });
    _openaiKeyCached = key;
  }
  return _openaiClient;
}

// ===================================================================
// ---------- Hosts ----------
// ===================================================================

function getOllamaHost() {
  return envGet('OLLAMA_HOST', 'http://127.0.0.1:11434');
}
function getLMStudioHost() {
  return envGet('LMSTUDIO_HOST', 'http://127.0.0.1:1234');
}

// ===================================================================
// ---------- System prompt loader ----------
// ===================================================================

const PROMPTS_DIR = path.join(process.cwd(), 'prompts');
const systemPromptCache = {};

function loadSystemPromptSync(lang = 'pt', useRag = true) {
  const key = `${lang}_${useRag ? 'rag' : 'norag'}`;
  if (systemPromptCache[key]) return systemPromptCache[key];

  const fileName = `system_${lang}_${useRag ? 'rag' : 'norag'}.txt`;
  const fullPath = path.join(PROMPTS_DIR, fileName);

  try {
    if (fs.existsSync(fullPath)) {
      const txt = fs.readFileSync(fullPath, 'utf8');
      systemPromptCache[key] = txt.trim();
      return systemPromptCache[key];
    }
  } catch (e) {
    console.error('[SYSTEM PROMPT] erro:', e.message);
  }

  const fallbackRag =
    'You are a helpful assistant that answers questions about software metrics for blockchain systems. ' +
    'Use the provided context when available. Be concise and technical.';

  const fallbackNoRag =
    'You are a helpful assistant that answers questions about software metrics for blockchain systems. ' +
    'Answer clearly, concisely, and technically.';

  systemPromptCache[key] = useRag ? fallbackRag : fallbackNoRag;
  return systemPromptCache[key];
}

// ===================================================================
// ---------- Tradução ----------
// ===================================================================

async function translateText(text, targetLanguage) {
  if (!text || typeof text !== 'string') return text;
  if (!targetLanguage || typeof targetLanguage !== 'string') return text;
  if (typeof _fetch !== 'function') return text;

  try {
    const url =
      'https://translate.googleapis.com/translate_a/single' +
      `?client=gtx&sl=auto&tl=${encodeURIComponent(targetLanguage)}` +
      `&dt=t&q=${encodeURIComponent(text)}`;

    const res = await _fetch(url);
    if (!res.ok) return text;

    const data = await res.json();
    const translated = (data[0] || []).map((chunk) => chunk[0]).join('');
    return translated || text;
  } catch (e) {
    console.error('[TRANSLATE] erro:', e.message);
    return text;
  }
}

// ===================================================================
// ---------- Provider selection ----------
// ===================================================================

function resolveProvider(modelName) {
  const lower = (modelName || '').toLowerCase();

  if (lower === 'openai' || lower.startsWith('openai-')) return 'openai';
  if (lower === 'ollama' || lower.startsWith('ollama-')) return 'ollama';
  if (lower === 'lmstudio' || lower.startsWith('lmstudio-')) return 'lmstudio';

  if (
    lower.startsWith('llama2') ||
    lower.startsWith('llama-2') ||
    lower.startsWith('llama3') ||
    lower.startsWith('llama-3')
  ) return 'ollama';

  return 'lmstudio';
}

function resolveOpenAIModel(modelName) {
  const OPENAI_MODEL = getOpenAIModelDefault();
  const lower = (modelName || '').toLowerCase();

  if (lower === 'openai') return OPENAI_MODEL;
  if (lower.startsWith('openai-')) {
    const m = modelName.slice('openai-'.length).trim();
    return m || OPENAI_MODEL;
  }
  return OPENAI_MODEL;
}

// ===================================================================
// ---------- Unified LLM call ----------
// ===================================================================

async function generateWithLLM(systemPrompt, userPrompt, modelName, opts = {}) {
  const chosenModel = modelName;
  if (!chosenModel) return null;

  const provider = resolveProvider(chosenModel);
  const temperature = typeof opts.temperature === 'number' ? opts.temperature : 0.2;

  // OPENAI
  if (provider === 'openai') {
    const client = getOpenAIClient();
    const key = getOpenAIKey();
    if (!client || !key) return null;

    const realModel = resolveOpenAIModel(chosenModel);

    try {
      const completion = await client.chat.completions.create({
        model: realModel,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt },
        ],
        temperature,
      });

      return (completion.choices?.[0]?.message?.content || '').trim();
    } catch (e) {
      console.error('[OPENAI] erro:', e.message);
      return null;
    }
  }

  // OLLAMA
  if (provider === 'ollama') {
    const OLLAMA_HOST = getOllamaHost();
    if (!OLLAMA_HOST || typeof _fetch !== 'function') return null;

    try {
      const fullPrompt = `${systemPrompt}\n\n${userPrompt}`;
      const res = await _fetch(`${OLLAMA_HOST}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: chosenModel,
          prompt: fullPrompt,
          stream: false,
          options: { temperature },
        }),
      });

      if (!res.ok) return null;
      const json = await res.json();
      return (json.response || '').trim();
    } catch (e) {
      console.error('[OLLAMA] erro:', e.message);
      return null;
    }
  }

  // LM STUDIO
  const LMSTUDIO_HOST = getLMStudioHost();
  if (!LMSTUDIO_HOST || typeof _fetch !== 'function') return null;

  try {
    const res = await _fetch(`${LMSTUDIO_HOST}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: chosenModel,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt },
        ],
        temperature,
      }),
    });

    if (!res.ok) return null;
    const json = await res.json();
    return (json.choices?.[0]?.message?.content || '').trim();
  } catch (e) {
    console.error('[LMStudio] erro:', e.message);
    return null;
  }
}

// ===================================================================
// ✅ Contratos: curtos, estilo evaluateMetrics.js (3–5 frases)
// - Sem seções fixas
// - Proíbe "paper-tone" (foi proposto/estudos mostram) sem evidência explícita
// ===================================================================

function buildShortAnswerContract(answerLanguage, { strictToContext } = {}) {
  const lang = (answerLanguage || 'pt').toLowerCase();

  if (lang === 'pt') {
    const lines = [
      'Responda de forma técnica e direta.',
      'Máximo de 3 a 5 frases. Sem listas, sem headings, sem seções.',
      'Não inclua introduções ou conclusões. Vá direto ao ponto.',
      'Não mencione IDs/códigos internos (ex.: "t12").',
      'Evite tom de artigo ("foi proposto", "estudos mostram", "na literatura") a menos que isso esteja explicitamente no contexto.',
      strictToContext
        ? 'Use APENAS o que estiver presente no CONTEXTO. Se não houver base suficiente, diga isso em 1 frase e pare.'
        : 'Se faltar dado específico, responda com comportamento típico (sem inventar números) e deixe isso claro na primeira frase.',
    ];
    return lines.join('\n');
  }

  const lines = [
    'Answer technically and directly.',
    'Maximum 3 to 5 sentences. No lists, no headings, no extra sections.',
    'No long intros or conclusions. Go straight to the point.',
    'Do not mention internal IDs/codes (e.g., "t12").',
    'Avoid paper-like claims ("was proposed", "studies show", "in the literature") unless explicitly present in the context.',
    strictToContext
      ? 'Use ONLY what is present in the CONTEXT. If there is not enough basis, say so in 1 sentence and stop.'
      : 'If specific data is missing, answer using typical behavior (no made-up numbers) and make that explicit in the first sentence.',
  ];
  return lines.join('\n');
}

function stripInternalCodesFromAnswer(text) {
  let t = String(text || '');
  t = t.replace(/metric\s*id\s*[:=]\s*\\?["']?[a-z]\d{1,6}\\?["']?/gi, '');
  t = t.replace(/metricId\s*[:=]\s*\\?["']?[a-z]\d{1,6}\\?["']?/g, '');
  t = t.replace(/[\(\[\{]\s*[a-z]\d{1,6}\s*[\)\]\}]/gi, '');
  t = t.replace(/\s{2,}/g, ' ').replace(/\s+\n/g, '\n').trim();
  return t;
}

function postprocessShortAnswer(raw, { maxSentences = 5 } = {}) {
  if (!raw) return '';
  let s = String(raw).trim();

  // remove markdown comum
  s = s.replace(/\*\*/g, '');
  s = s.replace(/^[-*]\s+/gm, '');
  s = s.replace(/```[\s\S]*?```/g, '');

  // corta por frases
  const parts = s
    .split(/(?<=[\.\!\?])\s+/)
    .map(p => p.trim())
    .filter(Boolean);

  const picked = parts.slice(0, Math.max(3, Math.min(maxSentences, 6)));
  return picked.join(' ').trim();
}

function looksLikeNoInfoAnswer(text, lang = 'pt') {
  const t = String(text || '').toLowerCase();
  if (!t) return true;

  const patternsPt = [
    'não há informação suficiente',
    'não encontrei informação suficiente',
    'informação não disponível',
    'não disponível no contexto',
    'não tenho informação suficiente',
    'sem informação suficiente',
    'não é possível responder com segurança',
  ];

  const patternsEn = [
    'not enough information',
    'insufficient information',
    'information not available',
    'not available in the context',
    'cannot answer safely',
  ];

  const patterns = (lang || 'pt').toLowerCase() === 'pt' ? patternsPt : patternsEn;
  return patterns.some(p => t.includes(p));
}

// ===================================================================
// ✅ Classificador/Guard (mantidos, mas sem “caso especial” por tópico)
// ===================================================================

async function llmGuardMetricQuestion({ userText, metric, modelName, answerLanguage = 'pt' }) {
  const lang = (answerLanguage || 'pt').toLowerCase();
  const outLang = lang === 'pt' ? 'Portuguese' : 'English';

  const system =
    'You are a strict input validator for a Metrics Q&A system. ' +
    'Return ONLY valid JSON. No markdown. No extra text.';

  const user =
    `User message:\n"""${String(userText || '').trim()}"""\n\n` +
    `Metric context (if any): "${String(metric || '').trim()}"\n\n` +
    'Return JSON with exactly these keys:\n' +
    '{ "is_metric_question": boolean, "reason": string, "suggested_question": string }\n' +
    `The "suggested_question" must be in ${outLang}.\n`;

  const raw = await generateWithLLM(system, user, modelName, { temperature: 0.0 });
  if (!raw) return { is_metric_question: true, reason: 'validator_failed', suggested_question: '' };

  const txt = String(raw).trim();
  const start = txt.indexOf('{');
  const end = txt.lastIndexOf('}');
  if (start === -1 || end === -1 || end <= start) {
    return { is_metric_question: true, reason: 'invalid_validator_output', suggested_question: '' };
  }

  try {
    const obj = JSON.parse(txt.slice(start, end + 1));
    return {
      is_metric_question: Boolean(obj.is_metric_question),
      reason: typeof obj.reason === 'string' ? obj.reason : '',
      suggested_question: typeof obj.suggested_question === 'string' ? obj.suggested_question : '',
    };
  } catch {
    return { is_metric_question: true, reason: 'validator_parse_failed', suggested_question: '' };
  }
}

// Heurística genérica (sem tópico específico):
// se fala de experimento/impacto/condições/resultado, preferimos "analysis-style"
// mas a resposta continua curta (sem seções).
function isAnalysisStyleQuestionHeuristic(question) {
  const q = String(question || '').toLowerCase().trim();
  if (!q) return false;

  const patterns = [
    /\bhow\b/, /\bwhy\b/, /\baffect\b/, /\bimpact\b/, /\binfluence\b/,
    /\bunder\s+load\b/, /\bworkload\b/, /\bexperiment\b/, /\bresults?\b/,
    /\baccording\s+to\b/, /\bstudy\b/, /\bobserved\b/, /\bbehavior\b/,
    /\btrade-?off\b/, /\blimitations?\b/, /\bvariance\b/, /\bjitter\b/,
    /\btimeout\b/, /\brounds?\b/, /\bvalidators?\b/,
  ];

  return patterns.some((re) => re.test(q));
}

// ===================================================================
// ✅ Helpers: normalize metric + history
// ===================================================================

function normalizeMetric(opts) {
  const metricId = String(opts?.metricId || '').trim();
  const metricName = String(opts?.metricName || '').trim();
  const legacyMetric = String(opts?.metric || '').trim();
  const metric = metricName || legacyMetric || metricId || 'Latency';
  return { metric, metricId, metricName };
}

function clampText(s, max = 900) {
  const t = String(s || '');
  return t.length > max ? t.slice(0, max) + '…' : t;
}

function formatHistoryForPrompt(history, maxTurns = 6) {
  if (!Array.isArray(history) || history.length === 0) return '';

  const last = history.slice(-maxTurns);
  const blocks = last
    .map((h, i) => {
      const q = clampText(h?.question);
      const a = clampText(h?.chosenText || h?.response || h?.answer);
      if (!q && !a) return null;
      return [`# Turn ${i + 1}`, q ? `User: ${q}` : null, a ? `Assistant: ${a}` : null]
        .filter(Boolean)
        .join('\n');
    })
    .filter(Boolean);

  if (!blocks.length) return '';
  return `\n\n=== RECENT CHAT HISTORY ===\n${blocks.join('\n\n')}\n=== END CHAT HISTORY ===\n`;
}

// ===================================================================
// ✅ Docs helpers (normalize + rerank + compress) — genérico
// ===================================================================

function normalizeDocs(docs, origin = 'local') {
  if (!Array.isArray(docs)) return [];
  return docs
    .map((d) => ({
      content: (d.content || d.context || d.text || '').toString(),
      source: d.source || d.metadata?.source || d.path || 'unknown',
      origin: d.origin || origin,
      similarity:
        typeof d.similarity === 'number'
          ? d.similarity
          : (d.similarity != null ? Number(d.similarity) : undefined),
      page: d.page ?? d.metadata?.page,
    }))
    .filter((x) => x.content && x.content.trim().length > 0);
}

function topKeywords(text, n = 12) {
  const stop = new Set([
    'a','o','os','as','de','da','do','das','dos',
    'em','no','na','nos','nas','um','uma','e','é','que',
    'com','para','por','se','sem','ou','ao','à','às','sobre',
    'como','qual','quais','porque','porquê','por que','por quê'
  ]);

  return String(text || '')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .split(/\s+/)
    .filter(t => t && !stop.has(t) && t.length > 2)
    .slice(0, n);
}

// Rerank genérico: keywords + sim + bônus web + penalidade por texto longo
function rerankDocs(docs, question, topK) {
  if (!Array.isArray(docs) || docs.length === 0) return [];
  const qkw = topKeywords(question, 14);

  const scored = docs.map((d) => {
    const text = (d.content || '').toLowerCase();
    const source = String(d.source || '').toLowerCase();

    let score = 0;

    for (const kw of qkw) if (kw && text.includes(kw)) score += 6;

    const sim = (typeof d.similarity === 'number') ? d.similarity : Number(d.similarity || 0);
    if (!Number.isNaN(sim) && sim > 0) score += sim * 2;

    if (d.origin === 'web' || source.startsWith('http://') || source.startsWith('https://')) score += 2.0;

    const len = text.length;
    if (len > 8000) score -= 2;
    else if (len > 4000) score -= 1;

    return { d, score };
  });

  scored.sort((a, b) => b.score - a.score);

  let picked = scored.filter(x => x.score > 0);
  if (picked.length < Math.min(topK, 4)) picked = scored.slice(0, Math.max(topK, 4));

  return picked.slice(0, Math.max(topK, 3)).map(x => x.d);
}

// Contexto em blocos "Trecho N:" (sem fontes)
function compressContextForLLM(docs, { maxChars = 4500, perDocChars = 1200 } = {}) {
  let out = '';
  const used = [];
  let idx = 1;

  for (const d of docs) {
    if (!d?.content) continue;

    const piece = String(d.content).trim().slice(0, perDocChars);
    const block = `Trecho ${idx}:\n${piece}\n`;

    if (out.length + block.length > maxChars) break;

    out += (out ? '\n' : '') + block;
    used.push(d);
    idx += 1;

    if (out.length >= maxChars) break;
  }

  return { context: out.trim(), usedDocs: used };
}

// Evidência mínima genérica: overlap de keywords da pergunta no contexto
function hasMinimumEvidence(question, docs, { minHits = 2 } = {}) {
  if (!question) return false;
  if (!Array.isArray(docs) || docs.length === 0) return false;

  const qkw = topKeywords(question, 12);
  if (!qkw.length) return false;

  const joined = docs.map(d => String(d?.content || '')).join('\n').toLowerCase();

  let hits = 0;
  for (const kw of qkw) {
    if (kw && joined.includes(kw)) hits += 1;
    if (hits >= minHits) return true;
  }
  return false;
}

// ===================================================================
// Service principal
// ===================================================================

class MetricRAGChatService {
  constructor() {
    this.ragByAlgorithm = {};

    this.answerLanguage = (envGet('ANSWER_LANGUAGE', 'pt') || 'pt').toLowerCase();

    this.retrieveTopK = envInt('RETRIEVE_TOP_K', 20);
    this.finalContextDocs = envInt('FINAL_CONTEXT_DOCS', 6);
    this.minSimilarity = envFloat('MIN_LOCAL_SIMILARITY', 0.6);

    this.contextMaxChars = envInt('CONTEXT_MAX_CHARS', 4500);
    this.contextDocChars = envInt('CONTEXT_DOC_CHARS', 1200);
    this.debugContext = envBool('DEBUG_CONTEXT', false);

    // ✅ WebRAG: habilita se WEB_RAG_ENABLED=1 (sem gate; sempre tenta)
    const rawWebEnabled = envGet('WEB_RAG_ENABLED', '0');
    this.webRagEnabled = rawWebEnabled === '1' || envBool('WEB_RAG_ENABLED', false);

    this.webRagMaxResults = envInt('WEB_RAG_MAX_RESULTS', 10);
    this.webRagEngine = envGet('WEB_RAG_ENGINE', 'multi');

    // ✅ Qualidade: fallback no-rag quando RAG vier "no info"
    // (mantém RAG bom quando ele já respondeu bem)
    this.allowFallbackWithoutEvidence =
      envGet('ALLOW_FALLBACK_WITHOUT_EVIDENCE', '1') === '1' || envBool('ALLOW_FALLBACK_WITHOUT_EVIDENCE', true);

    this.webRag = new WebRAGService({
      enabled: this.webRagEnabled,
      fetchImpl: _fetch,
      maxResults: this.webRagMaxResults,
      engine: this.webRagEngine,

      tavilyEndpoint: envGet('TAVILY_ENDPOINT', ''),
      tavilyApiKey: envGet('TAVILY_API_KEY', ''),
      braveEndpoint: envGet('BRAVE_SEARCH_ENDPOINT', ''),
      braveApiKey: envGet('BRAVE_SEARCH_API_KEY', ''),
      arxivEndpoint: envGet('ARXIV_ENDPOINT', ''),
      serperApiKey: envGet('SERPER_API_KEY', ''),
    });

    console.log(
      `[MetricRAGChatService] init: ANSWER_LANGUAGE=${this.answerLanguage} ` +
      `RETRIEVE_TOP_K=${this.retrieveTopK} FINAL_CONTEXT_DOCS=${this.finalContextDocs} ` +
      `MIN_LOCAL_SIMILARITY=${this.minSimilarity} CONTEXT_MAX_CHARS=${this.contextMaxChars} ` +
      `WEB_RAG_ENABLED(raw=${rawWebEnabled}) => ${this.webRagEnabled} MAX_RESULTS=${this.webRagMaxResults} ENGINE=${this.webRagEngine} ` +
      `ALLOW_FALLBACK_WITHOUT_EVIDENCE=${this.allowFallbackWithoutEvidence}`
    );
  }

  async ensureInitializedForAlgorithm(algorithmName) {
    if (!algorithmName) throw new Error('algorithmName é obrigatório');

    if (!this.ragByAlgorithm[algorithmName]) {
      const rag = new MetricRAGService(algorithmName);
      await rag.connect();
      this.ragByAlgorithm[algorithmName] = rag;
      console.log(`[MetricRAGChatService] RAG conectado para algorithm=${algorithmName}`);
    }
  }

  async askQuestion(question, optsOrModel) {
    if (typeof optsOrModel === 'string') {
      return this._askQuestionInternal(question, {
        metric: 'Latency',
        metricId: '',
        metricName: '',
        history: [],
        algorithm: optsOrModel,
        useRag: true,
      });
    }

    const opts = optsOrModel || {};
    const { metric, metricId, metricName } = normalizeMetric(opts);

    let history = opts.history || [];
    if (typeof history === 'string') {
      try { history = JSON.parse(history); } catch { history = []; }
    }
    if (!Array.isArray(history)) history = [];

    return this._askQuestionInternal(question, {
      metric,
      metricId,
      metricName,
      history,
      algorithm: opts.algorithm || 'llama3',
      useRag: typeof opts.useRag === 'boolean' ? opts.useRag : true,
    });
  }

  async _askQuestionInternal(question, { metric, metricId, metricName, history, algorithm, useRag }) {
    if (!question) throw new Error('Pergunta vazia.');

    const langStr = this.answerLanguage === 'pt' ? 'Portuguese' : 'English';
    const historyBlock = formatHistoryForPrompt(history, 6);

    const metricLabel = (metricName || metric || '').trim() || 'Latency';

    console.log(
      `[MetricRAGChatService] askQuestion: useRag=${useRag} metric=${metricLabel} algorithm=${algorithm} ` +
      `historyTurns=${Array.isArray(history) ? history.length : 0}`
    );

    // Guard
    const guard = await llmGuardMetricQuestion({
      userText: question,
      metric: metricLabel,
      modelName: algorithm,
      answerLanguage: this.answerLanguage,
    });

    if (!guard.is_metric_question) {
      console.warn('[MetricRAGChatService] guard bloqueou:', guard.reason);
      return this.answerLanguage === 'pt'
        ? '⚠️ A mensagem enviada não parece ser uma pergunta sobre métricas.'
        : '⚠️ Your message does not look like a metrics question.';
    }

    const analysisStyle = isAnalysisStyleQuestionHeuristic(question);

    // traduz pergunta para EN (retriever/WebRAG)
    const qEn = await translateText(question, 'en');
    const qOriginal = String(question || '').trim();

    const systemPrompt = loadSystemPromptSync(this.answerLanguage, useRag);

    // ------------------------------------------------------------
    // NO-RAG
    // ------------------------------------------------------------
    if (!useRag) {
      const contract = buildShortAnswerContract(this.answerLanguage, { strictToContext: false });

      const userPrompt = [
        `You are answering a question related to the metric "${metricLabel}".`,
        '',
        contract,
        '',
        historyBlock
          ? ('NOTE: Chat history below is ONLY for disambiguation. Do NOT treat it as evidence.\n' + historyBlock.trimEnd())
          : '',
        `Question (in English): ${qEn}`,
        '',
        `Answer in ${langStr}:`,
      ].filter(Boolean).join('\n');

      const rawAnswer = await generateWithLLM(systemPrompt, userPrompt, algorithm, { temperature: 0.2 });
      if (!rawAnswer) {
        return this.answerLanguage === 'pt'
          ? 'Não foi possível obter uma resposta da LLM no momento.'
          : 'Could not get an answer from the LLM right now.';
      }

      const cleaned = stripInternalCodesFromAnswer(rawAnswer);
      const short = postprocessShortAnswer(cleaned, { maxSentences: 5 });
      return this.answerLanguage === 'pt' ? translateText(short, 'pt') : short;
    }

    // ------------------------------------------------------------
    // RAG local
    // ------------------------------------------------------------
    await this.ensureInitializedForAlgorithm(algorithm);
    const rag = this.ragByAlgorithm[algorithm];

    let localDocsRaw = [];
    try {
      localDocsRaw = await rag.queryDocuments(metric, qEn, this.retrieveTopK, {
        minSimilarity: this.minSimilarity,
        fetchMultiplier: 3,
      });
    } catch (e) {
      console.warn('[MetricRAGChatService] queryDocuments falhou:', e.message);
    }

    let localDocs = normalizeDocs(localDocsRaw, 'local');
    const bestLocalSim = localDocs.length
      ? Math.max(...localDocs.map(d => Number(d.similarity || 0)))
      : 0;

    console.log(
      `[MetricRAGChatService] localDocs=${localDocs.length} bestLocalSim=${Number(bestLocalSim).toFixed(3)} ` +
      `(minSimilarity=${this.minSimilarity})`
    );

    // ------------------------------------------------------------
    // ✅ WebRAG SEM RESTRIÇÃO (se habilitado, sempre tenta)
    // ------------------------------------------------------------
    let webDocs = [];
    let usedWeb = false;

    const canWeb =
      this.webRagEnabled &&
      typeof this.webRag?.query === 'function' &&
      typeof _fetch === 'function';

    if (canWeb) {
      console.log('[MetricRAGChatService] 🌐 WebRAG habilitado — buscando SEM restrição...');
      try {
        const webDocsRaw = await this.webRag.query(qEn, metricLabel);
        webDocs = normalizeDocs(webDocsRaw, 'web');

        // (opcional) evita explosão de ruído
        const maxWeb = Math.max(this.retrieveTopK, 20);
        if (webDocs.length > maxWeb) webDocs = webDocs.slice(0, maxWeb);

        usedWeb = webDocs.length > 0;
        console.log(`[MetricRAGChatService] 🌐 WebRAG retornou webDocs=${webDocs.length}`);
      } catch (e) {
        console.warn('[MetricRAGChatService] 🌐 WebRAG falhou:', e.message);
      }
    } else {
      console.log(`[MetricRAGChatService] 🌐 WebRAG indisponível (enabled=${this.webRagEnabled}, fetch=${typeof _fetch})`);
    }

    // ------------------------------------------------------------
    // Combine + rerank + final top N
    // ------------------------------------------------------------
    let combined = [...localDocs, ...webDocs];

    if (!combined.length) {
      return this.answerLanguage === 'pt'
        ? 'Não encontrei informação suficiente na base para responder a essa pergunta.'
        : 'I did not find enough information to answer this question.';
    }

    // rerank amplo (usa pergunta ORIGINAL para keywords)
    combined = rerankDocs(combined, qOriginal, Math.max(this.retrieveTopK, 10));

    // final top N
    let finalDocs = rerankDocs(combined, qOriginal, this.finalContextDocs);
    if (!finalDocs.length) finalDocs = combined.slice(0, this.finalContextDocs);

    const localCount = finalDocs.filter(d => d.origin !== 'web').length;
    const webCount = finalDocs.filter(d => d.origin === 'web').length;

    console.log(
      `[MetricRAGChatService] ✅ finalDocs=${finalDocs.length} (local=${localCount}, web=${webCount}) ` +
      (usedWeb ? '• (local + WebRAG)' : '• (apenas local)')
    );

    const { context, usedDocs } = compressContextForLLM(finalDocs, {
      maxChars: this.contextMaxChars,
      perDocChars: this.contextDocChars,
    });

    if (this.debugContext) {
      console.log('\n[MetricRAGChatService][DEBUG_CONTEXT] --- CONTEXTO ENVIADO AO LLM (primeiros 2000 chars) ---\n');
      console.log(context.slice(0, 2000));
      console.log('\n[MetricRAGChatService][DEBUG_CONTEXT] --- FIM CONTEXTO ---\n');
      console.log('[MetricRAGChatService][DEBUG_CONTEXT] usedDocs sources (interno):');
      usedDocs.forEach((d, i) => {
        console.log(`  - #${i + 1} origin=${d.origin} sim=${d.similarity ?? 'n/a'} source=${d.source} page=${d.page ?? ''}`);
      });
    }

    const hasEvidence = hasMinimumEvidence(qOriginal, finalDocs, { minHits: 2 });

    // ------------------------------------------------------------
    // Prompt final RAG (short + strict-to-context)
    // ------------------------------------------------------------
    const contract = buildShortAnswerContract(this.answerLanguage, { strictToContext: true });

    const strictNote = (!hasEvidence)
      ? (this.answerLanguage === 'pt'
        ? 'IMPORTANTE: o contexto recuperado pode não ter evidência suficiente para sustentar uma resposta completa. Se faltar base, diga isso em 1 frase e pare.'
        : 'IMPORTANT: the retrieved context may not have enough evidence to support a complete answer. If the basis is insufficient, say so in 1 sentence and stop.')
      : '';

    const userPrompt = [
      `You are answering a question related to the metric "${metricLabel}".`,
      '',
      contract,
      '',
      strictNote,
      strictNote ? '' : '',
      'Context:',
      context,
      '',
      `Question (in English): ${qEn}`,
      '',
      `Answer in ${langStr}:`,
    ].filter(Boolean).join('\n');

    const rawAnswer = await generateWithLLM(systemPrompt, userPrompt, algorithm, { temperature: 0.2 });
    if (!rawAnswer) {
      return this.answerLanguage === 'pt'
        ? 'Não foi possível obter uma resposta da LLM no momento.'
        : 'Could not get an answer from the LLM right now.';
    }

    const cleaned = stripInternalCodesFromAnswer(rawAnswer);
    const shortRag = postprocessShortAnswer(cleaned, { maxSentences: 5 });

    // ------------------------------------------------------------
    // ✅ Qualidade: se RAG vier “no info” / fraco, cai para fallback NO-RAG curto
    // (assim o RAG nunca fica pior do que o no-rag quando o contexto é ruim)
    // ------------------------------------------------------------
    const ragLooksBad =
      !shortRag ||
      shortRag.length < 60 ||
      looksLikeNoInfoAnswer(shortRag, this.answerLanguage);

    if (this.allowFallbackWithoutEvidence && ragLooksBad) {
      console.log('[MetricRAGChatService] 🔁 RAG fraco/no-info — acionando fallback NO-RAG curto.');

      const fallbackContract = buildShortAnswerContract(this.answerLanguage, { strictToContext: false });

      const fallbackPrompt = [
        `You are answering a question related to the metric "${metricLabel}".`,
        '',
        fallbackContract,
        '',
        (this.answerLanguage === 'pt'
          ? 'Nota: não encontrei base suficiente no contexto recuperado. Responda com base em conhecimento geral, deixando isso explícito na primeira frase.'
          : 'Note: I did not find enough basis in the retrieved context. Answer using general knowledge, making that explicit in the first sentence.'),
        '',
        // usa a pergunta ORIGINAL para o modelo (melhor fluência), mas mantém também EN
        `Question: ${qOriginal}`,
        '',
        `Answer in ${langStr}:`,
      ].join('\n');

      const fbRaw = await generateWithLLM(
        loadSystemPromptSync(this.answerLanguage, false),
        fallbackPrompt,
        algorithm,
        { temperature: 0.2 }
      );

      if (fbRaw && fbRaw.trim()) {
        const fbClean = stripInternalCodesFromAnswer(fbRaw);
        const fbShort = postprocessShortAnswer(fbClean, { maxSentences: 5 });
        return this.answerLanguage === 'pt' ? translateText(fbShort, 'pt') : fbShort;
      }
    }

    // Se chegou aqui: RAG está bom o suficiente
    return this.answerLanguage === 'pt' ? translateText(shortRag, 'pt') : shortRag;
  }
}

module.exports = MetricRAGChatService;
