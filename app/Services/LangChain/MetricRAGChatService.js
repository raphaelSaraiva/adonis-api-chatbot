'use strict';

const MetricRAGService = use('App/Services/LangChain/MetricRAGService');
const WebRAGService = use('App/Services/LangChain/WebRAGService');
const path = use('path');
const fs = use('fs');


// OpenAI client
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
  console.error(
    '[MetricRAGChatService] ATENÇÃO: _fetch continua indisponível; chamadas HTTP ' +
      'para tradução / Ollama / LM Studio / WebRAG vão falhar.'
  );
}

// ===================================================================
// ✅ ENV helper (compatível com Adonis e com execução fora do bootstrap)
// ===================================================================

function envGet(key, fallback = '') {
  try {
    const Env = use('Env'); // ⚠️ não coloque no topo do arquivo
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

// ---------- OpenAI (lazy init) ---------- //

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

// ---------- Hosts ---------- //

function getOllamaHost() {
  return envGet('OLLAMA_HOST', 'http://127.0.0.1:11434');
}
function getLMStudioHost() {
  return envGet('LMSTUDIO_HOST', 'http://127.0.0.1:1234');
}

// ===================================================================
// ---------- Helpers de system prompt ----------
// ===================================================================

const PROMPTS_DIR = path.join(process.cwd(), 'prompts');
const systemPromptCache = {};

function loadSystemPromptSync(lang = 'pt', useRag = true) {
  const key = `${lang}_${useRag ? 'rag' : 'norag'}`;
  if (systemPromptCache[key]) {
    console.log('[SYSTEM PROMPT] cache hit para', key);
    return systemPromptCache[key];
  }

  const fileName = `system_${lang}_${useRag ? 'rag' : 'norag'}.txt`;
  const fullPath = path.join(PROMPTS_DIR, fileName);

  console.log('[SYSTEM PROMPT] tentando carregar', fullPath);
  try {
    if (fs.existsSync(fullPath)) {
      const txt = fs.readFileSync(fullPath, 'utf8');
      systemPromptCache[key] = txt.trim();
      console.log('[SYSTEM PROMPT] carregado com sucesso', fileName);
      return systemPromptCache[key];
    }
  } catch (e) {
    console.error('[SYSTEM PROMPT] erro ao carregar', fileName, e.message);
  }

  console.warn('[SYSTEM PROMPT] usando fallback para', key);

  const fallbackRag =
    'You are a helpful assistant that answers questions about software metrics for blockchain systems. ' +
    'Use strictly the provided context. If the context is insufficient, say that you are not sure.';

  const fallbackNoRag =
    'You are a helpful assistant that answers questions about software metrics for blockchain systems. ' +
    'Answer clearly and concisely. If you are not sure, say that you are not sure.';

  systemPromptCache[key] = useRag ? fallbackRag : fallbackNoRag;
  return systemPromptCache[key];
}

// ===================================================================
// ---------- Tradução (Google translate endpoint) ----------
// ===================================================================

async function translateText(text, targetLanguage) {
  console.log('[TRANSLATE] chamado com:', {
    textSnippet: (text || '').slice(0, 80),
    targetLanguage,
  });

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
    console.error('[TRANSLATE] erro ao traduzir:', e.message);
    return text;
  }
}

// ===================================================================
// ---------- Seleção de provider ----------
// ===================================================================

function resolveProvider(modelName) {
  console.log('[LLM] Resolving provider for model:', modelName);
  const lower = (modelName || '').toLowerCase();

  if (lower === 'openai' || lower.startsWith('openai-')) return 'openai';
  if (lower === 'ollama' || lower.startsWith('ollama-')) return 'ollama';
  if (lower === 'lmstudio' || lower.startsWith('lmstudio-')) return 'lmstudio';

  if (
    lower.startsWith('llama2') ||
    lower.startsWith('llama-2') ||
    lower.startsWith('llama3') ||
    lower.startsWith('llama-3')
  ) {
    console.log('[LLM] Resolved provider (heuristic): OLLAMA');
    return 'ollama';
  }

  console.log('[LLM] Resolved provider (default): lmstudio');
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
// ---------- Chamada unificada de LLM ----------
// ===================================================================

async function generateWithLLM(systemPrompt, userPrompt, modelName) {
  const chosenModel = modelName;
  if (!chosenModel) return null;

  const provider = resolveProvider(chosenModel);
  console.log(`[LLM] Provider=${provider} | model=${chosenModel}`);

  // OPENAI
  if (provider === 'openai') {
    const client = getOpenAIClient();
    const key = getOpenAIKey();

    if (!client || !key) {
      console.error('[OPENAI] OPENAI_API_KEY não definido ou cliente não inicializado.');
      return null;
    }

    const realModel = resolveOpenAIModel(chosenModel);
    console.log('[OPENAI] model resolvido:', realModel);

    try {
      const completion = await client.chat.completions.create({
        model: realModel,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt },
        ],
        temperature: 0.2,
      });

      const content = completion.choices?.[0]?.message?.content || '';
      return content.trim();
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
          options: { temperature: 0.2 },
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
        temperature: 0.2,
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
// ✅ Helpers docs: normalize + rerank + combine local/web
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

function topKeywords(text, n = 10) {
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

function rerankDocs(docs, question, topK) {
  if (!Array.isArray(docs) || docs.length === 0) return [];
  const qkw = topKeywords(question, 10);

  const scored = docs.map((d) => {
    const text = (d.content || '').toLowerCase();
    const source = String(d.source || '').toLowerCase();

    let score = 0;

    // keywords do enunciado
    for (const kw of qkw) if (kw && text.includes(kw)) score += 3;

    // similaridade local ajuda
    const sim = (typeof d.similarity === 'number') ? d.similarity : Number(d.similarity || 0);
    if (!Number.isNaN(sim) && sim > 0) score += sim * 2;

    // docs web ganham leve bônus
    if (d.origin === 'web' || source.startsWith('http://') || source.startsWith('https://')) score += 1.0;

    // penaliza muito longos
    if (text.length > 8000) score -= 1.5;

    return { d, score };
  });

  scored.sort((a, b) => b.score - a.score);
  const picked = scored.filter(x => x.score > 0).slice(0, Math.max(3, topK)).map(x => x.d);

  return picked.length ? picked : scored.slice(0, Math.max(3, topK)).map(x => x.d);
}

// ===================================================================
// ✅ Helpers: normalizar métrica e formatar histórico para o prompt
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
      const a = clampText(h?.chosenText || h?.response);
      if (!q && !a) return null;

      return [
        `# Turn ${i + 1}`,
        q ? `User: ${q}` : null,
        a ? `Assistant: ${a}` : null,
      ].filter(Boolean).join('\n');
    })
    .filter(Boolean);

  if (!blocks.length) return '';
  return `\n\n=== RECENT CHAT HISTORY ===\n${blocks.join('\n\n')}\n=== END CHAT HISTORY ===\n`;
}

// ===================================================================
// Service principal (RAG + NO-RAG + WEBRAG)
// ===================================================================

class MetricRAGChatService {
  constructor() {
    this.ragByAlgorithm = {};

    this.answerLanguage = (envGet('ANSWER_LANGUAGE', 'pt') || 'pt').toLowerCase();
    this.retrieveTopK = envInt('RETRIEVE_TOP_K', 6);
    this.minSimilarity = envFloat('MIN_LOCAL_SIMILARITY', 0.0);

    // ✅ WEB RAG flags
    this.webRagEnabled = envBool('WEB_RAG_ENABLED', false);
    this.webRagMaxResults = envInt('WEB_RAG_MAX_RESULTS', 10);
    this.webRagEngine = envGet('WEB_RAG_ENGINE', 'multi');
    this.webRagMinLocalSim = envFloat('WEB_RAG_MIN_LOCAL_SIM', 0.75); // só chama web se bestSim < isso

    // ✅ instancia WebRAG (igual evaluateMetrics)
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

    console.log('[MetricRAGChatService] construído com config:', {
      answerLanguage: this.answerLanguage,
      retrieveTopK: this.retrieveTopK,
      minSimilarity: this.minSimilarity,
      webRagEnabled: this.webRagEnabled,
      webRagMaxResults: this.webRagMaxResults,
      webRagEngine: this.webRagEngine,
      webRagMinLocalSim: this.webRagMinLocalSim,
    });
  }

  async ensureInitializedForAlgorithm(algorithmName) {
    if (!algorithmName) throw new Error('algorithmName é obrigatório');

    if (!this.ragByAlgorithm[algorithmName]) {
      console.log('[MetricRAGChatService] criando novo MetricRAGService para', algorithmName);
      const rag = new MetricRAGService(algorithmName);
      await rag.connect();
      this.ragByAlgorithm[algorithmName] = rag;
      console.log(`[MetricRAGChatService] RAG inicializado para "${algorithmName}"`);
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

    // 1) traduz pergunta para EN
    const qEn = await translateText(question, 'en');

    // 2) system prompt
    const systemPrompt = loadSystemPromptSync(this.answerLanguage, useRag);

    // 3) NO-RAG
    if (!useRag) {
      const userPrompt = [
        `You are answering a question about the metric "${metric}".`,
        metricId || metricName
          ? `Metric details: metricId="${metricId || ''}" metricName="${metricName || ''}".`
          : '',
        historyBlock ? historyBlock.trimEnd() : '',
        `Question (in English): ${qEn}`,
        '',
        `Answer in ${langStr}:`,
      ].filter(Boolean).join('\n');

      const rawAnswer = await generateWithLLM(systemPrompt, userPrompt, algorithm);
      if (!rawAnswer) return 'Não foi possível obter uma resposta da LLM no momento.';

      if (this.answerLanguage === 'pt') return translateText(rawAnswer, 'pt');
      return rawAnswer;
    }

    // 4) RAG local
    await this.ensureInitializedForAlgorithm(algorithm);
    const rag = this.ragByAlgorithm[algorithm];

    const localDocsRaw = await rag.queryDocuments(metric, qEn, this.retrieveTopK, {
      minSimilarity: this.minSimilarity,
      fetchMultiplier: 3,
    });

    const localDocs = normalizeDocs(localDocsRaw, 'local');
    const bestLocalSim = localDocs.length
      ? Math.max(...localDocs.map(d => Number(d.similarity || 0)))
      : 0;

    console.log('[MetricRAGChatService] local docs:', {
      count: localDocs.length,
      bestLocalSim,
      minSimilarity: this.minSimilarity,
    });

    // 5) WEB RAG (complemento) — igual evaluateMetrics
    let webDocs = [];
    let usedWeb = false;

    const shouldUseWeb =
      this.webRagEnabled &&
      typeof this.webRag?.query === 'function' &&
      typeof _fetch === 'function' &&
      (localDocs.length === 0 || bestLocalSim < this.webRagMinLocalSim);

    if (shouldUseWeb) {
      console.log(
        `[MetricRAGChatService] WEB_RAG_ENABLED=1 — chamando WebRAG (bestLocalSim=${bestLocalSim.toFixed(3)} < ${this.webRagMinLocalSim})`
      );

      try {
        const webDocsRaw = await this.webRag.query(qEn, metric);
        webDocs = normalizeDocs(webDocsRaw, 'web');
        usedWeb = webDocs.length > 0;

        console.log('[MetricRAGChatService] web docs:', { count: webDocs.length, engine: this.webRagEngine });
      } catch (e) {
        console.warn('[MetricRAGChatService] WebRAG falhou:', e.message);
      }
    } else {
      console.log('[MetricRAGChatService] WebRAG não usado:', {
        webRagEnabled: this.webRagEnabled,
        localCount: localDocs.length,
        bestLocalSim,
        webRagMinLocalSim: this.webRagMinLocalSim,
      });
    }

    // 6) combina e rerankeia (local + web)
    let combined = [...localDocs, ...webDocs];
    if (!combined.length) {
      return 'Não encontrei informação suficiente na base para responder a essa pergunta.';
    }

    combined = rerankDocs(combined, question, Math.max(6, this.retrieveTopK));

    // 7) monta contexto final (com meta)
    const context = combined
      .map((d, idx) => {
        const header = `[#${idx + 1} | origin=${d.origin} | source=${d.source || ''} | page=${d.page ?? ''}]`;
        return `${header}\n${d.content || ''}`.trim();
      })
      .join('\n\n---\n\n');

    console.log('[MetricRAGChatService] Contexto montado:', {
      docs: combined.length,
      usedWeb,
      contextLen: context.length,
    });

    // 8) prompt final
    const userPrompt = [
      `You are answering a question about the metric "${metric}".`,
      metricId || metricName
        ? `Metric details: metricId="${metricId || ''}" metricName="${metricName || ''}".`
        : '',
      'Use ONLY the information provided in the following context chunks.',
      'If the context is insufficient, say that you are not sure.',
      historyBlock ? historyBlock.trimEnd() : '',
      '',
      'Context:',
      context,
      '',
      `Question (in English): ${qEn}`,
      '',
      `Answer in ${langStr}:`,
    ].filter(Boolean).join('\n');

    const rawAnswer = await generateWithLLM(systemPrompt, userPrompt, algorithm);
    if (!rawAnswer) return 'Não foi possível obter uma resposta da LLM no momento.';

    if (this.answerLanguage === 'pt') return translateText(rawAnswer, 'pt');
    return rawAnswer;
  }
}

module.exports = MetricRAGChatService;
