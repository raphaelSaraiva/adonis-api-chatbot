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
  console.error(
    '[MetricRAGChatService] ATENÇÃO: _fetch continua indisponível; chamadas HTTP ' +
      'para tradução / Ollama / LM Studio / WebRAG vão falhar.'
  );
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
    'Use strictly the provided context. If the context is insufficient, say that you are not sure.';

  const fallbackNoRag =
    'You are a helpful assistant that answers questions about software metrics for blockchain systems. ' +
    'Answer clearly and concisely. If you are not sure, say that you are not sure.';

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
// ✅ Contract (sem referências) + limpeza anti-id
// ===================================================================

function buildAnswerContract(answerLanguage, { isRag = false } = {}) {
  const lang = (answerLanguage || 'pt').toLowerCase();

  const basePt = [
    'Siga EXATAMENTE a estrutura abaixo (sem adicionar seções extras):',
    '1. Definição da métrica',
    '2. Como a métrica é medida',
    '3. Principais fatores que influenciam a métrica',
    '4. Interpretação prática dos valores',
    '',
    'Regras:',
    '- Use parágrafos curtos e objetivos.',
    '- Não inclua introduções longas nem conclusões fora das 4 seções.',
    '- NÃO cite IDs/códigos internos (ex.: "t12") em nenhuma parte da resposta.',
  ];

  const baseEn = [
    'Follow EXACTLY the structure below (do not add extra sections):',
    '1. Metric definition',
    '2. How the metric is measured',
    '3. Main factors influencing the metric',
    '4. Practical interpretation of values',
    '',
    'Rules:',
    '- Use short, objective paragraphs.',
    '- Do not add long intros or conclusions outside the 4 sections.',
    '- DO NOT mention internal IDs/codes (e.g., "t12") anywhere.',
  ];

  if (lang === 'pt') {
    if (isRag) {
      basePt.push(
        '- Use APENAS o que estiver presente no CONTEXTO fornecido.',
        '- Se o contexto não trouxer informação suficiente para alguma seção, escreva: "Informação não disponível" (na própria seção).',
        '- Não complete com conhecimento externo.'
      );
    } else {
      basePt.push(
        '- Não inclua detalhes avançados que o usuário não pediu explicitamente.',
        '- Se não tiver certeza, diga que não tem certeza.'
      );
    }
    return basePt.join('\n');
  }

  if (isRag) {
    baseEn.push(
      '- Use ONLY what is present in the provided CONTEXT.',
      '- If the context is insufficient for a section, write: "Information not available" inside that section.',
      '- Do not fill gaps with external knowledge.'
    );
  } else {
    baseEn.push(
      '- Do not add advanced details unless explicitly asked by the user.',
      '- If you are not sure, say you are not sure.'
    );
  }
  return baseEn.join('\n');
}

function buildAnalysisContract(answerLanguage, { isRag = false } = {}) {
  const lang = (answerLanguage || 'pt').toLowerCase();

  if (lang === 'pt') {
    const lines = [
      'Responda DIRETAMENTE à pergunta do usuário.',
      'Estrutura obrigatória (sem adicionar seções extras):',
      '1. Condições de carga (workload) em que o fenômeno piora',
      '2. Evidência/efeito observado nas medições (o que muda na latência)',
      '3. Interpretação (por que isso acontece, baseado no que foi fornecido)',
      '',
      'Regras:',
      '- NÃO faça definição geral da métrica (evite “Latência é…”), a menos que a pergunta peça.',
      '- Use parágrafos curtos e objetivos.',
      '- NÃO cite IDs/códigos internos (ex.: "t12").',
    ];

    if (isRag) {
      lines.push(
        '- Use APENAS o que estiver presente no CONTEXTO fornecido.',
        '- Se o contexto não trouxer informação suficiente, diga explicitamente: "Informação não disponível no contexto".'
      );
    } else {
      lines.push(
        '- Não invente detalhes específicos do artigo se eles não foram fornecidos.',
        '- Se não tiver certeza, diga que não tem certeza.'
      );
    }

    return lines.join('\n');
  }

  const lines = [
    'Answer the user DIRECTLY.',
    'Required structure (do not add extra sections):',
    '1. Workload conditions where the phenomenon becomes worse',
    '2. Evidence/effect observed in measurements (how latency changes)',
    '3. Interpretation (why it happens, based on provided info)',
    '',
    'Rules:',
    '- Do NOT provide a generic metric definition (avoid “Latency is…”), unless asked.',
    '- Use short, objective paragraphs.',
    '- Do NOT mention internal IDs/codes (e.g., "t12").',
  ];

  if (isRag) {
    lines.push(
      '- Use ONLY what is present in the provided CONTEXT.',
      '- If the context is insufficient, say: "Information not available in the context".'
    );
  } else {
    lines.push(
      '- Do not invent paper-specific details if they were not provided.',
      '- If you are not sure, say you are not sure.'
    );
  }

  return lines.join('\n');
}

function stripInternalCodesFromAnswer(text) {
  let t = String(text || '');

  t = t.replace(/metric\s*id\s*[:=]\s*\\?["']?[a-z]\d{1,6}\\?["']?/gi, '');
  t = t.replace(/metricId\s*[:=]\s*\\?["']?[a-z]\d{1,6}\\?["']?/g, '');

  t = t.replace(/[\(\[\{]\s*[a-z]\d{1,6}\s*[\)\]\}]/gi, '');

  t = t.replace(/\b([a-z]\d{1,6})\b/gi, (m) => {
    if (/^p\d+$/i.test(m)) return m;
    return '';
  });

  t = t.replace(/\s{2,}/g, ' ').replace(/\s+\n/g, '\n').trim();
  return t;
}

// ===================================================================
// ✅ Guard via LLM (opcional) — mantido
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

// ===================================================================
// ✅ NEW: Classificador de tipo de pergunta (métrica vs análise)
// ===================================================================

async function llmClassifyQuestionType({ userText, metric, modelName, answerLanguage = 'pt' }) {
  const system =
    'You are a strict classifier for a Metrics Q&A system. ' +
    'Return ONLY valid JSON. No markdown. No extra text.';

  const user =
    `User message:\n"""${String(userText || '').trim()}"""\n\n` +
    `Metric context (if any): "${String(metric || '').trim()}"\n\n` +
    'Return JSON with exactly these keys:\n' +
    '{ "question_type": "metric_contract" | "analysis_contract", "reason": string }\n' +
    'Choose "metric_contract" when the user is asking for definition/measurement/factors/interpretation of the metric itself.\n' +
    'Choose "analysis_contract" when the user is asking about an experiment, workload conditions, paper results, comparative behavior, or "according to the article" style questions.\n';

  const raw = await generateWithLLM(system, user, modelName, { temperature: 0.0 });
  if (!raw) return { question_type: 'metric_contract', reason: 'classifier_failed' };

  const txt = String(raw).trim();
  const start = txt.indexOf('{');
  const end = txt.lastIndexOf('}');
  if (start === -1 || end === -1 || end <= start) {
    return { question_type: 'metric_contract', reason: 'invalid_classifier_output' };
  }

  try {
    const obj = JSON.parse(txt.slice(start, end + 1));
    const qt = obj?.question_type === 'analysis_contract' ? 'analysis_contract' : 'metric_contract';
    return { question_type: qt, reason: typeof obj.reason === 'string' ? obj.reason : '' };
  } catch {
    return { question_type: 'metric_contract', reason: 'classifier_parse_failed' };
  }
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
// ✅ Docs helpers (normalize + rerank)
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
    for (const kw of qkw) if (kw && text.includes(kw)) score += 3;

    const sim = (typeof d.similarity === 'number') ? d.similarity : Number(d.similarity || 0);
    if (!Number.isNaN(sim) && sim > 0) score += sim * 2;

    if (d.origin === 'web' || source.startsWith('http://') || source.startsWith('https://')) score += 1.0;
    if (text.length > 8000) score -= 1.5;

    return { d, score };
  });

  scored.sort((a, b) => b.score - a.score);
  const picked = scored.filter(x => x.score > 0).slice(0, Math.max(3, topK)).map(x => x.d);
  return picked.length ? picked : scored.slice(0, Math.max(3, topK)).map(x => x.d);
}

// ===================================================================
// Service principal
// ===================================================================

class MetricRAGChatService {
  constructor() {
    this.ragByAlgorithm = {};

    this.answerLanguage = (envGet('ANSWER_LANGUAGE', 'pt') || 'pt').toLowerCase();
    this.retrieveTopK = envInt('RETRIEVE_TOP_K', 6);
    this.minSimilarity = envFloat('MIN_LOCAL_SIMILARITY', 0.0);

    this.webRagEnabled = envBool('WEB_RAG_ENABLED', false);
    this.webRagMaxResults = envInt('WEB_RAG_MAX_RESULTS', 10);
    this.webRagEngine = envGet('WEB_RAG_ENGINE', 'multi');
    this.webRagMinLocalSim = envFloat('WEB_RAG_MIN_LOCAL_SIM', 0.75);

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
  }

  async ensureInitializedForAlgorithm(algorithmName) {
    if (!algorithmName) throw new Error('algorithmName é obrigatório');

    if (!this.ragByAlgorithm[algorithmName]) {
      const rag = new MetricRAGService(algorithmName);
      await rag.connect();
      this.ragByAlgorithm[algorithmName] = rag;
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

    // ✅ Guard (LLM) — opcional
    const guard = await llmGuardMetricQuestion({
      userText: question,
      metric,
      modelName: algorithm,
      answerLanguage: this.answerLanguage,
    });

    if (!guard.is_metric_question) {
      return this.answerLanguage === 'pt'
        ? '⚠️ A mensagem enviada não parece ser uma pergunta sobre métricas.'
        : '⚠️ Your message does not look like a metrics question.';
    }

    // ✅ NEW: classifica o “tipo” da pergunta para NÃO forçar definição
    const metricLabel = (metricName || metric || '').trim() || 'Latency';
    const qType = await llmClassifyQuestionType({
      userText: question,
      metric: metricLabel,
      modelName: algorithm,
      answerLanguage: this.answerLanguage,
    });
    const useAnalysisContract = qType.question_type === 'analysis_contract';

    // 1) traduz pergunta para EN
    const qEn = await translateText(question, 'en');

    // 2) system prompt
    const systemPrompt = loadSystemPromptSync(this.answerLanguage, useRag);

    // ----------------------------------------------------------------
    // NO-RAG
    // ----------------------------------------------------------------
    if (!useRag) {
      const contract = useAnalysisContract
        ? buildAnalysisContract(this.answerLanguage, { isRag: false })
        : buildAnswerContract(this.answerLanguage, { isRag: false });

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

      const rawAnswer = await generateWithLLM(systemPrompt, userPrompt, algorithm);
      if (!rawAnswer) return 'Não foi possível obter uma resposta da LLM no momento.';

      const cleaned = stripInternalCodesFromAnswer(rawAnswer);
      if (this.answerLanguage === 'pt') return translateText(cleaned, 'pt');
      return cleaned;
    }

    // ----------------------------------------------------------------
    // RAG local
    // ----------------------------------------------------------------
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

    // ----------------------------------------------------------------
    // WebRAG opcional
    // ----------------------------------------------------------------
    let webDocs = [];
    const shouldUseWeb =
      this.webRagEnabled &&
      typeof this.webRag?.query === 'function' &&
      typeof _fetch === 'function' &&
      (localDocs.length === 0 || bestLocalSim < this.webRagMinLocalSim);

    if (shouldUseWeb) {
      try {
        const webDocsRaw = await this.webRag.query(qEn, metric);
        webDocs = normalizeDocs(webDocsRaw, 'web');
      } catch (e) {
        console.warn('[MetricRAGChatService] WebRAG falhou:', e.message);
      }
    }

    // ----------------------------------------------------------------
    // combina e rerankeia
    // ----------------------------------------------------------------
    let combined = [...localDocs, ...webDocs];
    if (!combined.length) {
      return this.answerLanguage === 'pt'
        ? 'Não encontrei informação suficiente na base para responder a essa pergunta.'
        : 'I did not find enough information in the knowledge base to answer this question.';
    }

    combined = rerankDocs(combined, question, Math.max(6, this.retrieveTopK));

    // ----------------------------------------------------------------
    // contexto final (sem exigir citações)
    // ----------------------------------------------------------------
    const context = combined
      .map((d, idx) => {
        const header = `[#${idx + 1} | origin=${d.origin} | source=${d.source || ''} | page=${d.page ?? ''}]`;
        return `${header}\n${d.content || ''}`.trim();
      })
      .join('\n\n---\n\n');

    // ----------------------------------------------------------------
    // Prompt final RAG (sem referências, mas “strict context”)
    // ----------------------------------------------------------------
    const contract = useAnalysisContract
      ? buildAnalysisContract(this.answerLanguage, { isRag: true })
      : buildAnswerContract(this.answerLanguage, { isRag: true });

    const userPrompt = [
      `You are answering a question related to the metric "${metricLabel}".`,
      '',
      contract,
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

    const cleaned = stripInternalCodesFromAnswer(rawAnswer);
    if (this.answerLanguage === 'pt') return translateText(cleaned, 'pt');
    return cleaned;
  }
}

module.exports = MetricRAGChatService;
