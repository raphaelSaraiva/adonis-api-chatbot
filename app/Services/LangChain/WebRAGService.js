// services/WebRAGService.js
/* eslint-disable no-console */
'use strict';

const pdfParse = require('pdf-parse'); // necessário para ler PDF do arxiv

// Controla se vamos ou não baixar o PDF completo do arxiv.
// ARXIV_FETCH_PDF=true  -> tenta baixar PDF + extrair texto
// ARXIV_FETCH_PDF=false -> usa apenas título + resumo (abstract)
const ARXIV_FETCH_PDF = (process.env.ARXIV_FETCH_PDF || 'false').toLowerCase() === 'true';

// ===================================================================
// ✅ Helpers (genéricos)
// ===================================================================

function _normalizeForArxiv(s = '') {
  return String(s)
    .replace(/\s+/g, ' ')
    .replace(/"/g, '')
    .trim();
}

// Stopwords PT+EN (simples e eficiente)
function _extractKeywords(question, max = 6) {
  const q = String(question || '').toLowerCase();

  const stop = new Set([
    // EN
    'the','a','an','and','or','to','of','in','on','for','with','without','how','why','what','which','when',
    'does','do','did','are','is','was','were','be','been','being','between','across','from','into','by',
    'affect','impact','influence','cause','causes','causing','behavior','measurements','measurement',
    // PT
    'o','a','os','as','um','uma','uns','umas','de','da','do','das','dos','em','no','na','nos','nas',
    'para','por','com','sem','sobre','como','porque','porquê','qual','quais','quando','entre','afeta',
    'impacta','influencia','medicao','medição','medidas','comportamento'
  ]);

  const tokens = q
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .split(/\s+/)
    .map((t) => t.trim())
    .filter((t) => t && t.length >= 3 && !stop.has(t));

  const uniq = [];
  const seen = new Set();
  for (const t of tokens) {
    if (!seen.has(t)) {
      seen.add(t);
      uniq.push(t);
    }
    if (uniq.length >= max) break;
  }
  return uniq;
}

function _all(term) {
  const t = _normalizeForArxiv(term);
  if (!t) return '';
  return `all:"${t}"`;
}

function _nowMs() {
  return Date.now();
}

function _safeInt(v, fallback) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function _domainFromUrl(u) {
  try {
    const url = new URL(u);
    return (url.hostname || '').toLowerCase();
  } catch {
    return '';
  }
}

function _qualityFromUrl(url) {
  const u = String(url || '').toLowerCase();
  const host = _domainFromUrl(u);

  // alta autoridade acadêmica
  if (host.endsWith('arxiv.org')) return 3;
  if (host.endsWith('dl.acm.org') || host.endsWith('acm.org')) return 3;
  if (host.endsWith('ieee.org') || host.includes('ieeexplore')) return 3;
  if (host.endsWith('usenix.org')) return 3;
  if (host.includes('springer') || host.includes('sciencedirect')) return 3;

  // documentação / engenharia
  if (host.endsWith('github.com')) return 2;
  if (host.startsWith('docs.') || host.includes('readthedocs') || host.includes('developer.')) return 2;

  return 1;
}

// ===============================================================
// ✅ Query rewriting genérico (menos "genérico" na prática)
// - 2 queries (base + technical) por padrão
// - inclui termos de medição/experimento/config
// - permite 3ª query "implementation" via ENV (opcional)
// ===============================================================

function _buildWebQueriesGeneric(question, metric) {
  const q = String(question || '').trim();
  const m = String(metric || '').trim();

  const base = [q, m].filter(Boolean).join(' ').replace(/\s+/g, ' ').slice(0, 350);

  const enrich = [
    'benchmark',
    'evaluation',
    'measurement',
    'experiment',
    'results',
    'metrics',
    'methodology',
    'setup',
    'configuration',
    'parameters',
    'trade-offs',
    'limitations'
  ];

  // query técnica mais “densa”
  const technical = [q, m, ...enrich].filter(Boolean).join(' ').replace(/\s+/g, ' ').slice(0, 420);

  const enableTwo = (process.env.WEB_RAG_TWO_QUERIES || 'true').toLowerCase() === 'true';
  const enableThird = (process.env.WEB_RAG_THIRD_QUERY || 'false').toLowerCase() === 'true';

  const queries = [];
  queries.push(base);

  if (enableTwo && technical !== base) queries.push(technical);

  if (enableThird) {
    const impl = [q, m, 'implementation', 'engineering', 'tuning', 'production', 'troubleshooting']
      .filter(Boolean)
      .join(' ')
      .replace(/\s+/g, ' ')
      .slice(0, 420);
    if (impl !== base && impl !== technical) queries.push(impl);
  }

  // remove duplicatas
  const uniq = [];
  const seen = new Set();
  for (const x of queries) {
    const k = x.trim();
    if (!k) continue;
    if (seen.has(k)) continue;
    seen.add(k);
    uniq.push(k);
  }
  return uniq;
}

// ===============================================================
// ✅ Extração leve de HTML (genérico)
// ===============================================================

function _looksLikeHtmlNoise(text) {
  const t = String(text || '').toLowerCase();
  const badSignals = [
    'enable javascript',
    'sign up',
    'subscribe',
    'accept all cookies',
    'cookie policy',
    'privacy policy',
    'terms of service',
    'captcha',
    'unusual traffic',
    'please verify',
  ];
  return badSignals.some((s) => t.includes(s));
}

function _stripHtmlToText(html) {
  const text = String(html || '')
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<!--[\s\S]*?-->/g, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  return text;
}

function _chunkText(text, { maxChars = 2800, overlap = 200 } = {}) {
  const t = String(text || '').trim();
  if (!t) return [];
  if (t.length <= maxChars) return [t];

  const chunks = [];
  let i = 0;
  while (i < t.length) {
    const end = Math.min(i + maxChars, t.length);
    const slice = t.slice(i, end).trim();
    if (slice.length > 0) chunks.push(slice);
    if (end >= t.length) break;
    i = Math.max(0, end - overlap);
  }
  return chunks;
}

// Seleção genérica de parágrafos (arXiv PDF ou HTML longo)
// ✅ mais "informativo": prioriza sinais de medição/setup/resultados/limitações + presença de keywords
function _selectBestParagraphs(text, keywords, maxParas = 8) {
  const paras = String(text || '')
    .replace(/\r/g, '')
    .split(/\n{2,}/)
    .map((p) => p.trim())
    .filter((p) => p.length >= 220);

  if (!paras.length) return '';

  const kw = (Array.isArray(keywords) ? keywords : [])
    .map((x) => String(x || '').trim())
    .filter(Boolean);

  const evidenceTerms = [
    'benchmark','evaluation','measure','measurement','metrics','dataset','experiment','results','method',
    'methodology','setup','configuration','parameter','parameters','workload','throughput','latency',
    'performance','analysis','limitations','threats to validity','environment','deployment','topology'
  ];

  const scored = paras.map((p) => {
    const pl = p.toLowerCase();
    let s = 0;

    for (const k of kw) {
      const kl = k.toLowerCase();
      if (kl && pl.includes(kl)) s += 4;
    }
    for (const e of evidenceTerms) {
      if (pl.includes(e)) s += 1.6;
    }

    // bônus se tiver números (tende a ser mais técnico)
    if (/\b\d+(\.\d+)?\b/.test(pl)) s += 0.8;

    // penaliza parágrafo “muito curto / sem densidade”
    if (p.length < 350) s -= 0.6;

    return { p, s };
  });

  scored.sort((a, b) => b.s - a.s);
  const picked = scored.slice(0, maxParas).map((x) => x.p);

  return picked.join('\n\n').trim();
}

// ===============================================================
// ✅ Cache simples em memória (genérico)
// ===============================================================

class SimpleTTLCache {
  constructor({ ttlMs = 60 * 60 * 1000, maxEntries = 500 } = {}) {
    this.ttlMs = ttlMs;
    this.maxEntries = maxEntries;
    this.map = new Map(); // key -> { ts, value }
  }

  get(key) {
    const item = this.map.get(key);
    if (!item) return null;
    if (_nowMs() - item.ts > this.ttlMs) {
      this.map.delete(key);
      return null;
    }
    return item.value;
  }

  set(key, value) {
    if (this.map.size >= this.maxEntries) {
      let oldestK = null;
      let oldestTs = Infinity;
      for (const [k, v] of this.map.entries()) {
        if (v.ts < oldestTs) {
          oldestTs = v.ts;
          oldestK = k;
        }
      }
      if (oldestK != null) this.map.delete(oldestK);
    }
    this.map.set(key, { ts: _nowMs(), value });
  }
}

// ===================================================================
// ✅ WebRAGService (ajustado p/ ficar mais informativo e menos genérico)
// - fetch de páginas com "topN" e seleção de parágrafos
// - score por qualidade + sinais técnicos + overlap de keywords
// - dedup mais esperto (por URL e engine)
// - mantém docs com trecho (EXCERPT) mais “denso”
// ===================================================================

class WebRAGService {
  constructor({
    enabled,
    endpoint,              // endpoint "default" (Tavily, por ex.)
    apiKey,                // apiKey "default"
    maxResults = 5,
    fetchImpl,
    tavilyEndpoint,
    tavilyApiKey,
    braveEndpoint,
    braveApiKey,
    arxivEndpoint,
    serperEndpoint,
    serperApiKey,
    engine = 'tavily',     // 'tavily' | 'brave' | 'arxiv' | 'serper' | 'multi'
  }) {
    this.fetch = fetchImpl;
    this.maxResults = maxResults;

    // engine atual (ou "multi" se for combinar todos)
    this.engine = engine;

    // Bases configuradas via ENV para o modo "multi"
    // Ex.: WEB_RAG_BASES=tavily,arxiv,brave,serper
    this.bases = (process.env.WEB_RAG_BASES || 'tavily,brave,arxiv,serper')
      .split(',')
      .map((s) => s.trim().toLowerCase())
      .filter(Boolean);

    // Tavily
    this.tavilyEndpoint = tavilyEndpoint || endpoint || process.env.TAVILY_ENDPOINT;
    this.tavilyApiKey = tavilyApiKey || apiKey || process.env.TAVILY_API_KEY;

    // Brave
    this.braveEndpoint = braveEndpoint || process.env.BRAVE_SEARCH_ENDPOINT;
    this.braveApiKey = braveApiKey || process.env.BRAVE_SEARCH_API_KEY;

    // Arxiv
    this.arxivEndpoint = arxivEndpoint || process.env.ARXIV_ENDPOINT;

    // Serper.dev
    this.serperEndpoint = serperEndpoint || process.env.SERPER_ENDPOINT;
    this.serperApiKey = serperApiKey || process.env.SERPER_API_KEY;

    this.enabled =
      !!enabled &&
      typeof this.fetch === 'function' &&
      (
        this.tavilyEndpoint ||
        this.braveEndpoint ||
        this.arxivEndpoint ||
        this.serperEndpoint
      );

    // ---------- genéricos via ENV ----------
    this.webFetchPages = (process.env.WEB_RAG_FETCH_PAGES || 'true').toLowerCase() === 'true';
    this.webFetchTopN = _safeInt(process.env.WEB_RAG_FETCH_TOPN, 2); // por engine/query
    this.webMaxChunksPerUrl = _safeInt(process.env.WEB_RAG_MAX_CHUNKS_PER_URL, 1);
    this.webMaxTotalDocs = _safeInt(process.env.WEB_RAG_MAX_TOTAL_DOCS, Math.max(this.maxResults * 2, 10));

    this.chunkMaxChars = _safeInt(process.env.WEB_RAG_CHUNK_MAX_CHARS, 2800);
    this.chunkOverlap = _safeInt(process.env.WEB_RAG_CHUNK_OVERLAP, 200);

    // ✅ controla “densidade” do trecho retornado
    this.pageTextMaxChars = _safeInt(process.env.WEB_RAG_PAGE_TEXT_MAX_CHARS, 9000);
    this.pageExcerptMaxChars = _safeInt(process.env.WEB_RAG_PAGE_EXCERPT_MAX_CHARS, 3800);

    // cache
    const cacheTtlSec = _safeInt(process.env.WEB_RAG_CACHE_TTL_SEC, 3600);
    const cacheMax = _safeInt(process.env.WEB_RAG_CACHE_MAX_ENTRIES, 500);
    this.cache = new SimpleTTLCache({ ttlMs: cacheTtlSec * 1000, maxEntries: cacheMax });
  }

  _buildBaseQuery(question) {
    const q = `${question}`.trim();
    return q.replace(/\s+/g, ' ').slice(0, 400);
  }

  /**
   * Fetch leve do conteúdo de página (HTML -> texto)
   */
  async _tryFetchPageText(url) {
    if (!this.fetch || !this.webFetchPages) return null;

    const u = String(url || '').trim();
    if (!u) return null;

    // evita arquivos binários comuns
    if (/\.(pdf|zip|png|jpg|jpeg|gif|webp)$/i.test(u)) return null;

    try {
      const res = await this.fetch(u, { method: 'GET' });
      if (!res.ok) return null;

      const html = await res.text();
      if (!html || html.length < 800) return null;

      const text = _stripHtmlToText(html);
      if (!text || text.length < 900) return null;
      if (_looksLikeHtmlNoise(text)) return null;

      return text.slice(0, Math.max(this.pageTextMaxChars, 2000));
    } catch {
      return null;
    }
  }

  /**
   * Normaliza docs e aplica chunking leve + qualidade + metadados
   */
  _normalizeAndChunkDocs(rawDocs, { engine, query, metric } = {}) {
    const out = [];
    if (!Array.isArray(rawDocs) || rawDocs.length === 0) return out;

    const maxPerUrl = Math.max(this.webMaxChunksPerUrl, 1);

    for (let i = 0; i < rawDocs.length; i++) {
      const d = rawDocs[i] || {};
      const source = String(d.source || d.url || d.link || d.title || 'web');
      const content = String(d.content || d.text || d.snippet || d.summary || '').trim();
      if (!content || content.length < 180) continue;

      const quality = _qualityFromUrl(source);
      const host = _domainFromUrl(source);

      const chunks = _chunkText(content, {
        maxChars: this.chunkMaxChars,
        overlap: this.chunkOverlap,
      }).slice(0, maxPerUrl);

      chunks.forEach((c, idx) => {
        if (!c || c.length < 180) return;
        out.push({
          content: c,
          source,
          chunk_index: idx,
          engine: engine || d.engine,
          web_query: query || d._webQuery || '',
          metric: metric || '',
          quality,
          host,
        });
      });
    }

    return out;
  }

  /**
   * Score genérico de “densidade técnica” para priorizar conteúdo útil ao RAG
   */
  _technicalDensityScore(text, keywords = []) {
    const t = String(text || '').toLowerCase();
    if (!t) return 0;

    const evidenceTerms = [
      'benchmark','evaluation','measurement','metrics','experiment','results','method','methodology',
      'setup','configuration','parameter','workload','latency','throughput','performance','analysis',
      'limitations','threats to validity','deployment','topology','consensus','validator','block'
    ];

    let s = 0;

    // overlap de keywords da pergunta
    const kw = (Array.isArray(keywords) ? keywords : []).map(k => String(k || '').toLowerCase()).filter(Boolean);
    for (const k of kw) {
      if (k && t.includes(k)) s += 2.5;
    }

    // termos técnicos genéricos
    for (const e of evidenceTerms) {
      if (t.includes(e)) s += 1.1;
    }

    // números costumam significar resultados/parametrização
    if (/\b\d+(\.\d+)?\b/.test(t)) s += 0.7;

    // penaliza ruído óbvio
    if (_looksLikeHtmlNoise(t)) s -= 3;

    return s;
  }

  /**
   * Dedup por URL e limita total
   * - prefere maior quality
   * - prefere maior densidade técnica
   * - prefere chunk 0
   */
  _dedupAndLimit(docs, { keywords = [] } = {}) {
    if (!Array.isArray(docs) || docs.length === 0) return [];

    const byUrl = new Map();

    for (const d of docs) {
      const key = String(d.source || '').trim();
      if (!key) continue;

      const prev = byUrl.get(key);
      if (!prev) {
        byUrl.set(key, d);
        continue;
      }

      const prevQ = _safeInt(prev.quality, 1);
      const curQ = _safeInt(d.quality, 1);

      const prevD = this._technicalDensityScore(prev.content, keywords);
      const curD = this._technicalDensityScore(d.content, keywords);

      // escolhe melhor (qualidade > densidade > chunk 0 > tamanho)
      if (curQ > prevQ) byUrl.set(key, d);
      else if (curQ === prevQ) {
        if (curD > prevD) byUrl.set(key, d);
        else if (curD === prevD) {
          if (_safeInt(d.chunk_index, 0) < _safeInt(prev.chunk_index, 0)) byUrl.set(key, d);
          else if ((d.content || '').length > (prev.content || '').length) byUrl.set(key, d);
        }
      }
    }

    const unique = Array.from(byUrl.values());

    unique.sort((a, b) => {
      const qa = _safeInt(a.quality, 1);
      const qb = _safeInt(b.quality, 1);
      if (qb !== qa) return qb - qa;

      const da = this._technicalDensityScore(a.content, keywords);
      const db = this._technicalDensityScore(b.content, keywords);
      if (db !== da) return db - da;

      return (b.content || '').length - (a.content || '').length;
    });

    return unique.slice(0, Math.max(this.webMaxTotalDocs, 10));
  }

  /**
   * Interface pública
   */
  async query(question, metric) {
    if (!this.enabled) return [];

    const queries = _buildWebQueriesGeneric(question, metric);
    const baseQuery = this._buildBaseQuery(question);
    const kw = _extractKeywords(`${question} ${metric || ''}`, 10);

    const cacheKey = `${this.engine}::${metric || ''}::${queries.join('||')}`;
    const cached = this.cache.get(cacheKey);
    if (cached) {
      console.log(`🌐 WebRAG cache HIT (${this.engine}) -> ${cached.length}`);
      return cached;
    }

    const runMany = async (fn, engineName) => {
      const all = [];
      for (const q of (queries.length ? queries : [baseQuery])) {
        const docsRaw = await fn.call(this, q, metric);
        const normalized = this._normalizeAndChunkDocs(docsRaw, {
          engine: engineName,
          query: q,
          metric,
        });
        all.push(...normalized);
      }
      return all;
    };

    try {
      // MULTI
      if (this.engine === 'multi') {
        const results = [];
        const stats = {};

        const collect = (docs, engineName) => {
          const count = Array.isArray(docs) ? docs.length : 0;
          stats[engineName] = count;

          if (count > 0) {
            results.push(
              ...docs.map((doc, idx) => ({
                ...doc,
                source: `[${engineName}] ${doc.source || 'web'}`,
                chunk_index: doc.chunk_index ?? idx,
                engine: engineName,
              }))
            );
          }
        };

        const wanted = (this.bases && this.bases.length)
          ? this.bases
          : ['tavily', 'brave', 'arxiv', 'serper'];

        if (wanted.includes('tavily')) collect(await runMany(this._callTavily, 'tavily'), 'tavily');
        else stats.tavily = 0;

        if (wanted.includes('brave')) collect(await runMany(this._callBrave, 'brave'), 'brave');
        else stats.brave = 0;

        if (wanted.includes('arxiv')) collect(await runMany(this._callArxiv, 'arxiv'), 'arxiv');
        else stats.arxiv = 0;

        if (wanted.includes('serper')) collect(await runMany(this._callSerper, 'serper'), 'serper');
        else stats.serper = 0;

        console.log('🌐 WebRAG bases:', wanted);
        console.log('🌐 WebRAG stats:', stats);
        console.log(`🌐 WebRAG[multi] TOTAL (pré-dedup): ${results.length}`);

        const finalDocs = this._dedupAndLimit(results, { keywords: kw });

        console.log(`🌐 WebRAG[multi] TOTAL (pós-dedup): ${finalDocs.length}`);
        this.cache.set(cacheKey, finalDocs);
        return finalDocs;
      }

      // ENGINE ÚNICO
      if (this.engine === 'tavily') {
        const docs = await runMany(this._callTavily, 'tavily');
        const finalDocs = this._dedupAndLimit(docs, { keywords: kw });
        this.cache.set(cacheKey, finalDocs);
        return finalDocs;
      }

      if (this.engine === 'brave') {
        const docs = await runMany(this._callBrave, 'brave');
        const finalDocs = this._dedupAndLimit(docs, { keywords: kw });
        this.cache.set(cacheKey, finalDocs);
        return finalDocs;
      }

      if (this.engine === 'arxiv') {
        const docs = await runMany(this._callArxiv, 'arxiv');
        const finalDocs = this._dedupAndLimit(docs, { keywords: kw });
        this.cache.set(cacheKey, finalDocs);
        return finalDocs;
      }

      if (this.engine === 'serper') {
        const docs = await runMany(this._callSerper, 'serper');
        const finalDocs = this._dedupAndLimit(docs, { keywords: kw });
        this.cache.set(cacheKey, finalDocs);
        return finalDocs;
      }

      // fallback
      const docs = await runMany(this._callTavily, 'tavily');
      const finalDocs = this._dedupAndLimit(docs, { keywords: kw });
      this.cache.set(cacheKey, finalDocs);
      return finalDocs;
    } catch (e) {
      console.warn(`⚠️ Erro no WebRAG[${this.engine}]: ${e.message}`);
      return [];
    }
  }

  // ---------------- TAVILY (POST JSON) ----------------
  async _callTavily(query) {
    if (!this.fetch || !this.tavilyApiKey || !this.tavilyEndpoint) {
      console.log('WebRAG[tavily] desabilitado (sem fetch, key ou endpoint).');
      return [];
    }

    try {
      const body = {
        api_key: this.tavilyApiKey,
        query,
        max_results: this.maxResults,
        search_depth: 'advanced',
        include_answer: false,
      };

      const res = await this.fetch(this.tavilyEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      console.log('WebRAG[tavily] HTTP', res.status, res.statusText);

      if (!res.ok) {
        console.warn('WebRAG[tavily] body erro:', await res.text());
        return [];
      }

      const json = await res.json();
      const raw = json.results || json.data || json.documents || [];
      console.log('WebRAG[tavily] raw length:', raw.length);

      // ✅ tenta enriquecer com fetch de páginas nos top N
      const mapped = [];
      const kw = _extractKeywords(query, 10);

      for (let idx = 0; idx < raw.length; idx++) {
        const item = raw[idx] || {};
        const link = item.url || item.source || '';
        const title = (item.title || '').toString();
        const snippet = (item.content || item.text || item.snippet || item.summary || '').toString();

        let content = `${title}\n\n${snippet}`.trim();

        if (this.webFetchPages && link && idx < Math.max(this.webFetchTopN, 0)) {
          const pageText = await this._tryFetchPageText(link);
          if (pageText) {
            const selected = _selectBestParagraphs(pageText, kw, 8);
            const excerpt = (selected && selected.length > 300) ? selected : pageText.slice(0, this.pageExcerptMaxChars);
            content = `${title}\n\n${snippet}\n\nEXCERPT:\n${excerpt}`.trim();
          }
        }

        if (!content || content.trim().length < 200) continue;

        mapped.push({
          content,
          source: link || item.title || 'tavily',
          chunk_index: idx,
        });
      }

      console.log('WebRAG[tavily] mapped length:', mapped.length);
      return mapped;
    } catch (e) {
      console.warn(`⚠️ WebRAG[tavily] erro: ${e.message}`);
      return [];
    }
  }

  // ---------------- BRAVE (GET com ?q=...) ----------------
  async _callBrave(query) {
    if (!this.fetch || !this.braveApiKey || !this.braveEndpoint) {
      console.log('WebRAG[brave] desabilitado (sem fetch, key ou endpoint).');
      return [];
    }

    try {
      const url = new URL(this.braveEndpoint);
      const safeCount = Number.isFinite(Number(this.maxResults)) ? Number(this.maxResults) : 5;

      url.searchParams.set('q', query || '');
      url.searchParams.set('count', String(safeCount));

      console.log('WebRAG[brave] URL:', url.toString());

      const res = await this.fetch(url.toString(), {
        method: 'GET',
        headers: {
          'X-Subscription-Token': this.braveApiKey,
          'Accept': 'application/json',
          'User-Agent': 'sworker-metric-bot/1.0',
        },
      });

      console.log('WebRAG[brave] HTTP', res.status, res.statusText);

      if (!res.ok) {
        const body = await res.text();
        console.warn('WebRAG[brave] body erro:', body);
        return [];
      }

      const json = await res.json();
      const web = json.web && Array.isArray(json.web.results) ? json.web.results : [];
      console.log('WebRAG[brave] raw length:', web.length);

      const mapped = [];
      const kw = _extractKeywords(query, 10);

      for (let idx = 0; idx < web.length; idx++) {
        const item = web[idx] || {};
        const link = item.url || item.link || '';
        const title = (item.title || '').toString();
        const snippet = (item.description || '').toString();

        let content = `${title}\n\n${snippet}`.trim();
        if (!content || content.length < 150) continue;

        if (this.webFetchPages && link && idx < Math.max(this.webFetchTopN, 0)) {
          const pageText = await this._tryFetchPageText(link);
          if (pageText) {
            const selected = _selectBestParagraphs(pageText, kw, 8);
            const excerpt = (selected && selected.length > 300) ? selected : pageText.slice(0, this.pageExcerptMaxChars);
            content = `${title}\n\n${snippet}\n\nEXCERPT:\n${excerpt}`.trim();
          }
        }

        if (!content || content.trim().length < 200) continue;

        mapped.push({
          content,
          source: link || 'brave',
          chunk_index: idx,
        });
      }

      console.log('WebRAG[brave] mapped length:', mapped.length);
      return mapped;
    } catch (e) {
      console.warn(`⚠️ WebRAG[brave] erro: ${e.message}`);
      return [];
    }
  }

  // ---------------- SERPER (POST JSON) ----------------
  async _callSerper(query) {
    if (!this.fetch || !this.serperApiKey || !this.serperEndpoint) {
      console.log('WebRAG[serper] desabilitado (sem fetch, key ou endpoint).');
      return [];
    }

    try {
      const body = {
        q: query || '',
        num: this.maxResults,
      };

      const res = await this.fetch(this.serperEndpoint, {
        method: 'POST',
        headers: {
          'X-API-KEY': this.serperApiKey,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      });

      console.log('WebRAG[serper] HTTP', res.status, res.statusText);

      if (!res.ok) {
        const text = await res.text();
        console.warn('WebRAG[serper] body erro:', text);
        return [];
      }

      const json = await res.json();
      const organic = Array.isArray(json.organic) ? json.organic : [];
      console.log('WebRAG[serper] raw length:', organic.length);

      const mapped = [];
      const kw = _extractKeywords(query, 10);

      for (let idx = 0; idx < organic.length; idx++) {
        const item = organic[idx] || {};
        const title = (item.title || '').toString();
        const snippet = (item.snippet || item.description || '').toString();
        const link = item.link || item.url || '';

        let content = `${title}\n\n${snippet}`.trim();
        if (!content || content.length < 150) continue;

        if (this.webFetchPages && link && idx < Math.max(this.webFetchTopN, 0)) {
          const pageText = await this._tryFetchPageText(link);
          if (pageText) {
            const selected = _selectBestParagraphs(pageText, kw, 8);
            const excerpt = (selected && selected.length > 300) ? selected : pageText.slice(0, this.pageExcerptMaxChars);
            content = `${title}\n\n${snippet}\n\nEXCERPT:\n${excerpt}`.trim();
          }
        }

        if (!content || content.length < 200) continue;

        mapped.push({
          content,
          source: link || 'serper',
          chunk_index: idx,
        });
      }

      console.log('WebRAG[serper] mapped length:', mapped.length);
      return mapped;
    } catch (e) {
      console.warn(`⚠️ WebRAG[serper] erro: ${e.message}`);
      return [];
    }
  }

  // ---------------- ARXIV (Atom + PDF OPCIONAL) ----------------
  async _callArxiv(query, metric) {
    if (!this.fetch || !this.arxivEndpoint) {
      console.log('WebRAG[arxiv] desabilitado (sem fetch ou endpoint).');
      return [];
    }

    try {
      const base = _normalizeForArxiv(query);
      const metricTerm = _normalizeForArxiv(metric);

      const kw = _extractKeywords(base, 10);

      // Query arXiv: all:"blockchain" AND ( all:"<metric>" OR all:"kw1" OR ... )
      const must = [_all('blockchain')].filter(Boolean);

      const should = [];
      if (metricTerm) should.push(_all(metricTerm));
      for (const k of kw) should.push(_all(k));

      let mainQuery = '';
      if (should.length) mainQuery = `${must.join(' AND ')} AND (${should.join(' OR ')})`;
      else mainQuery = `${must.join(' AND ')}`;

      const q = encodeURIComponent(mainQuery);
      const url = `${this.arxivEndpoint}?search_query=${q}&start=0&max_results=${this.maxResults}`;

      console.log('WebRAG[arxiv] mainQuery:', mainQuery);
      console.log('WebRAG[arxiv] URL:', url);
      console.log('WebRAG[arxiv] ARXIV_FETCH_PDF:', ARXIV_FETCH_PDF);

      const res = await this.fetch(url, {
        method: 'GET',
        headers: { 'Accept': 'application/atom+xml,text/xml' },
      });

      console.log('WebRAG[arxiv] HTTP', res.status, res.statusText);

      if (!res.ok) {
        console.warn(`⚠️ WebRAG[arxiv] HTTP ${res.status}: ${res.statusText}`);
        return [];
      }

      const xml = await res.text();
      const entries = xml.split('<entry>').slice(1);

      const docs = [];
      let idx = 0;

      for (const entry of entries) {
        const titleMatch = entry.match(/<title[^>]*>([\s\S]*?)<\/title>/i);
        const summaryMatch = entry.match(/<summary[^>]*>([\s\S]*?)<\/summary>/i);
        const idMatch = entry.match(/<id>([^<]+)<\/id>/i);

        const pdfLinkMatch = entry.match(/<link[^>]*title="pdf"[^>]*href="([^"]+)"[^>]*>/i);
        const altLinkMatch = entry.match(/<link[^>]*rel="alternate"[^>]*href="([^"]+)"[^>]*>/i);

        const rawTitle = titleMatch ? titleMatch[1] : '';
        const rawSummary = summaryMatch ? summaryMatch[1] : '';

        const title = rawTitle.trim().replace(/\s+/g, ' ');
        const summary = rawSummary.trim().replace(/\s+/g, ' ');

        const pageLink = altLinkMatch
          ? altLinkMatch[1].trim()
          : (idMatch ? idMatch[1].trim() : 'arxiv');

        let pdfUrl = pdfLinkMatch ? pdfLinkMatch[1].trim() : null;
        if (!pdfUrl && idMatch) {
          const idUrl = idMatch[1].trim();
          if (idUrl.includes('/abs/')) pdfUrl = idUrl.replace('/abs/', '/pdf/') + '.pdf';
        }

        if (!title && !summary) continue;

        let baseContent = `${title}\n\n${summary}`.trim();
        if (!baseContent || baseContent.length < 50) baseContent = title || summary || '';

        let fullText = '';
        let usedPdf = false;

        // ✅ PDF: seleciona melhores parágrafos ao invés de despejar texto inteiro
        if (ARXIV_FETCH_PDF && pdfUrl) {
          try {
            console.log('WebRAG[arxiv] Baixando PDF:', pdfUrl);
            const pdfRes = await this.fetch(pdfUrl, { method: 'GET' });

            if (!pdfRes.ok) {
              console.warn(`⚠️ WebRAG[arxiv] falha ao baixar PDF (${pdfRes.status}): ${pdfRes.statusText}`);
            } else {
              const arrayBuffer = await pdfRes.arrayBuffer();
              const buffer = Buffer.from(arrayBuffer);
              const parsed = await pdfParse(buffer);

              const raw = (parsed.text || '')
                .replace(/\r/g, '')
                .replace(/[ \t]+\n/g, '\n')
                .replace(/\n{3,}/g, '\n\n')
                .trim();

              if (raw && raw.length > 500) {
                const selected = _selectBestParagraphs(raw, [metricTerm, ...kw].filter(Boolean), 10);
                if (selected && selected.length > 350) {
                  fullText = selected;
                  usedPdf = true;
                }
              }
            }
          } catch (err) {
            console.warn('⚠️ WebRAG[arxiv] erro ao processar PDF:', err.message);
          }
        }

        const content = usedPdf
          ? `${title}\n\n${summary}\n\nEXCERPT:\n${fullText}`.trim()
          : baseContent;

        if (!content || content.length < 80) continue;

        // filtro final: exige "blockchain" E (métrica OU keyword)
        const lower = content.toLowerCase();
        const hasBlockchain = lower.includes('blockchain');

        const mustSignals = [];
        if (metricTerm) mustSignals.push(metricTerm.toLowerCase());
        mustSignals.push(...kw.map((x) => x.toLowerCase()));

        const hasAnySignal = mustSignals.length
          ? mustSignals.some((term) => term && lower.includes(term))
          : true;

        if (!hasBlockchain || !hasAnySignal) continue;

        docs.push({
          content,
          source: pageLink || pdfUrl || 'arxiv',
          chunk_index: idx++,
        });
      }

      console.log('WebRAG[arxiv] mapped length:', docs.length);
      return docs;
    } catch (e) {
      console.warn(`⚠️ WebRAG[arxiv] erro: ${e.message}`);
      return [];
    }
  }
}

module.exports = WebRAGService;
