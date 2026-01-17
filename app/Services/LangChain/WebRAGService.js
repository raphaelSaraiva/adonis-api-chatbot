// services/WebRAGService.js
/* eslint-disable no-console */
'use strict';

const pdfParse = require('pdf-parse'); // <-- necessário para ler PDF do arxiv

// Controla se vamos ou não baixar o PDF completo do arxiv.
// ARXIV_FETCH_PDF=true  -> tenta baixar PDF + extrair texto
// ARXIV_FETCH_PDF=false -> usa apenas título + resumo (abstract)
const ARXIV_FETCH_PDF = (process.env.ARXIV_FETCH_PDF || 'false').toLowerCase() === 'true';

// ===================================================================
// ✅ Helpers (arXiv query baseada na PERGUNTA)
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

  // Remove duplicados preservando ordem
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

    // Tavily (mantido)
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
  }

  /**
   * Query "normalizada" que será enviada para os motores
   */
  _buildBaseQuery(question, metric) {
    const q = `${question}`.trim();

    // sem quebras de linha loucas (Brave é chato com isso)
    return q.replace(/\s+/g, ' ').slice(0, 400); // limita tamanho
  }

  /**
   * Interface pública
   */
  async query(question, metric) {
    if (!this.enabled) return [];

    const baseQuery = this._buildBaseQuery(question, metric);

    try {
      if (this.engine === 'multi') {
        const results = [];
        const stats = {}; // <--- aqui vamos contar por engine

        const collect = (docs, engine) => {
          const count = Array.isArray(docs) ? docs.length : 0;
          stats[engine] = count;

          if (count > 0) {
            results.push(
              ...docs.map((doc, idx) => ({
                ...doc,
                source: `[${engine}] ${doc.source || 'web'}`,
                chunk_index: doc.chunk_index ?? idx,
                engine, // <--- salva de onde veio
              }))
            );
          }
        };

        // quais bases usar neste modo multi
        const wanted = (this.bases && this.bases.length)
          ? this.bases
          : ['tavily', 'brave', 'arxiv', 'serper'];

        // chama cada engine individualmente, respeitando WEB_RAG_BASES
        if (wanted.includes('tavily')) {
          collect(await this._callTavily(baseQuery), 'tavily');
        } else {
          stats.tavily = 0;
        }

        if (wanted.includes('brave')) {
          collect(await this._callBrave(baseQuery), 'brave');
        } else {
          stats.brave = 0;
        }

        if (wanted.includes('arxiv')) {
          collect(await this._callArxiv(baseQuery, metric), 'arxiv');
        } else {
          stats.arxiv = 0;
        }

        if (wanted.includes('serper')) {
          collect(await this._callSerper(baseQuery), 'serper');
        } else {
          stats.serper = 0;
        }

        console.log('🌐 WebRAG bases:', wanted);
        console.log('🌐 WebRAG stats:', stats);
        console.log(`🌐 WebRAG[multi] TOTAL retornado: ${results.length}`);

        return results;
      }

      // engine único
      if (this.engine === 'tavily') return this._callTavily(baseQuery);
      if (this.engine === 'brave') return this._callBrave(baseQuery);
      if (this.engine === 'arxiv') return this._callArxiv(baseQuery, metric);
      if (this.engine === 'serper') return this._callSerper(baseQuery);

      // fallback: só Tavily
      return this._callTavily(baseQuery);
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

      const mapped = raw
        .map((item, idx) => {
          const content = (
            item.content ||
            item.text ||
            item.snippet ||
            item.summary ||
            ''
          ).toString();

          if (!content || content.trim().length < 200) {
            return null;
          }

          return {
            content,
            source: item.url || item.source || item.title || 'tavily',
            chunk_index: idx,
          };
        })
        .filter(Boolean);

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

      // ✅ FIX: não use process.env diretamente (pode virar "undefined")
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

      // estrutura típica: { web: { results: [...] } }
      const web = json.web && Array.isArray(json.web.results) ? json.web.results : [];
      console.log('WebRAG[brave] raw length:', web.length);

      const mapped = web
        .map((item, idx) => {
          const content = (item.title + '\n\n' + (item.description || '')).toString();
          if (!content || content.trim().length < 200) return null;
          return {
            content,
            source: item.url || item.link || 'brave',
            chunk_index: idx,
          };
        })
        .filter(Boolean);

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
        num: this.maxResults, // quantos resultados orgânicos você quer
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

      // Serper normalmente retorna: { "organic": [ ... ] }
      const organic = Array.isArray(json.organic) ? json.organic : [];
      console.log('WebRAG[serper] raw length:', organic.length);

      const mapped = organic
        .map((item, idx) => {
          const title = (item.title || '').toString();
          const snippet = (item.snippet || item.description || '').toString();
          const content = `${title}\n\n${snippet}`.trim();

          if (!content || content.length < 150) return null;

          return {
            content,
            source: item.link || item.url || 'serper',
            chunk_index: idx,
          };
        })
        .filter(Boolean);

      console.log('WebRAG[serper] mapped length:', mapped.length);
      return mapped;
    } catch (e) {
      console.warn(`⚠️ WebRAG[serper] erro: ${e.message}`);
      return [];
    }
  }

  // ---------------- ARXIV (Atom + PDF OPCIONAL, com query baseada na PERGUNTA + métrica) ----------------
  async _callArxiv(query, metric) {
    if (!this.fetch || !this.arxivEndpoint) {
      console.log('WebRAG[arxiv] desabilitado (sem fetch ou endpoint).');
      return [];
    }

    try {
      const base = _normalizeForArxiv(query);
      const metricTerm = _normalizeForArxiv(metric);

      // ✅ Extrai keywords da pergunta (query já vem "normalizada" do chat)
      const kw = _extractKeywords(base, 6);

      // ✅ Query arXiv: all:"blockchain" AND ( all:"Latency" OR all:"clock" OR ... )
      const must = [_all('blockchain')].filter(Boolean);

      const should = [];
      if (metricTerm) should.push(_all(metricTerm));
      for (const k of kw) should.push(_all(k));

      let mainQuery = '';
      if (should.length) {
        mainQuery = `${must.join(' AND ')} AND (${should.join(' OR ')})`;
      } else {
        mainQuery = `${must.join(' AND ')}`;
      }

      const q = encodeURIComponent(mainQuery);
      const url = `${this.arxivEndpoint}?search_query=${q}&start=0&max_results=${this.maxResults}`;

      console.log('WebRAG[arxiv] mainQuery:', mainQuery);
      console.log('WebRAG[arxiv] URL:', url);
      console.log('WebRAG[arxiv] ARXIV_FETCH_PDF:', ARXIV_FETCH_PDF);

      const res = await this.fetch(url, {
        method: 'GET',
        headers: {
          'Accept': 'application/atom+xml,text/xml',
        },
      });

      console.log('WebRAG[arxiv] HTTP', res.status, res.statusText);

      if (!res.ok) {
        console.warn(`⚠️ WebRAG[arxiv] HTTP ${res.status}: ${res.statusText}`);
        return [];
      }

      const xml = await res.text();

      // parsing: pega <entry>...
      const entries = xml.split('<entry>').slice(1);
      const docs = [];
      let idx = 0;

      for (const entry of entries) {
        const titleMatch = entry.match(/<title[^>]*>([\s\S]*?)<\/title>/i);
        const summaryMatch = entry.match(/<summary[^>]*>([\s\S]*?)<\/summary>/i);
        const idMatch = entry.match(/<id>([^<]+)<\/id>/i);

        // link "pdf", se existir
        const pdfLinkMatch = entry.match(
          /<link[^>]*title="pdf"[^>]*href="([^"]+)"[^>]*>/i
        );
        // link "alternate" (página HTML do artigo)
        const altLinkMatch = entry.match(
          /<link[^>]*rel="alternate"[^>]*href="([^"]+)"[^>]*>/i
        );

        const rawTitle = titleMatch ? titleMatch[1] : '';
        const rawSummary = summaryMatch ? summaryMatch[1] : '';

        const title = rawTitle.trim().replace(/\s+/g, ' ');
        const summary = rawSummary.trim().replace(/\s+/g, ' ');

        const pageLink = altLinkMatch
          ? altLinkMatch[1].trim()
          : (idMatch ? idMatch[1].trim() : 'arxiv');

        // tenta descobrir URL do PDF
        let pdfUrl = pdfLinkMatch ? pdfLinkMatch[1].trim() : null;
        if (!pdfUrl && idMatch) {
          const idUrl = idMatch[1].trim(); // ex: http://arxiv.org/abs/xxxx.xxxx
          if (idUrl.includes('/abs/')) {
            pdfUrl = idUrl.replace('/abs/', '/pdf/') + '.pdf';
          }
        }

        if (!title && !summary) continue;

        // base mínima: título + resumo
        let baseContent = `${title}\n\n${summary}`.trim();
        if (!baseContent || baseContent.length < 50) {
          baseContent = title || summary || '';
        }

        let fullText = '';
        let usedPdf = false;

        // Tenta baixar o PDF e extrair texto APENAS se ARXIV_FETCH_PDF=true
        if (ARXIV_FETCH_PDF && pdfUrl) {
          try {
            console.log('WebRAG[arxiv] Baixando PDF:', pdfUrl);

            const pdfRes = await this.fetch(pdfUrl, {
              method: 'GET',
            });

            if (!pdfRes.ok) {
              console.warn(
                `⚠️ WebRAG[arxiv] falha ao baixar PDF (${pdfRes.status}): ${pdfRes.statusText}`
              );
            } else {
              const arrayBuffer = await pdfRes.arrayBuffer();
              const buffer = Buffer.from(arrayBuffer);
              const parsed = await pdfParse(buffer);

              fullText = (parsed.text || '')
                .replace(/\r/g, '')
                .replace(/[ \t]+\n/g, '\n')
                .replace(/\n{3,}/g, '\n\n')
                .trim();

              const MAX_CHARS = 20000; // evita explodir contexto
              if (fullText.length > MAX_CHARS) {
                fullText = fullText.slice(0, MAX_CHARS) + '\n\n[...]';
              }

              if (fullText && fullText.length > 200) {
                usedPdf = true;
              }
            }
          } catch (err) {
            console.warn('⚠️ WebRAG[arxiv] erro ao processar PDF:', err.message);
          }
        }

        // Monta conteúdo final
        let content;
        if (usedPdf) {
          content = `${title}\n\n${summary}\n\n${fullText}`.trim();
        } else {
          // Modo "abstract only" (ou fallback se PDF falhar)
          content = baseContent;
        }

        if (!content || content.length < 50) continue;

        // ✅ Filtro final: exige "blockchain" E (métrica OU keyword da pergunta)
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
