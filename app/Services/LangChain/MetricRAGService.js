// services/MetricRAGService.js
/* eslint-disable no-console */
'use strict';

const { Client } = require('pg');
const fs = require('fs');
const path = require('path');
const pdfParse = require('pdf-parse');
const { RecursiveCharacterTextSplitter } = require('@langchain/textsplitters');

const { OllamaEmbeddings } = require('@langchain/ollama');
const { OpenAIEmbeddings } = require('@langchain/openai');

const FIXED_DIM = 1536; // padronizamos tudo para 1536 (simples e robusto com pgvector)

/**
 * Serviço de ingestão/consulta RAG por métrica.
 * – Ingestão: lê PDFs em ./docs/<Métrica>/*.pdf, cria chunks com page + chunk_index e salva embeddings (vector(1536)).
 * – Consulta: dado metric + pergunta, retorna topK chunks com {id, source, page, chunk_index, content}.
 */
class MetricRAGService {
  constructor(algorithmName = 'llama2') {
    console.log('================= [MetricRAGService] CONSTRUCTOR =================');
    console.log('[MetricRAGService] algorithmName =', algorithmName);
    console.log('[MetricRAGService] DB_CONNECTION_STRING =', process.env.DB_CONNECTION_STRING);
    console.log('[MetricRAGService] OLLAMA_HOST =', process.env.OLLAMA_HOST);
    console.log('[MetricRAGService] OPENAI_API_KEY set? =', !!process.env.OPENAI_API_KEY);

    this.algorithmName = algorithmName;

    try {
      console.log('[MetricRAGService] criando Client PG...');
      this.client = new Client({
        connectionString: process.env.DB_CONNECTION_STRING,
      });
      console.log('[MetricRAGService] Client PG criado com sucesso');
    } catch (err) {
      console.error('[MetricRAGService] ERRO ao criar Client PG:', err);
      throw err;
    }

    try {
      console.log('[MetricRAGService] chamando getEmbeddingsForModel...');
      this.embeddings = this.getEmbeddingsForModel(algorithmName);
      console.log('[MetricRAGService] embeddings criados com sucesso:', typeof this.embeddings);
    } catch (err) {
      console.error('[MetricRAGService] ERRO ao criar embeddings:', err);
      throw err;
    }

    this.modelId = null;
    this.dimension = FIXED_DIM; // usamos dimensão fixa
    console.log('[MetricRAGService] dimension fixada em', this.dimension);
    console.log('===================================================================');
  }

  // ---------------- Embeddings ----------------
  getEmbeddingsForModel(modelName) {
    const m = (modelName || '').toLowerCase();
    console.log('----- [MetricRAGService] getEmbeddingsForModel -----');
    console.log('[MetricRAGService] modelName recebido =', modelName);
    console.log('[MetricRAGService] modelName lower   =', m);

    const ollamaBaseUrl = process.env.OLLAMA_HOST || 'http://127.0.0.1:11434';
    console.log('[MetricRAGService] ollamaBaseUrl =', ollamaBaseUrl);

    switch (m) {
      case 'llama2':
      case 'llama3':
      case 'qwen2.5-7b-instruct':
        console.log('[MetricRAGService] escolhendo OllamaEmbeddings (nomic-embed-text)');
        return new OllamaEmbeddings({
          model: 'nomic-embed-text',
          baseUrl: ollamaBaseUrl,
        });

      case 'openai':
        console.log('[MetricRAGService] escolhendo OpenAIEmbeddings');
        return new OpenAIEmbeddings({
          apiKey: process.env.OPENAI_API_KEY,
        });

      default: {
        console.warn(
          `⚠️ [MetricRAGService] Unsupported embedding model "${modelName}", ` +
          `fallback para OllamaEmbeddings (nomic-embed-text).`
        );
        return new OllamaEmbeddings({
          model: 'nomic-embed-text',
          baseUrl: ollamaBaseUrl,
        });
      }
    }
  }

  // ---------------- Conexão ----------------
  async connect() {
    console.log('----- [MetricRAGService] connect() chamado -----');
    try {
      await this.client.connect();
      console.log('✅ [MetricRAGService] Connected to database');
    } catch (err) {
      console.error('❌ [MetricRAGService] ERRO ao conectar no banco:', err);
      throw err;
    }
  }

  async disconnect() {
    console.log('----- [MetricRAGService] disconnect() chamado -----');
    try {
      await this.client.end();
      console.log('🔌 [MetricRAGService] Disconnected from database');
    } catch (err) {
      console.error('❌ [MetricRAGService] ERRO ao desconectar do banco:', err);
      throw err;
    }
  }

  // ---------------- Modelo/Dimensão ----------------
  async ensureModelRegistered() {
    console.log('----- [MetricRAGService] ensureModelRegistered() -----');
    console.log('[MetricRAGService] algorithmName =', this.algorithmName);

    try {
      const { rows } = await this.client.query(
        'SELECT id FROM embedding_models WHERE name = $1',
        [this.algorithmName]
      );
      console.log('[MetricRAGService] embedding_models rows length =', rows.length);

      if (rows.length > 0) {
        console.log('[MetricRAGService] Modelo já registrado no banco.');
        return;
      }

      console.log('[MetricRAGService] Registrando novo embedding model no banco...');
      await this.client.query(
        `INSERT INTO embedding_models (name, description, dimension)
         VALUES ($1, $2, $3)`,
        [this.algorithmName, `Auto-registered (fixed-dim=${FIXED_DIM})`, FIXED_DIM]
      );
      console.log(
        `📌 [MetricRAGService] Registered embedding model: ${this.algorithmName} (dim=${FIXED_DIM})`
      );
    } catch (err) {
      console.error('❌ [MetricRAGService] ERRO em ensureModelRegistered:', err);
      throw err;
    }
  }

  async getModelId() {
    console.log('----- [MetricRAGService] getModelId() -----');
    try {
      await this.ensureModelRegistered();

      const { rows } = await this.client.query(
        'SELECT id, dimension FROM embedding_models WHERE name = $1',
        [this.algorithmName]
      );

      console.log('[MetricRAGService] getModelId: rows length =', rows.length);

      if (rows.length === 0) {
        console.error(
          `❌ [MetricRAGService] Embedding model not found in DB: ${this.algorithmName}`
        );
        throw new Error(`❌ Embedding model not found: ${this.algorithmName}`);
      }

      this.modelId = rows[0].id;
      this.dimension = FIXED_DIM; // mantemos fixa
      console.log('[MetricRAGService] modelId =', this.modelId);
      console.log('[MetricRAGService] dimension =', this.dimension);
    } catch (err) {
      console.error('❌ [MetricRAGService] ERRO em getModelId:', err);
      throw err;
    }
  }

  // ---------------- Util ----------------
  _sanitizeForPostgres(str) {
    if (typeof str !== 'string') return '';
    let s = str.replace(/\u0000/g, '');
    try { s = s.normalize('NFC'); } catch (_) { }
    s = s.replace(/[\u0001-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]/g, '');
    s = s.replace(/[ \t]+\n/g, '\n').replace(/[ \t]{2,}/g, ' ');
    return s.trim();
  }

  _padOrTrunc(vec, dim = FIXED_DIM) {
    if (!Array.isArray(vec)) return new Array(dim).fill(0);
    if (vec.length === dim) return vec;
    if (vec.length > dim) return vec.slice(0, dim);
    const out = vec.slice();
    while (out.length < dim) out.push(0);
    return out;
  }

  // Tenta separar texto por páginas a partir do pdf-parse (muitos PDFs usam \f)
  async _pdfToPages(buffer) {
    console.log('----- [MetricRAGService] _pdfToPages() -----');
    const data = await pdfParse(buffer);
    const raw = (data.text || '').replace(/\r\n/g, '\n');
    let parts = raw.split('\f').map(t => t.trim()).filter(Boolean);
    if (parts.length === 0) {
      // fallback: sem separador de página; devolve tudo como página 1
      parts = [raw.trim()];
    }
    console.log('[MetricRAGService] _pdfToPages → total pages =', parts.length);
    return parts.map((t, i) => ({ page: i + 1, text: t }));
  }

  async _chunkText(text, { chunkSize = 2000, chunkOverlap = 200 } = {}) {
    console.log('----- [MetricRAGService] _chunkText() -----');
    const splitter = new RecursiveCharacterTextSplitter({ chunkSize, chunkOverlap });
    const chunks = await splitter.splitText(text);
    console.log('[MetricRAGService] _chunkText → total chunks =', chunks.length);
    return chunks.map(c => this._sanitizeForPostgres(c)).filter(Boolean);
  }

  // ---------------- Ingestão ----------------
  async addPDFDocument(metric, pdfPath, opts = {}) {
    try {
      console.log(`📄 [MetricRAGService] Processing PDF: ${pdfPath} (metric=${metric})`);
      const abs = path.resolve(process.cwd(), pdfPath);
      console.log('[MetricRAGService] PDF absolute path =', abs);

      const buffer = fs.readFileSync(abs);
      const pages = await this._pdfToPages(buffer); // [{page, text}]

      let chunkIndex = 0; // sequencial no documento inteiro
      for (const { page, text } of pages) {
        console.log(`[MetricRAGService] Processando page=${page}, text length=${text.length}`);
        const chunks = await this._chunkText(text, opts);
        console.log(
          `[MetricRAGService] page=${page} → ${chunks.length} chunks (chunkIndex inicia em ${chunkIndex})`
        );

        for (const chunk of chunks) {
          let embedding = await this.embeddings.embedQuery(chunk);
          embedding = this._padOrTrunc(embedding, this.dimension);
          const vectorString = `[${embedding.join(',')}]`;

          await this.client.query(
            `INSERT INTO metrics_documents
             (model_id, metric, source, page, chunk_index, content, embedding)
             VALUES ($1,$2,$3,$4,$5,$6,$7::vector)`,
            [this.modelId, metric, pdfPath, page, chunkIndex++, chunk, vectorString]
          );
        }
      }
      console.log(`📌 [MetricRAGService] Inserido PDF: ${pdfPath} → metric=${metric}`);
    } catch (err) {
      console.error(`❌ [MetricRAGService] Error processing PDF "${pdfPath}": ${err.message}`, err);
    }
  }

  async ingestAllDocuments(docsRoot = './docs', opts = {}) {
    console.log('----- [MetricRAGService] ingestAllDocuments() -----');
    console.log('[MetricRAGService] docsRoot =', docsRoot);

    await this.getModelId();

    const metricDirs = fs.readdirSync(docsRoot, { withFileTypes: true })
      .filter((d) => d.isDirectory())
      .map((d) => d.name);

    console.log('[MetricRAGService] metricDirs encontrados:', metricDirs);

    for (const metric of metricDirs) {
      const metricPath = path.join(docsRoot, metric);
      const pdfFiles = fs.readdirSync(metricPath)
        .filter((f) => f.toLowerCase().endsWith('.pdf'));

      console.log(`📘 [MetricRAGService] Métrica "${metric}" — ${pdfFiles.length} PDFs`);
      for (const pdf of pdfFiles) {
        await this.addPDFDocument(metric, path.join(metricPath, pdf), opts);
      }
    }
    console.log('✅ [MetricRAGService] Finished ingesting all documents');
  }

  // ---------------- Consulta ----------------
  async queryDocuments(metric, question, topK = 3, options = {}) {
    console.log('----- [MetricRAGService] queryDocuments() -----');
    console.log('[MetricRAGService] params =', { metric, questionSnippet: (question || '').slice(0, 80), topK, options });

    const { minSimilarity = 0.0, fetchMultiplier = 3 } = options;
    console.log('[MetricRAGService] minSimilarity =', minSimilarity, 'fetchMultiplier =', fetchMultiplier);

    await this.getModelId();

    console.log('[MetricRAGService] Gerando embedding da pergunta...');
    let qVec = await this.embeddings.embedQuery(question);
    console.log('[MetricRAGService] qVec length (antes do pad) =', Array.isArray(qVec) ? qVec.length : 'not array');
    qVec = this._padOrTrunc(qVec, this.dimension);
    console.log('[MetricRAGService] qVec length (depois do pad) =', qVec.length);

    const qStr = `[${qVec.join(',')}]`;
    console.log('[MetricRAGService] qStr length (string) =', qStr.length);

    console.log('[MetricRAGService] Executando SELECT em metrics_documents...');
    const res = await this.client.query(
      `SELECT
         id,
         source,
         page,
         chunk_index,
         content,
         embedding <=> $3::vector AS distance
       FROM metrics_documents
      WHERE model_id = $1
        AND metric = $2
      ORDER BY distance
      LIMIT $4`,
      [this.modelId, metric, qStr, topK * fetchMultiplier] // busca mais p/ poder filtrar
    );

    console.log('[MetricRAGService] rows retornadas =', res.rows.length);

    const docs = res.rows.map(r => {
      const distance = Number(r.distance);
      const similarity = 1 / (1 + distance); // ~0–1 (maior = mais parecido)
      return {
        id: r.id,
        source: r.source,
        page: typeof r.page === 'number' ? r.page : null,
        chunk_index: typeof r.chunk_index === 'number' ? r.chunk_index : null,
        content: r.content,
        distance,
        similarity,
      };
    });

    console.log('[MetricRAGService] docs (com similarity) length =', docs.length);
    console.log('[MetricRAGService] similarity stats (min/max) =', {
      min: docs.length ? Math.min(...docs.map(d => d.similarity)) : null,
      max: docs.length ? Math.max(...docs.map(d => d.similarity)) : null,
    });

    // filtra por similaridade mínima e pega os topK finais
    const filtered = docs
      .filter(d => d.similarity >= minSimilarity)
      .slice(0, topK);

    console.log('[MetricRAGService] filtered length =', filtered.length);
    console.log('----- [MetricRAGService] queryDocuments() FIM -----');

    return filtered;
  }

}

module.exports = MetricRAGService;
