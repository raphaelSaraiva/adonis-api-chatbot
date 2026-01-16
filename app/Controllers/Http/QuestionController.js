'use strict';

const MetricRAGChatService = use('App/Services/LangChain/MetricRAGChatService');
const metricRAGChatService = new MetricRAGChatService();

class QuestionController {
  async askQuestion({ request, response }) {
    console.log('endpoint /ask-question chamado');

    const question = String(request.input('question') || '').trim();
    const modelName = String(request.input('model') || 'llama2').trim();
    const mode = String(request.input('mode') || 'both').toLowerCase(); // both | rag | norag

    // ✅ recebe metricId/metricName (compat com legado "metric")
    const metricId = String(request.input('metricId') || '').trim();
    const metricName = String(request.input('metricName') || '').trim();
    const legacyMetric = String(request.input('metric') || '').trim();

    // ✅ regra final de métrica
    const metric = metricName || legacyMetric || metricId || 'Latency';

    // ✅ histórico (array ou string JSON)
    let history = request.input('history');
    if (history == null) history = request.input('chatHistory');
    if (history == null) history = [];

    if (typeof history === 'string') {
      try {
        history = JSON.parse(history);
      } catch (e) {
        console.warn('[QuestionController] history veio como string mas falhou JSON.parse. Ignorando.');
        history = [];
      }
    }

    if (!Array.isArray(history)) {
      console.warn('[QuestionController] history não é array. Ignorando.');
      history = [];
    }

    const HISTORY_MAX_TURNS = 10;
    history = history.slice(-HISTORY_MAX_TURNS);

    console.log('[QuestionController] payload:', {
      question,
      modelName,
      mode,
      metric,
      metricId: metricId || null,
      metricName: metricName || null,
      historyCount: history.length,
    });

    if (!question) {
      return response.status(400).send({ error: 'Pergunta é obrigatória.' });
    }

    try {
      console.log('[QuestionController] ensureInitialized...');
      const algorithmName = String(modelName || 'llama2').trim();
      await metricRAGChatService.ensureInitializedForAlgorithm(algorithmName);
      console.log('[QuestionController] ensureInitialized OK');

      let response_rag = null;
      let response_norag = null;

      const preview = (txt, n = 400) => {
        if (!txt) return '';
        const s = String(txt);
        return s.length > n ? `${s.slice(0, n)}... [${s.length} chars]` : s;
      };

      const baseOptions = {
        metric,
        metricId: metricId || null,
        metricName: metricName || null,
        history,
        algorithm: modelName,
      };

      if (mode === 'both' || mode === 'rag') {
        console.log('[QuestionController] gerando resposta 1 (useRag=true)...');
        response_rag = await metricRAGChatService.askQuestion(question, {
          ...baseOptions,
          useRag: true,
        });
        console.log('[QuestionController] ✅ resposta (RAG) gerada!');
        console.log('[QuestionController] RAG len =', (response_rag || '').length);
        console.log('[QuestionController] RAG preview:\n', preview(response_rag));
      }

      if (mode === 'both' || mode === 'norag') {
        console.log('[QuestionController] gerando resposta 2 (useRag=false)...');
        response_norag = await metricRAGChatService.askQuestion(question, {
          ...baseOptions,
          useRag: false,
        });
        console.log('[QuestionController] ✅ resposta (NO-RAG) gerada!');
        console.log('[QuestionController] NO-RAG len =', (response_norag || '').length);
        console.log('[QuestionController] NO-RAG preview:\n', preview(response_norag));
      }

      // =========================
      // ✅ Retorno FINAL (limpo)
      // =========================

      if (mode === 'rag') {
        return response.status(200).send({
          question,
          model: modelName,
          metric,
          metricId: metricId || '',
          metricName: metricName || '',
          mode,
          historyCount: history.length,
          response_rag: response_rag || '',
        });
      }

      if (mode === 'norag') {
        return response.status(200).send({
          question,
          model: modelName,
          metric,
          metricId: metricId || '',
          metricName: metricName || '',
          mode,
          historyCount: history.length,
          response_norag: response_norag || '',
        });
      }

      // mode === 'both'
      return response.status(200).send({
        question,
        model: modelName,
        metric,
        metricId: metricId || '',
        metricName: metricName || '',
        mode,
        historyCount: history.length,
        response_rag: response_rag || '',
        response_norag: response_norag || '',
      });
    } catch (error) {
      console.error('[QuestionController] Erro ao processar a pergunta:', error.message || error);
      return response.status(500).send({
        error: 'Erro interno ao processar a pergunta. Tente novamente mais tarde.',
      });
    }
  }
}

module.exports = QuestionController;
