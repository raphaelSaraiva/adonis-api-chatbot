// start/pdfService.js
'use strict';

require('dotenv').config(); // ✅ garante process.env antes de qualquer coisa

const MetricRAGChatService = use('App/Services/LangChain/MetricRAGChatService');

const chatService = new MetricRAGChatService();

async function initializeService() {
  try {
    if (!process.env.DB_CONNECTION_STRING) {
      throw new Error('DB_CONNECTION_STRING não está definido. Verifique o .env');
    }

    const defaultAlg = (process.env.DEFAULT_ALGORITHM || process.env.DEFAULT_MODEL || 'llama2').trim();
    await chatService.ensureInitializedForAlgorithm(defaultAlg);

    console.log('✅ MetricRAGChatService inicializado para:', defaultAlg);
  } catch (error) {
    console.error('Erro ao inicializar MetricRAGChatService global:', error.message || error);
    process.exit(1);
  }
}

initializeService();
module.exports = chatService;
