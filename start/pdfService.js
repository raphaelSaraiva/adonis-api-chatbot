const MetricRAGChatService = use('App/Services/LangChain/MetricRAGChatService');

// Instância global do PdfQAService
const qaService = new MetricRAGChatService();

async function initializeService() {
  try {
    console.log('Inicializando o PdfQAService globalmente...');
    await qaService.ensureInitialized(); // Garante que o serviço esteja inicializado
    console.log('PdfQAService inicializado globalmente.');
  } catch (error) {
    console.error('Erro ao inicializar o PdfQAService global:', error.message || error);
    // Opcional: Encerra a aplicação caso a inicialização seja crítica
    process.exit(1);
  }
}

// Inicializa o serviço
initializeService();

module.exports = qaService; // Exporta a instância para uso em outros módulos
