const LLMTestingService = use('App/Services/Tests/LLMTestingService');

class LLMTestingController {
  /**
   * Inicializa e executa os testes com métricas detalhadas.
   */
  async runTests({ request, response }) {
    const { testCases } = request.post();

    // Validação inicial
    if (!testCases || !Array.isArray(testCases)) {
      return response.status(400).json({
        message: 'Formato inválido. Certifique-se de enviar "testCases" como uma lista.',
      });
    }

    const testService = new LLMTestingService();
    await testService.initialize();

    try {
      // Executar os testes
      const results = await testService.runBatchTests(testCases);

      // Gerar relatório detalhado com métricas
      const report = testService.generateReport();

      // Retornar os resultados e métricas detalhadas
      return response.status(200).json({
        message: 'Testes concluídos com sucesso.',
        report,
        results,
      });
    } catch (error) {
      console.error('Erro ao executar os testes:', error);

      return response.status(500).json({
        message: 'Erro interno ao executar os testes.',
        error: error.message || error,
      });
    }
  }
}

module.exports = LLMTestingController;
