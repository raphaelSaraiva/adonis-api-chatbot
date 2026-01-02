const PdfQAService = require('../LangChain/PdfQAService');
const fs = require('fs');
const levenshtein = require('fast-levenshtein');
const rouge = require('rouge');
const axios = require('axios');

class LLMTestingService {
  constructor() {
    this.pdfQAService = new PdfQAService();
    this.testResults = [];
  }

  async initialize() {
    await this.pdfQAService.ensureInitialized();
    console.log('Serviço de teste inicializado.');
  }

  async testSingleModel(question, expectedAnswer, modelName) {
    const startTime = Date.now();
    const generatedAnswer = await this.pdfQAService.askQuestion(question, modelName);
    const endTime = Date.now();

    const timeTaken = endTime - startTime;
    const isCorrect = this.isAnswerCorrect(generatedAnswer, expectedAnswer);

    // Métricas Automáticas
    const levenshteinDistance = levenshtein.get(generatedAnswer.trim(), expectedAnswer.trim());
    const precision = this.calculatePrecision(generatedAnswer, expectedAnswer);
    const recall = this.calculateRecall(generatedAnswer, expectedAnswer);
    const f1Score = this.calculateF1Score(precision, recall);

    const bleuScore = this.calculateBLEU(generatedAnswer, expectedAnswer);
    const rougeScores = this.calculateROUGE(generatedAnswer, expectedAnswer);
    const bertScore = await this.calculateBERTScore(generatedAnswer, expectedAnswer);

    return {
      modelName,
      question,
      expectedAnswer,
      generatedAnswer,
      isCorrect,
      timeTaken,
      levenshteinDistance,
      precision,
      recall,
      f1Score,
      bleuScore,
      rougeScores,
      bertScore,
    };
  }

  async runBatchTests(testCases) {
    const results = [];
    for (const modelName of this.pdfQAService.models) {
      console.log('run batch model:::', modelName);
      for (const testCase of testCases) {
        console.log(`Testando modelo ${modelName} com pergunta: ${testCase.question}`);
        const result = await this.testSingleModel(testCase.question, testCase.expectedAnswer, modelName);
        results.push(result);
      }
    }
    this.testResults = results;
    return results;
  }

  isAnswerCorrect(generatedAnswer, expectedAnswer) {
    return generatedAnswer.trim().toLowerCase() === expectedAnswer.trim().toLowerCase();
  }

  calculatePrecision(generatedAnswer, expectedAnswer) {
    const generatedTokens = new Set(generatedAnswer.toLowerCase().split(/\s+/));
    const expectedTokens = new Set(expectedAnswer.toLowerCase().split(/\s+/));
    const truePositives = [...generatedTokens].filter((token) => expectedTokens.has(token)).length;
    return truePositives / generatedTokens.size || 0;
  }

  calculateRecall(generatedAnswer, expectedAnswer) {
    const generatedTokens = new Set(generatedAnswer.toLowerCase().split(/\s+/));
    const expectedTokens = new Set(expectedAnswer.toLowerCase().split(/\s+/));
    const truePositives = [...generatedTokens].filter((token) => expectedTokens.has(token)).length;
    return truePositives / expectedTokens.size || 0;
  }

  calculateF1Score(precision, recall) {
    if (precision + recall === 0) return 0;
    return (2 * precision * recall) / (precision + recall);
  }

  calculateBLEU(generatedAnswer, expectedAnswer) {
    const tokenize = (text) => text.toLowerCase().split(/\s+/);
    
    const nGramCounts = (tokens, n) => {
      const counts = {};
      for (let i = 0; i <= tokens.length - n; i++) {
        const nGram = tokens.slice(i, i + n).join(' ');
        counts[nGram] = (counts[nGram] || 0) + 1;
      }
      return counts;
    };
  
    const clipCounts = (candidateCounts, referenceCounts) => {
      const clipped = {};
      for (const nGram in candidateCounts) {
        clipped[nGram] = Math.min(candidateCounts[nGram], referenceCounts[nGram] || 0);
      }
      return clipped;
    };
  
    const precision = (candidateTokens, referenceTokens, n) => {
      const candidateCounts = nGramCounts(candidateTokens, n);
      const referenceCounts = nGramCounts(referenceTokens, n);
      const clippedCounts = clipCounts(candidateCounts, referenceCounts);
      const clippedSum = Object.values(clippedCounts).reduce((a, b) => a + b, 0);
      const total = Object.values(candidateCounts).reduce((a, b) => a + b, 0);
      return total > 0 ? clippedSum / total : 0;
    };
  
    const candidateTokens = tokenize(generatedAnswer);
    const referenceTokens = tokenize(expectedAnswer);
  
    const precisions = [1, 2, 3, 4].map((n) => precision(candidateTokens, referenceTokens, n));
  
    // Cálculo da média geométrica das precisões
    const geometricMean = precisions.reduce((a, b) => a * (b || 1), 1) ** (1 / precisions.length);
  
    // Penalidade por comprimento
    const brevityPenalty = Math.exp(1 - Math.max(1, referenceTokens.length / candidateTokens.length));
  
    // BLEU Score
    return brevityPenalty * geometricMean;
  }

  calculateROUGE(generatedAnswer, expectedAnswer) {
    // Tokeniza as strings de entrada
    const tokenize = (text) => text.toLowerCase().match(/\w+/g) || [];
  
    const calculateOverlap = (candidateTokens, referenceTokens, n = 1) => {
      const nGrams = (tokens, n) => {
        const nGramsList = [];
        for (let i = 0; i <= tokens.length - n; i++) {
          nGramsList.push(tokens.slice(i, i + n).join(' '));
        }
        return nGramsList;
      };
  
      const candidateNGrams = nGrams(candidateTokens, n);
      const referenceNGrams = nGrams(referenceTokens, n);
  
      const intersection = candidateNGrams.filter((nGram) => referenceNGrams.includes(nGram));
      return {
        overlap: intersection.length,
        totalCandidate: candidateNGrams.length,
        totalReference: referenceNGrams.length,
      };
    };
  
    const calculateROUGE1 = (candidateTokens, referenceTokens) => {
      const { overlap, totalCandidate, totalReference } = calculateOverlap(candidateTokens, referenceTokens, 1);
      const precision = overlap / totalCandidate || 0;
      const recall = overlap / totalReference || 0;
      const f1 = (2 * precision * recall) / (precision + recall) || 0;
      return { precision, recall, f1 };
    };
  
    const calculateROUGE2 = (candidateTokens, referenceTokens) => {
      const { overlap, totalCandidate, totalReference } = calculateOverlap(candidateTokens, referenceTokens, 2);
      const precision = overlap / totalCandidate || 0;
      const recall = overlap / totalReference || 0;
      const f1 = (2 * precision * recall) / (precision + recall) || 0;
      return { precision, recall, f1 };
    };
  
    const calculateLCS = (candidateTokens, referenceTokens) => {
      const cLen = candidateTokens.length;
      const rLen = referenceTokens.length;
      const table = Array.from({ length: cLen + 1 }, () => Array(rLen + 1).fill(0));
  
      for (let i = 1; i <= cLen; i++) {
        for (let j = 1; j <= rLen; j++) {
          if (candidateTokens[i - 1] === referenceTokens[j - 1]) {
            table[i][j] = table[i - 1][j - 1] + 1;
          } else {
            table[i][j] = Math.max(table[i - 1][j], table[i][j - 1]);
          }
        }
      }
  
      const lcsLength = table[cLen][rLen];
      const precision = lcsLength / cLen || 0;
      const recall = lcsLength / rLen || 0;
      const f1 = (2 * precision * recall) / (precision + recall) || 0;
      return { precision, recall, f1 };
    };
  
    const candidateTokens = tokenize(generatedAnswer);
    const referenceTokens = tokenize(expectedAnswer);
  
    const rouge1 = calculateROUGE1(candidateTokens, referenceTokens);
    const rouge2 = calculateROUGE2(candidateTokens, referenceTokens);
    const rougeL = calculateLCS(candidateTokens, referenceTokens);
  
    return {
      rouge1,
      rouge2,
      rougeL,
    };
  }

  async calculateBERTScore(generatedAnswer, expectedAnswer) {
    const apiKey = process.env.HUGGING_FACE_API_KEY;
    const model = 'bert-base-uncased'; // Substitua por um modelo relevante
    const url = `https://api-inference.huggingface.co/models/${model}`;

    const headers = {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    };

    const payload = {
      inputs: {
        source_sentence: expectedAnswer,
        candidate_sentence: generatedAnswer,
      },
    };

    try {
      const response = await axios.post(url, payload, { headers });
      return response.data.score || 0; // Ajuste conforme o retorno da API
    } catch (error) {
      console.error('Erro ao calcular BERTScore:', error.message || error);
      return 0;
    }
  }

  generateReport() {
    const totalTests = this.testResults.length;
    const correctAnswers = this.testResults.filter((result) => result.isCorrect).length;
    const accuracy = (correctAnswers / totalTests) * 100;
    const averageTime = this.testResults.reduce((sum, result) => sum + result.timeTaken, 0) / totalTests;
    const averageLevenshtein = this.testResults.reduce((sum, result) => sum + result.levenshteinDistance, 0) / totalTests;

    return {
      totalTests,
      correctAnswers,
      accuracy: `${accuracy.toFixed(2)}%`,
      averageTime: `${averageTime.toFixed(2)}ms`,
      averageLevenshtein: averageLevenshtein.toFixed(2),
    };
  }

  saveResultsToFile(filePath) {
    fs.writeFileSync(filePath, JSON.stringify(this.testResults, null, 2), 'utf8');
    console.log(`Resultados salvos em: ${filePath}`);
  }
}

module.exports = LLMTestingService;
