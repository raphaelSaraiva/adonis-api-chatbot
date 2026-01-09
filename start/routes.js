'use strict'

/*
|--------------------------------------------------------------------------
| Routes
|--------------------------------------------------------------------------
|
| Http routes are entry points to your web application. You can create
| routes for different URL's and bind Controller actions to them.
|
| A complete guide on routing is available here.
| http://adonisjs.com/docs/4.1/routing
|
*/

/** @type {typeof import('@adonisjs/framework/src/Route/Manager')} */
const Route = use('Route')

Route.on('/').render('welcome')

// Rota para processar perguntas
Route.post('/ask-question', 'QuestionController.askQuestion');
Route.get('/feedbacks', 'FeedbackController.index') // Recupera todos os feedbacks
Route.post('/feedbacks', 'FeedbackController.store') // Salva feedbacks

// Rota para testes
Route.post('/llm-tests', 'LLMTestingController.runTests');

Route.get('/health', 'HealthController.index').middleware(['health'])