'use strict'

class HealthcheckMiddleware {
  async handle ({ request, response }, next) {
    // evita que o Adonis tente criar sessão/cookies
    request.request.session = null
    request.request._adonisSession = null

    // responde direto (não chama next)
    return response.status(200).json({
      status: 'ok',
      service: 'adonis-api-chatbot',
      timestamp: new Date().toISOString(),
    })
  }
}

module.exports = HealthcheckMiddleware
