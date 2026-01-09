'use strict'

class HealthController {
  async index ({ response }) {
    return response.status(200).json({
      status: 'ok',
      service: 'adonis-api-chatbot',
      timestamp: new Date().toISOString(),
    })
  }
}

module.exports = HealthController
