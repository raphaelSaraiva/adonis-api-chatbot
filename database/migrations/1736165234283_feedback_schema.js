'use strict'

/** @type {import('@adonisjs/lucid/src/Schema')} */
const Schema = use('Schema')

class FeedbackSchema extends Schema {
  up () {
    this.create('feedbacks', (table) => {
      table.increments() // ID único
      table.string('form_id').notNullable() // Identificador único do formulário
      table.string('question_id').notNullable() // ID da pergunta
      table.string('question_type').notNullable() // Tipo da pergunta (technical ou evaluation)
      table.text('question_text').notNullable() // Texto da pergunta
      table.text('response').notNullable() // Resposta do usuário
      table.text('comment').nullable() // Comentário adicional
      table.timestamps() // Campos created_at e updated_at
    })
  }

  down () {
    this.drop('feedbacks')
  }
}

module.exports = FeedbackSchema
