'use strict'

/** @type {import('@adonisjs/lucid/src/HttpContext')} */
const Feedback = use('App/Models/Feedback')

class FeedbackController {
    async store({ request, response }) {
        try {
            const { form_id, responses } = request.post()

            if (!form_id || !responses || !Array.isArray(responses)) {
                return response.status(400).json({
                    message: 'Formato inválido. Certifique-se de enviar "form_id" e "responses".',
                })
            }

            const existingFeedback = await Feedback.query().where('form_id', form_id).first()

            if (existingFeedback) {
                return response.status(400).json({
                    message: `Já existe um conjunto de feedbacks registrado com o form_id "${form_id}".`,
                })
            }

            const feedbacks = responses.map((feedback) => ({
                form_id,
                question_id: feedback.question_id,
                question_type: feedback.question_type,
                question_text: feedback.question_text, // Incluindo o texto da pergunta
                response: feedback.response,
                comment: feedback.comment || null,
            }))

            await Feedback.createMany(feedbacks)

            return response.status(201).json({
                message: 'Feedback enviado com sucesso!',
                data: feedbacks,
            })
        } catch (error) {
            console.error('Erro ao salvar feedback:', error)
            return response.status(500).json({
                message: 'Erro interno ao salvar feedback.',
            })
        }
    }

    async index({ response }) {
        try {
            const feedbacks = await Feedback.query().orderBy('form_id').fetch()

            const groupedFeedbacks = feedbacks.toJSON().reduce((grouped, feedback) => {
                if (!grouped[feedback.form_id]) {
                    grouped[feedback.form_id] = []
                }
                grouped[feedback.form_id].push(feedback)
                return grouped
            }, {})

            return response.status(200).json({
                message: 'Feedbacks recuperados com sucesso.',
                data: groupedFeedbacks,
            })
        } catch (error) {
            console.error('Erro ao recuperar feedbacks:', error)
            return response.status(500).json({
                message: 'Erro interno ao recuperar feedbacks.',
            })
        }
    }
}

module.exports = FeedbackController
