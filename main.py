import telebot 

bot = telebot.TeleBot('5234132360:AAGCgEIJDnB9cwZay_XA7Uy9kfJTdDMwvYc')

@bot.message_handler(commads=["start"])
def start(m, res=False):
	bot.send_message(m.chat.id, 'Привет! Я бот, который поможет решить твои уравнения!)')

@bot.message_handler(content_types=["text"])
def hande_text(message):
	bot.send_message(message.chat.id, 'Вы написали: ' + message.text)


# Запуск
bot.polling(none_stop=True, interval=0)