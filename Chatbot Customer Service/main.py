# BOT-API KEY
import Constants as keys
from telegram import Update
# pip install python-telegram-bot
from telegram.ext import *
# from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import Responses as R

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def start_command(update, context):
    update.message.reply_text("Type something to start")

def help_command(update, context):
    update.message.reply_text("No help")

def handle_command(update, context):
    text = str(update.message.text).lower()
    response = R.sample_responses(text)
    print("Me: ",text)
    print("Bot: ", response)
    update.message.reply_text(response)

def error(update, context):
    print(f"Update {update} caused error {context.error}")
    update.message.reply_text("maaf untuk pertanyaan bapak/ibu masih belum dapat dimengerti oleh chatbot. Atas ketidakmampuan chatbot maka bapak/ibu dapat lansung menghubungi admin melalui no hp (0801241530453)")
def main():
    updater = Updater(keys.API_KEY, use_context = True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start_command))
    dp.add_handler(CommandHandler("start", help_command))

    dp.add_handler(MessageHandler(Filters.text, handle_command))

    dp.add_error_handler(error)
    updater.start_polling()
    updater.idle()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('User')
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
