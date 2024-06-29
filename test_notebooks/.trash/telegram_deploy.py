import os

import yaml
from dotenv import load_dotenv
from src.utils.utils import fix_text
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

ENV_PATH = "../.env/"

load_dotenv(f"{ENV_PATH}vkr.env")

threshold = 0.4

with open("strings/cedr.yaml") as f:
    cedr_strings = yaml.safe_load(f)

with open("strings/ru-go-emotions.yaml") as f:
    ru_go_emotions_strings = yaml.safe_load(f)

with open("strings/russian-sentiment.yaml") as f:
    russian_sentiment_strings = yaml.safe_load(f)

cedr = pipeline(model="seara/rubert-tiny2-cedr", device=0)
ru_go_emotions = pipeline(model="seara/rubert-tiny2-ru-go-emotions", device=0)
russian_sentiment = pipeline(model="seara/rubert-tiny2-russian-sentiment", device=0)


# tokenizer = AutoTokenizer.from_pretrained("VadimAI/Dialog-system")
# model = AutoModelForCausalLM.from_pretrained("VadimAI/Dialog-system")
# model.cuda()

# tokenizer.pad_token = tokenizer.eos_token

print("Loaded models")


def get_emotions(text, threshold, top_k=3):
    gm = ru_go_emotions(text, top_k=top_k, truncation=True)
    gm_processed = [item for item in gm if item["score"] >= threshold]
    if not gm_processed:
        gm_processed.append(gm[0])

    cr = cedr(text, top_k=top_k, truncation=True)
    cr_processed = [item for item in cr if item["score"] >= threshold]
    if not cr_processed:
        cr_processed.append(cr[0])

    rs = russian_sentiment(text, truncation=True, max_length=100)

    return {
        "ru-go-emotions": gm_processed,
        "cedr": cr_processed,
        "russian-sentiment": rs,
    }


def form_answer(text, threshold):
    emotions = get_emotions(text, threshold)
    answer = []
    emotions_set = []
    for key, value in emotions.items():
        answer.append(f"Датасет: {fix_text(key)}\n")
        for score in value:
            if key == "ru-go-emotions":
                label = ru_go_emotions_strings[score["label"]]
            elif key == "cedr":
                label = cedr_strings[score["label"]]
            elif key == "russian-sentiment":
                label = russian_sentiment_strings[score["label"]]
            new_string = fix_text(f"{label}: {score['score']*100:.2f}%")
            answer.append(f"*{new_string}*\n")
            emotions_set.append(fix_text(label))
        answer.append("\n")
    answer = "".join(answer)
    return answer, set(emotions_set)


# def generate_response(prompt, max_length=256):
#     input_ids = tokenizer.encode(prompt, return_tensors="pt")

#     input_ids = input_ids.to(model.device)

#     output = model.generate(
#         input_ids, max_length=max_length, num_return_sequences=1, temperature=1000
#     )

#     full_response = tokenizer.decode(output[0], skip_special_tokens=True)

#     bot_index = full_response.find("<bot>")

#     if bot_index != -1:
#         response = full_response[bot_index + 5 :].strip()
#     else:
#         response = full_response.strip()

#     return response


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print(update.message.text)
    answer, emotions = form_answer(update.message.text, threshold)
    print(emotions)
    await update.message.reply_text("Найденные эмоции: " + f"*{', '.join(emotions)}*", parse_mode="MARKDOWNV2")
    # await update.message.reply_text(answer, parse_mode="MARKDOWNV2")
    # await update.message.reply_text(generate_response(update.message.text))


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Напишите сообщение")


def main() -> None:
    application = Application.builder().token(os.environ["TG_TOKEN"]).build()

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    application.add_handler(CommandHandler("start", start))

    application.run_polling()


if __name__ == "__main__":
    main()
