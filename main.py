import sys
import re
import torch
from flask import Flask, request, render_template_string
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

app = Flask(__name__)

# 📝 Встроенный HTML-шаблон
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-Ассистент</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container">
        <header>
            <h1>🎬 AI-Ассистент</h1>
            <p>Анализ настроения, рекомендации фильмов и мгновенный перевод</p>
        </header>

        <section class="card">
            <h2>🎭 Анализ настроения</h2>
            <form method="POST" action="/">
                <textarea name="message" rows="3" placeholder="Опиши своё состояние...">{{ user_text }}</textarea>
                <button type="submit" class="btn btn-primary">Получить рекомендацию</button>
            </form>
            {% if recommendation %}
                <div class="result-box success">{{ recommendation|safe }}</div>
            {% endif %}
        </section>

        <section class="card">
            <h2>🌍 Перевод на английский</h2>
            <form method="POST" action="/translate">
                <textarea name="text_to_translate" rows="3" placeholder="Введите текст на русском..."></textarea>
                <button type="submit" class="btn btn-secondary">Перевести</button>
            </form>
            {% if translation_result %}
                <div class="result-box info">
                    <b>Оригинал:</b> {{ original_text }}<br>
                    <b>Перевод:</b> {{ translation_result }}
                </div>
            {% endif %}
        </section>

        <section class="card">
            <h2>📨 Тестовый маршрут</h2>
            <form method="POST" action="/submit">
                <textarea name="message" rows="2" placeholder="Текст для /submit..."></textarea>
                <button type="submit" class="btn btn-tertiary">Отправить</button>
            </form>
            {% if submit_reply %}
                <div class="result-box warning"><b>Ответ:</b> {{ submit_reply }}</div>
            {% endif %}
        </section>
    </div>
</body>
</html>
"""

print(f"🐍 Python: {sys.version.split()[0]}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Устройство: {device}")

# 🔹 1. Анализ настроения (pipeline стабилен для этой задачи)
print("⏳ Загрузка модели анализа настроения...")
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis", 
        model="blanchefort/rubert-base-cased-sentiment",
        device=0 if torch.cuda.is_available() else -1
    )
    print("✅ Sentiment готов")
except Exception as e:
    print(f"⚠️ Sentiment не загружен: {e}")
    sentiment_analyzer = None

# 🔹 2. ПЕРЕВОД (прямая загрузка модели → ИСКЛЮЧАЕТ ОШИБКУ check_task)
print("⏳ Загрузка модели перевода (ru -> en)...")
try:
    trans_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    trans_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en").to(device)
    trans_model.eval()
    print("✅ Translator готов")
except Exception as e:
    print(f"⚠️ Translator не загружен: {e}")
    trans_tokenizer, trans_model = None, None

# 🔹 3. Генерация рекомендаций (RuGPT-3)
print("⏳ Загрузка модели генерации...")
try:
    gpt_tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")
    gpt_model = AutoModelForCausalLM.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2").to(device)
    gpt_model.eval()
    if gpt_tokenizer.pad_token is None:
        gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
    print("✅ GPT-модель готова")
except Exception as e:
    print(f"⚠️ GPT не загружена: {e}")
    gpt_tokenizer, gpt_model = None, None

# 🛠 Вспомогательные функции
def translate_text(text):
    if not trans_model or not trans_tokenizer:
        return "Модель перевода недоступна"
    inputs = trans_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = trans_model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
    return trans_tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_film_title(generated_text):
    if not generated_text: return ""
    m = re.search(r'«([^»]{2,100})»', generated_text)
    if m: return m.group(1).strip()
    clean_text = generated_text.strip().splitlines()[0]
    clean_text = re.sub(r'^(Фильм|Рекомендую|Советую):\s*', '', clean_text, flags=re.IGNORECASE)
    return clean_text[:80].strip()

def generate_recommendation(mood):
    if not gpt_model or not gpt_tokenizer:
        return "Модель генерации недоступна"
    prompt = f"Посоветуй один популярный фильм для человека, у которого {mood} настроение. Назови только название фильма в кавычках и кратко объясни почему."
    inputs = gpt_tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = gpt_model.generate(
            **inputs, max_length=100, do_sample=True, top_p=0.9,
            temperature=0.7, pad_token_id=gpt_tokenizer.eos_token_id
        )

    full_text = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_text.replace(prompt, "", 1).strip()

# 🌐 Маршруты
@app.route("/", methods=["GET", "POST"])
def index():
    recommendation, user_text = "", ""
    if request.method == "POST":
        user_text = request.form.get("message", "")
        if not user_text.strip():
            recommendation = "Пожалуйста, введите текст."
        elif not sentiment_analyzer:
            recommendation = "Модель анализа настроения не загружена."
        else:
            try:
                result = sentiment_analyzer(user_text)[0]
                label = result["label"]
                mood = {"POSITIVE": "хорошее", "NEGATIVE": "плохое"}.get(label, "нейтральное")
                
                raw_ai_text = generate_recommendation(mood)
                film_title = extract_film_title(raw_ai_text)
                recommendation = f"<b>Настроение:</b> {mood}.<br><b>Рекомендация:</b> {film_title if film_title else raw_ai_text[:150]}"
            except Exception as e:
                print(f"Error: {e}")
                recommendation = "Ошибка при обработке запроса."

    return render_template_string(HTML_TEMPLATE, recommendation=recommendation, user_text=user_text, translation_result="", original_text="", submit_reply="")

@app.route("/translate", methods=["POST"])
def translate_route():
    text = request.form.get("text_to_translate", "")
    original_text = text
    translation_result = ""
    
    if text.strip():
        try:
            translation_result = translate_text(text)
        except Exception as e:
            translation_result = f"Ошибка перевода: {str(e)[:100]}"
            print(f"Translation Error: {e}")
            
    return render_template_string(HTML_TEMPLATE, recommendation="", user_text="", translation_result=translation_result, original_text=original_text, submit_reply="")

@app.route("/submit", methods=["POST"])
def submit():
    user_message = request.form.get("message", "")
    reply = "Ты ничего не ввел" if not user_message.strip() else f"Я получил твой текст: {user_message}!"
    
    if "хорошо" in user_message.lower(): reply = "Ты молодец"
    elif "плохо" in user_message.lower(): reply = "Что случилось?"
        
    return render_template_string(HTML_TEMPLATE, recommendation="", user_text="", translation_result="", original_text="", submit_reply=reply)

if __name__ == "__main__":
    # use_reloader=False предотвращает двойную загрузку моделей в dev-режиме
    app.run(debug=False, host="127.0.0.1", port=5000, use_reloader=False)