import sys
import re
import torch
import threading
from flask import Flask, request, render_template_string, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

app = Flask(__name__)
app.secret_key = "ai-assistant-secret-key"  # Для сессий/уведомлений

# 🔐 Блокировка для потокобезопасного доступа к моделям
model_lock = threading.Lock()

# 🎨 Улучшенный HTML-шаблон с современным дизайном
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>✨ AI-Ассистент</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body>
    <div class="container">
        <header class="hero">
            <div class="hero-content">
                <h1><i class="fas fa-robot"></i> AI-Ассистент</h1>
                <p>Анализ настроения • Рекомендации фильмов • Мгновенный перевод</p>
            </div>
            <div class="status-badge">
                <span class="status-dot"></span>
                Модели загружены
            </div>
        </header>

        <!-- Карточка: Анализ настроения -->
        <section class="card card-gradient">
            <div class="card-header">
                <h2><i class="fas fa-theater-masks"></i> 🎭 Анализ настроения</h2>
            </div>
            <form method="POST" action="/" class="form-group">
                <div class="input-wrapper">
                    <textarea name="message" id="mood-input" rows="3" 
                              placeholder="Как вы себя чувствуете сегодня? Опишите своё настроение..." 
                              maxlength="500">{{ user_text }}</textarea>
                    <div class="char-counter"><span id="mood-count">0</span>/500</div>
                    <button type="button" class="clear-btn" onclick="clearField('mood-input')"><i class="fas fa-times"></i></button>
                </div>
                <button type="submit" class="btn btn-primary btn-lg">
                    <i class="fas fa-magic"></i> Получить рекомендацию
                </button>
            </form>
            {% if recommendation %}
                <div class="result-box success animate-fadeIn">
                    <div class="result-icon"><i class="fas fa-check-circle"></i></div>
                    <div class="result-content">{{ recommendation|safe }}</div>
                    <button class="copy-btn" onclick="copyText(this)"><i class="fas fa-copy"></i> Копировать</button>
                </div>
            {% endif %}
        </section>

        <!-- Карточка: Перевод -->
        <section class="card card-gradient">
            <div class="card-header">
                <h2><i class="fas fa-language"></i> 🌍 Перевод RU → EN</h2>
            </div>
            <form method="POST" action="/translate" class="form-group">
                <div class="input-wrapper">
                    <textarea name="text_to_translate" id="translate-input" rows="3" 
                              placeholder="Введите текст на русском для перевода..." 
                              maxlength="1000"></textarea>
                    <div class="char-counter"><span id="translate-count">0</span>/1000</div>
                    <button type="button" class="clear-btn" onclick="clearField('translate-input')"><i class="fas fa-times"></i></button>
                </div>
                <button type="submit" class="btn btn-secondary btn-lg">
                    <i class="fas fa-exchange-alt"></i> Перевести
                </button>
            </form>
            {% if translation_result %}
                <div class="result-box info animate-fadeIn">
                    <div class="result-header">
                        <span><i class="fas fa-file-alt"></i> Оригинал:</span>
                        <button class="copy-btn small" onclick="copyText(this, '{{ original_text|escapejs }}')"><i class="fas fa-copy"></i></button>
                    </div>
                    <p class="original-text">{{ original_text }}</p>
                    <div class="result-header">
                        <span><i class="fas fa-globe-americas"></i> Перевод:</span>
                        <button class="copy-btn small" onclick="copyText(this, '{{ translation_result|escapejs }}')"><i class="fas fa-copy"></i></button>
                    </div>
                    <p class="translated-text">{{ translation_result }}</p>
                </div>
            {% endif %}
        </section>

        <!-- Карточка: Тест -->
        <section class="card card-gradient">
            <div class="card-header">
                <h2><i class="fas fa-paper-plane"></i> 📨 Тестовый маршрут</h2>
            </div>
            <form method="POST" action="/submit" class="form-group">
                <div class="input-wrapper">
                    <textarea name="message" id="submit-input" rows="2" 
                              placeholder="Введите текст для теста..." 
                              maxlength="200"></textarea>
                    <div class="char-counter"><span id="submit-count">0</span>/200</div>
                </div>
                <button type="submit" class="btn btn-tertiary">
                    <i class="fas fa-paper-plane"></i> Отправить
                </button>
            </form>
            {% if submit_reply %}
                <div class="result-box warning animate-fadeIn">
                    <i class="fas fa-comment-alt"></i> <b>Ответ:</b> {{ submit_reply }}
                </div>
            {% endif %}
        </section>

        <footer class="footer">
            <p>✨ Создано с любовью • Работает на трансформерах • {{ device_info }}</p>
        </footer>
    </div>

    <script>
        // Счётчики символов
        document.querySelectorAll('textarea').forEach(el => {
            const counter = document.getElementById(el.id.replace('input', 'count'));
            el.addEventListener('input', () => {
                counter.textContent = el.value.length;
            });
            // Инициализация
            counter.textContent = el.value.length;
        });

        // Очистка поля
        function clearField(id) {
            document.getElementById(id).value = '';
            document.getElementById(id.replace('input', 'count')).textContent = '0';
            document.getElementById(id).focus();
        }

        // Копирование текста
        function copyText(btn, text = null) {
            const content = text || btn.closest('.result-box').querySelector('.translated-text, .original-text, .result-content')?.innerText;
            if (content) {
                navigator.clipboard.writeText(content.trim()).then(() => {
                    const original = btn.innerHTML;
                    btn.innerHTML = '<i class="fas fa-check"></i> Скопировано!';
                    btn.classList.add('copied');
                    setTimeout(() => {
                        btn.innerHTML = original;
                        btn.classList.remove('copied');
                    }, 2000);
                });
            }
        }

        // Плавное появление
        document.addEventListener('DOMContentLoaded', () => {
            document.querySelectorAll('.animate-fadeIn').forEach(el => {
                el.style.opacity = '0';
                el.style.transform = 'translateY(10px)';
                setTimeout(() => {
                    el.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
                    el.style.opacity = '1';
                    el.style.transform = 'translateY(0)';
                }, 100);
            });
        });
    </script>
</body>
</html>
"""

print(f"🐍 Python: {sys.version.split()[0]}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Устройство: {device}")

# 🔹 1. Анализ настроения
print("⏳ Загрузка модели анализа настроения...")
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis", 
        model="blanchefort/rubert-base-cased-sentiment",
        device=0 if torch.cuda.is_available() else -1,
        return_all_scores=False
    )
    print("✅ Sentiment готов")
except Exception as e:
    print(f"⚠️ Sentiment не загружен: {e}")
    sentiment_analyzer = None

# 🔹 2. Перевод (ru -> en) — улучшенная загрузка
print("⏳ Загрузка модели перевода...")
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


# 🛠 Вспомогательные функции с улучшенной обработкой
def translate_text(text):
    """Перевод текста с улучшенной предобработкой"""
    if not trans_model or not trans_tokenizer:
        return "Модель перевода временно недоступна"
    
    # Предобработка: очистка и обрезка
    text = ' '.join(text.strip().split())
    if len(text) > 1000:
        text = text[:1000] + "..."
    
    with model_lock:  # Потокобезопасность
        inputs = trans_tokenizer(
            text, return_tensors="pt", padding=True, 
            truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = trans_model.generate(
                **inputs, 
                max_length=512, 
                num_beams=4, 
                early_stopping=True,
                no_repeat_ngram_size=2  # Улучшение качества
            )
    return trans_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def extract_film_title(generated_text):
    """Извлечение названия фильма с улучшенной логикой"""
    if not generated_text: 
        return ""
    
    # Поиск в кавычках «...» или "..."
    patterns = [r'«([^»]{2,80})»', r'"([^"]{2,80})"', r'\'([^\']{2,80})\'']
    for pattern in patterns:
        m = re.search(pattern, generated_text)
        if m: 
            return m.group(1).strip()
    
    # fallback: первая строка без префиксов
    clean_text = generated_text.strip().splitlines()[0]
    clean_text = re.sub(r'^(Фильм|Рекомендую|Советую|Название):\s*', '', clean_text, flags=re.IGNORECASE)
    return clean_text[:80].strip()


def generate_recommendation(mood):
    """Генерация рекомендации с оптимизированными параметрами"""
    if not gpt_model or not gpt_tokenizer:
        return "Модель генерации временно недоступна"
    
    prompt = (f"Посоветуй один популярный фильм для человека с {mood} настроением. "
              "Ответь кратко: название фильма в кавычках и одно предложение почему.")
    
    inputs = gpt_tokenizer(prompt, return_tensors="pt").to(device)
    
    with model_lock:
        with torch.no_grad():
            outputs = gpt_model.generate(
                **inputs, 
                max_length=90,  # Чуть короче для скорости
                do_sample=True, 
                top_p=0.92,
                temperature=0.75,
                pad_token_id=gpt_tokenizer.eos_token_id,
                repetition_penalty=1.15  # Уменьшает повторы
            )
    
    full_text = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_text.replace(prompt, "", 1).strip()


# 🌐 Маршруты с улучшенной обработкой
@app.route("/", methods=["GET", "POST"])
def index():
    recommendation, user_text = "", ""
    if request.method == "POST":
        user_text = request.form.get("message", "").strip()
        
        if not user_text:
            recommendation = "⚠️ Пожалуйста, введите текст для анализа."
        elif not sentiment_analyzer:
            recommendation = "⚠️ Модель анализа настроения загружается..."
        else:
            try:
                result = sentiment_analyzer(user_text[:500])[0]  # Лимит для безопасности
                label = result["label"]
                mood_map = {"POSITIVE": "хорошее", "NEGATIVE": "плохое", "NEUTRAL": "нейтральное"}
                mood = mood_map.get(label, "нейтральное")
                
                raw_ai_text = generate_recommendation(mood)
                film_title = extract_film_title(raw_ai_text)
                
                if film_title:
                    recommendation = (
                        f"<b>🎯 Настроение:</b> {mood}<br>"
                        f"<b>🎬 Рекомендация:</b> <i>«{film_title}»</i><br>"
                        f"<small>{raw_ai_text.split('»')[-1].strip()[:120] if '»' in raw_ai_text else raw_ai_text[:120]}...</small>"
                    )
                else:
                    recommendation = f"<b>🎯 Настроение:</b> {mood}<br><b>🎬 Ответ:</b> {raw_ai_text[:180]}"
                    
            except Exception as e:
                print(f"❌ Error in sentiment: {e}")
                recommendation = "⚠️ Произошла ошибка при обработке. Попробуйте ещё раз."

    return render_template_string(
        HTML_TEMPLATE, 
        recommendation=recommendation, 
        user_text=user_text, 
        translation_result="", 
        original_text="", 
        submit_reply="",
        device_info=f"GPU" if torch.cuda.is_available() else "CPU"
    )


@app.route("/translate", methods=["POST"])
def translate_route():
    text = request.form.get("text_to_translate", "").strip()
    original_text = text
    translation_result = ""
    
    if not text:
        translation_result = "⚠️ Введите текст для перевода"
    elif not trans_model:
        translation_result = "⚠️ Модель перевода загружается..."
    else:
        try:
            translation_result = translate_text(text)
            if len(translation_result) > 300:
                translation_result = translation_result[:300] + "..."
        except Exception as e:
            translation_result = f"⚠️ Ошибка: {str(e)[:80]}"
            print(f"Translation Error: {e}")
            
    return render_template_string(
        HTML_TEMPLATE, 
        recommendation="", 
        user_text="", 
        translation_result=translation_result, 
        original_text=original_text[:200] + "..." if len(original_text) > 200 else original_text, 
        submit_reply="",
        device_info=f"GPU" if torch.cuda.is_available() else "CPU"
    )


@app.route("/submit", methods=["POST"])
def submit():
    user_message = request.form.get("message", "").strip()
    
    if not user_message:
        reply = "💬 Вы не ввели текст"
    elif "хорош" in user_message.lower():
        reply = "🌟 Отлично! Рад за вас!"
    elif "плох" in user_message.lower() or "груст" in user_message.lower():
        reply = "💙 Держитесь! Всё наладится. Хотите рекомендацию фильма?"
    elif "спасиб" in user_message.lower():
        reply = "🙏 Всегда пожалуйста!"
    else:
        reply = f"✅ Получено: <i>{user_message[:100]}{'...' if len(user_message) > 100 else ''}</i>"
        
    return render_template_string(
        HTML_TEMPLATE, 
        recommendation="", 
        user_text="", 
        translation_result="", 
        original_text="", 
        submit_reply=reply,
        device_info=f"GPU" if torch.cuda.is_available() else "CPU"
    )


@app.route("/health")
def health():
    """Эндпоинт для проверки статуса"""
    return jsonify({
        "status": "ok",
        "models": {
            "sentiment": sentiment_analyzer is not None,
            "translator": trans_model is not None,
            "generator": gpt_model is not None
        },
        "device": str(device)
    })


if __name__ == "__main__":
    print("🚀 Запуск сервера на http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000, use_reloader=False, threaded=True)