import os
import random
from datetime import datetime
from flask import Flask, render_template, request, url_for
from flask_wtf import FlaskForm
from wtforms import IntegerField, SubmitField, BooleanField, SelectField
from wtforms.validators import DataRequired, NumberRange, Optional
from werkzeug.utils import secure_filename
from processor import process_image, plot_histograms_comparison, plot_noise_analysis

# ==========================================
# Инициализация приложения
# ==========================================
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'super-secret-key-2026')

# Настройка загрузки
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# ==========================================
# Создание капчи
# ==========================================
def generate_captcha():
    """Простая статическая капча для защиты от ботов"""
    question = "Сколько будет 2 + 2?"
    answer = "4"
    return question, answer

# ==========================================
# Форма (с новым полем выбора фильтра)
# ==========================================
class ImageProcessForm(FlaskForm):
    """Форма для обработки изображений"""
    intensity = IntegerField(
        'Интенсивность сглаживания (1-10)',
        validators=[
            DataRequired(message="Введите значение интенсивности"),
            NumberRange(min=1, max=10, message="Значение должно быть от 1 до 10")
        ],
        default=5
    )
    filter_type = SelectField(
        'Тип фильтра',
        choices=[
            ('gaussian', 'Фильтр Гаусса (универсальный)'),
            ('bilateral', 'Билатеральный фильтр (сохраняет края)')
        ],
        default='gaussian'
    )
    add_watermark = BooleanField('Добавить временную метку (дата/время)', default=False)
    submit = SubmitField('Запустить обработку')

# ==========================================
# Маршруты
# ==========================================
@app.route('/', methods=['GET', 'POST'])
def index():
    """Главная страница с обработкой"""
    form = ImageProcessForm()
    
    # Переменные для результатов
    original_img = None
    processed_img = None
    hist_comparison = None
    noise_plot = None
    error_message = None
    
    # Генерируем капчу
    captcha_q, captcha_a = generate_captcha()
    
    if form.validate_on_submit():
        # Проверка капчи
        user_captcha = request.form.get('captcha_answer', '').strip()
        
        if user_captcha != captcha_a:
            error_message = "❌ Неверный ответ на капчу! Попробуйте ещё раз."
        else:
            # Получаем файл
            file = request.files.get('image')
            
            if file and file.filename:
                # Сохраняем оригинал
                filename = secure_filename(file.filename)
                original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(original_path)
                original_img = f'uploads/{filename}'
                
                # Готовим путь для результата
                proc_filename = f'processed_{filename}'
                proc_path = os.path.join(app.config['UPLOAD_FOLDER'], proc_filename)
                
                # Получаем параметры
                intensity = form.intensity.data
                filter_type = form.filter_type.data
                add_watermark = form.add_watermark.data
                
                try:
                    # === ОСНОВНАЯ ОБРАБОТКА ===
                    # Преобразуем интенсивность (1-10) в параметры фильтра
                    # Для Гаусса: ядро = 3 + (intensity * 2), но нечётное
                    kernel_size = 3 + (intensity // 2) * 2
                    if kernel_size > 21:
                        kernel_size = 21
                    sigma = intensity * 0.8
                    
                    # Вызываем новый процессор
                    process_image(
                        input_path=original_path,
                        output_path=proc_path,
                        kernel_size=kernel_size,
                        sigma=sigma,
                        filter_type=filter_type,
                        add_timestamp=add_watermark
                    )
                    
                    processed_img = f'uploads/{proc_filename}'
                    
                    # === ГИСТОГРАММЫ (сравнительные) ===
                    hist_filename = f'hist_comp_{filename}.png'
                    hist_path = os.path.join(app.config['UPLOAD_FOLDER'], hist_filename)
                    plot_histograms_comparison(original_path, proc_path, hist_path)
                    hist_comparison = f'uploads/{hist_filename}'
                    
                    # === ГРАФИК ШУМА ===
                    noise_filename = f'noise_analysis_{filename}.png'
                    noise_path = os.path.join(app.config['UPLOAD_FOLDER'], noise_filename)
                    plot_noise_analysis(original_path, proc_path, noise_path)
                    noise_plot = f'uploads/{noise_filename}'
                    
                except Exception as e:
                    error_message = f"⚠️ Ошибка обработки: {str(e)}"
                    print(f"[ERROR] {e}")
            else:
                error_message = "⚠️ Пожалуйста, выберите изображение!"
    
return render_template(
    'index.html',
    form=form,
    original=original_img,
    processed=processed_img,
    hist_comparison=hist_comparison,
    noise_plot=noise_plot,
    error=error_message,
    captcha_question=captcha_q
)

# ==========================================
# Запуск
# ==========================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
