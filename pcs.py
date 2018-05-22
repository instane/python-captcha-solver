'''
argparse - Обработка входных параметров
cv2 - Обертка OpenCV для Python
pytesseract - Обертка для Tesseract(OCR)
pyplot из matplotlib - Отрисовки графики
floor и ceil из math - Округление расчётов
'''
import argparse, cv2, pytesseract 
from matplotlib import pyplot as plt
from math import floor, ceil

# инициализация парсера аргументов, задание описания программы
parser = argparse.ArgumentParser(description='Captcha symbol recognition.')
# добавление аргумента изображения
parser.add_argument('image', metavar='I', type=str, help='an image for the recognition')
# добавление аргумента метода распознавания изображения
parser.add_argument('--method', '-m', dest='method', default='ocr',
                   help='recognition method (default: ocr)')

# считывание входных параметров
args = parser.parse_args()

# Функция для представления данных при помощи pyplot
def present_data(images_dict):
    # определяем размерность сетки
    x, y = ceil((len(images_dict)) / 2), floor((len(images_dict)) / 2)
    # задаем номер изображения
    num = 0
    # цикл для прогона по всем изображением
    for key, value in images_dict.items():
        # обозначаем место для изображения
        plt.subplot(x, y, num+1)
        # показываем изображение
        plt.imshow(value, 'gray')
        # пишем название
        plt.title(key)
        # увеличиваем номер изображения
        num += 1
    # показываем созданное окно
    plt.show()

# Функция для определения объектов на изображение(найденные помещаем в прямоугольник)
def draw_rectangle(image, contours):
    # вывод количества найденных контуров
    print("Contours length is", len(contours))
    # переменная счетчик
    i = 0
    # словарь для координат прямоугольника
    areas = {}
    # цикл для каждой точки найденных контуров
    for cont in contours:
        # инкрементируем счетчик
        i += 1
        # вычисляем область контура
        areas[i] = cv2.contourArea(cont)
    # сортируем координаты
    areas_sorted = sorted(areas.items(), key=lambda x: x[1], reverse=True)
    # цикл для всех найденых координат
    for i in range(0,len(areas)):
        # номер контура
        cont_num = areas_sorted[i][0] - 1
        # если контур не равен 999
        if cont_num != 999:
            # вычисляет вершины прямоугольника
            x,y,w,h = cv2.boundingRect(contours[cont_num])
            # рисуем прямоугольник на изображении
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
    # возвращаем изображение с нарисованными контурами
    return image

# Функция для определения контуров
def find_contours(image):
    # находим контуры
    imcont, contours, hierarchy = cv2.findContours(image, 1, 2)
    # возвращаем контуры
    return contours

# Функция для загрузки изображения
def load_image(image_path):
    # загружаем и возвращаем изображение
    return cv2.imread(image_path)

# Функция для предварительной обработки
def preprocess(image):
    # обесцвечиваем
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # фильтр для удаления шума
    den = cv2.fastNlMeansDenoising(img_gray, None, 21, 5, 21)
    # бинарное изображение
    thresh = cv2.threshold(den,127,255,0)[1]
    # возвращаем предварительно обрадотанные изображения
    return den, thresh

# Функция для распознавания символов
def recognize(image, method):
    # если метод ocr
    if method == 'ocr':
        # то пытаемся определить тесерактом
        return pytesseract.image_to_string(image, config='nobatch digits').replace(' ', '')
    # иначе
    else: 
        # возвращаем не реализовано
        return 'NOT IMPLEMENTED YET'

# загружаем изображение
img = load_image(args.image)
# проводим предварительную обработку
den, thresh = preprocess(img)
# находим контуры
contours = find_contours(thresh)

# сохраняем промежуточное изображение
cv2.imwrite('temp.png',thresh)
# считываем для дальнейшей обработки
imcont = cv2.imread('temp.png')

# отрисовываем найденные контуры
imcont = draw_rectangle(imcont, contours)

# выводим результата обработки
print("Possible solution with {} method is {}".format(args.method, recognize(thresh, args.method)))

# показываем полученный результат
present_data({'orig': img, 'fastNlMeansDenoising': den, 'thresh': thresh, 'rectangle': imcont})
