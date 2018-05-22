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
    x, y = ceil((len(images_dict)) / 2), floor((len(images_dict)) / 2) # определяем размерность сетки
    num = 0 # номер изображения
    for key, value in images_dict.items(): # цикл для прогона по всем изображением
        plt.subplot(x, y, num+1) # обозначаем место для изображения
        plt.imshow(value, 'gray') # показываем изображение
        plt.title(key) # пишем название
        num += 1 # увеличиваем номер изображения
    plt.show() # показываем созданное окно

# Функция для определения объектов на изображение(найденные помещаем в прямоугольник)
def draw_rectangle(image, contours):
    print("Contours length is", len(contours)) # вывод количества найденных контуров
    i = 0 # переменная счетчик
    areas = {} # словарь для координат прямоугольника
    for cont in contours: # цикл для каждой точки найденных контуров
        i += 1 # инкрементируем счетчик
        areas[i] = cv2.contourArea(cont) # вычисляет область контура
    areas_sorted = sorted(areas.items(), key=lambda x: x[1], reverse=True) # сортируем координаты
    for i in range(0,len(areas)): # цикл для всех найденых координат
        cont_num = areas_sorted[i][0] - 1 # номер контура
        if cont_num != 999: # если контур не равен 999
            x,y,w,h = cv2.boundingRect(contours[cont_num]) # вычисляет вершины прямоугольника
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1) # рисуем прямоугольник на изображении
    return image

# Функция для определения контуров
def find_contours(image):
    imcont, contours, hierarchy = cv2.findContours(image, 1, 2) # находим контуры
    return contours # возвращаем контуры

# Функция для загрузки изображения
def load_image(image_path):
    return cv2.imread(image_path) # загружаем и возвращаем изображение

# Функция для предварительной обработки
def preprocess(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # обесцвечиваем
    den = cv2.fastNlMeansDenoising(img_gray, None, 21, 5, 21) # фильтр для удаления шума
    thresh = cv2.threshold(den,127,255,0)[1] # бинарное изображение
    return den, thresh

# Функция для распознавания символов
def recognize(image, method):
    if method == 'ocr': # если метов ocr
        # то пытаемся определить тесерактом
        return pytesseract.image_to_string(image, config='nobatch digits').replace(' ', '')
    else: # иначе
        return 'NOT IMPLEMENTED YET' # не реализовано

# основная часть
img = load_image(args.image) # загружаем изображение
den, thresh = preprocess(img) # проводим предварительную обработку
contours = find_contours(thresh) # находим контуры

cv2.imwrite('temp.png',thresh) # сохраняем промежуточное изображение
imcont = cv2.imread('temp.png') # считываем для дальнейшей обработки

imcont = draw_rectangle(imcont, contours); # отрисовываем найденные контуры

# выводим результата обработки
print("Possible solution with {} method is {}".format(args.method, recognize(thresh, args.method)))

# показываем полученный результат
present_data({'orig': img, 'fastNlMeansDenoising': den, 'thresh': thresh, 'rectangle': imcont})
