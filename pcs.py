'''
argparse - Обработка входных параметров
cv2 - Обертка OpenCV для Python
pytesseract - Обертка для Tesseract(OCR)
pyplot из matplotlib - Отрисовки графики
floor и ceil из math - Округление расчётов
'''
import argparse, cv2, pytesseract, numpy
from matplotlib import pyplot as plt
from math import floor, ceil

# инициализация парсера аргументов, задание описания программы
parser = argparse.ArgumentParser(description='Captcha symbol recognition.')
# добавление аргумента изображения
parser.add_argument('image', metavar='IMAGE', type=str, nargs='?', help='an image for the recognition')
# добавление аргумента вывода списка доступных методов распознавания
parser.add_argument('--method-list', '-ml', action='store_true', dest='method_list',
                   help='show recognition method list and exit')
# добавление аргумента метода распознавания изображения
parser.add_argument('--method', '-m', dest='method', default='ocr',
                   help='recognition method (default: ocr)')
# добавление аргумента отключения вывода промежуточного результата
parser.add_argument('--disable-intermediate-output', '-dio', action='store_true',
                   dest='no_intermediate', help='disable intermediate result output')
# добавление аргумента отключения распознавания
parser.add_argument('--disable-recognition', '-dr', action='store_true',
                   dest='no_recognize', help='disable image recognition')
# добавление аргумента расширенного вывода
parser.add_argument('--verbose', '-v', action='store_true',
                   dest='verbose', help='produce verbose  output')

# считывание входных параметров
args = parser.parse_args()

# словарь доступных методов распознавания
available_methods = {
    'ocr': 'recognition with Tesseract OCR',
    'nn': 'recognition with TensorFlow'
}

# функция для вывода строк только в режиме расширенного вывода
def vprint(*__args):
    # если программа запущена с аргументом --verbose
    if (args.verbose):
        # инициализируем пустую строку для вывода
        string = ''
        # цикл по каждому аргументу функции
        for arg in __args:
            # составляем строку для вывода
            string += '{} '.format(str(arg))
        # выводим полученную строку
        print(string)

# функция для вывода доступных методов распознавания
def print_method_list():
    # вывод строки 'доступные методы'
    print('available methods:')
    # цикл по всем значаениям словаря доступных методов
    for method, description in available_methods.items():
        # вывод форматированной строки
        print('  {}\t\t\t{}'.format(method, description))

# функция для представления данных при помощи pyplot
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

# функция для определения объектов на изображение(найденные помещаем в прямоугольник)
def draw_rectangle(image, contours):
    # вывод количества найденных контуров
    vprint("Contours length is", len(contours))
    # переменная счетчик
    i = 0
    # словарь для координат прямоугольника
    areas = {}
    actual_areas = {}
    # vprint(contours[9][0])
    # цикл для каждой точки найденных контуров
    for cont in contours:
        # вычисляем область контура
        areas[i] = cv2.contourArea(cont)
        x,y,w,h = cv2.boundingRect(cont)
        actual_area = w*h
        if actual_area > 10:
            actual_areas[i] = actual_area
        # инкрементируем счетчик
        i += 1
    # сортируем координаты
    areas_sorted = sorted(areas.items(), key=lambda x: x[1], reverse=True)
    actual_areas_sorted = sorted(actual_areas.items(), key=lambda x: x[1], reverse=True)
    # цикл для всех найденых координат
    for i in range(0,len(actual_areas)):
    # for i in range(0,4):
        # номер контура
        cont_num = actual_areas_sorted[i][0]
        vprint('cont_num', cont_num)
        # если контур не равен 999
        # if cont_num != 999:
        # вычисляет вершины прямоугольника
        x,y,w,h = cv2.boundingRect(contours[cont_num])
        vprint('area', w*h)
        # рисуем прямоугольник на изображении
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
    # возвращаем изображение с нарисованными контурами
    return image

# функция для определения контуров
def find_contours(image):
    # находим контуры
    # imcont, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    imcont, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vprint(hierarchy)
    # возвращаем контуры
    return contours

# функция для загрузки изображения
def load_image(image_path):
    # загружаем и возвращаем изображение
    return cv2.imread(image_path)

# функция для предварительной обработки
def preprocess(image):
    def dominant_color(image):
        colors = {0:0, 255:0}
        for line in image:
            for pixel in line:
                colors[pixel] += 1
        return 0 if colors[0] > colors[255] else 255
    # обесцвечиваем
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # фильтр для удаления шума
    # den = cv2.fastNlMeansDenoising(gray, None, 21, 5, 21)
    den = cv2.fastNlMeansDenoising(gray, None, 3, 9, 21)
    # бинарное изображение
    # thresh = cv2.threshold(den,127,255,0)[1]
    thresh = cv2.threshold(den,150,255,0)[1]
    if (dominant_color(thresh) == 255):
        thresh = cv2.threshold(den,150,255,1)[1]
    print(thresh[-1])
    # den = cv2.fastNlMeansDenoising(thresh, None, 21, 5, 21)

    # возвращаем предварительно обрадотанные изображения
    return gray, den, thresh

# функция для распознавания символов
def recognize(image, method):
    # если метод не из списка доступных
    if (not method in available_methods):
        # показываем сообщение с ошибкой о пустом аргументе IMAGE
        print(
            '{}: error: the value "{}" of following argument are unknown: --method\n'.format(__file__, method) +
            '{}: see available methods with --method-list argument'.format(__file__)
        )
        # выходим
        exit(1)

    # вызываем правильную функцию для распознавания
    return globals()['__recognize_{}'.format(method)](image)

# функция для распознавания символов с ocr
def __recognize_ocr(image):
    #return pytesseract.image_to_string(image, config='nobatch digits').replace(' ', '')
    return pytesseract.image_to_string(image).replace(' ', '')

# функция для распознавания символов с нейросетью
def __recognize_nn(image):
    return 'NOT IMPLEMENTED YET'

# если программа запущена с аргументом --method_list
if (args.method_list):
    # выводим доступные методы
    print_method_list()
    # выходим из программы
    exit(0)

# если программа запущена без аргумента IMAGE
if (not args.image):
    # показываем краткую справку по программе
    parser.print_usage()
    # показываем ошибку о пустом аргументе IMAGE
    print('{}: error: the following arguments are required: IMAGE'.format(__file__))
    # выходим
    exit(1)

# загружаем изображение
img = load_image(args.image)
# print('channels', img.channels())
b = img.copy()
# set green and red channels to 0
b[:, :, 1] = 0
b[:, :, 2] = 0


g = img.copy()
# set blue and red channels to 0
g[:, :, 0] = 0
g[:, :, 2] = 0

r = img.copy()
# set blue and green channels to 0
r[:, :, 0] = 0
r[:, :, 1] = 0
imgg = img.copy()
# kernel = numpy.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# kernel = numpy.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
kernel = numpy.array([[0,1,0], [1,-4,1], [0,1,0]])
kernel = numpy.ones((5,5),numpy.float32)/25
imgg = cv2.filter2D(imgg,-1,kernel)
# r = cv2.Laplacian(img.copy(),cv2.CV_64F)
# g = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
# b = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

# b = cv2.CreateImage(cv2.GetSize(image), image.depth, 1)
# g = cv2.CreateImage(cv2.GetSize(image), image.depth, 1)
# r = cv2.CreateImage(cv2.GetSize(image), image.depth, 1)
# cv2.Split(image, b, g, r, None)
# imgg = cv2.filter2D(imgg, -1, kernel)
# проводим предварительную обработку
gray, den, thresh = preprocess(img)

# если программа запущена без аргумента --disable-recognition
if (not args.no_recognize):
    # выводим результата обработки
    recognized_text = recognize(thresh, args.method)
    delimiter = recognized_text[2]
    print(delimiter)
    print("Possible solution with {} method is {}".format(args.method, recognized_text))
    if delimiter == ':':
        if len(recognized_text) == 5:
            h = recognized_text[:2]
            m = recognized_text[3:]
            print(h,m)
            print("Maybe time is {}:{} and it is {} minutes or seconds".format(h,m,int(h)*60+int(m)))
        if len(recognized_text) == 8:
            h = recognized_text[:2]
            m = recognized_text[3:5]
            s = recognized_text[6:]
            print(h,m,s)
            print("Maybe time is {}:{}:{} and it is {} seconds".format(h,m,s,int(h)*60*60+int(m)*60+int(s)))
    elif delimiter == '/':
        d = recognized_text[:2]
        m = recognized_text[3:5]
        y = recognized_text[6:]
        print("Maybe it is {} day, {} month and {} year".format(d,m,y))
    else:
        print("The number is {}".format(recognized_text))

# если программа запущена без аргумента --disable-intermediate-output
if (not args.no_intermediate):
    # находим контуры
    contours = find_contours(thresh)

    # сохраняем промежуточное изображение
    cv2.imwrite('temp.png',thresh)
    # считываем для дальнейшей обработки
    imcont = cv2.imread('temp.png')

    # отрисовываем найденные контуры
    imcont = draw_rectangle(imcont, contours)
    
    # показываем полученный результат
    present_data({'orig': img, 'gray':gray, 'imgg': imgg, 'r':r, 'g':g, 'b':b, 'fastNlMeansDenoising': den, 'thresh': thresh, 'rectangle': imcont})
