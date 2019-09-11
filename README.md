### Скрипт выполняющий детектирование и распознование автомобильного номера
В качестве детектора обучена использована модель ssd_inception_v2 из tensorflow object detection model zoo.  
Данные для обучения датасет с номерами РФ. Обучающая выборка - около 3000 изображений.  
Также использованы вспомогательные функции входящие в состав tensorfow Object detection API (визуализация детектирования и тд).  
В качестве OCR - tesseract


### На вход принимает: 
 * m - режим работы: 0 - режим детектирования на изображении; 1 - режим видео-детектирования.
 * i - путь к изображению для детектирования, для режима детектирования на изображении.

### На выходе: 
В случае режима детектирования на изображении - изображение с bbox'ом вокруг детектированного номера,  
 список со строками распознанных номеров.  
В случае детектирования на видео - окно с детектированием в реальном времени, вывод списка строк с распознанными номерами.

## Требования и настройка окружения для запуска скрипта:  
Скрипт для своей работы использует несколько вспомогательных инструментов, а именно:  
* **tensorflow object detection API** - для работы с нейросетевой моделью SSD_inception_v2 переобученную для обнаружения номеров автомобилей;  
* **tesseract** - в качестве open source OCR движка;  
* а также сопутсвующие и вспомогательные Python библиотеки и фреймворки;  

**Для запуска скрипта необходимо выполнить следующие действия:**  
1. Выполнить установку **tensorflow object detection API**  
    * клонировать [репозиторий tf models](https://github.com/tensorflow/models "Tensorflow Models") в любую удобную для себя директорию;  
    * добавить в переменную PYTHONPATH (или PATH на выбор) следующие пути:  
        [ директория tensorflow models ]/research;  
        [ директория tensorflow models ]/research/slim;  
        [ директория tensorflow models ]/research/nets;  
        [ директория tensorflow models ]/research/object_detection;  
        [ директория tensorflow models ]/research/object_detection/utils;  
2. Выполнить установку tesseract [tesseract для Windows](https://github.com/UB-Mannheim/tesseract/wiki), [tesseract для Linux](https://github.com/tesseract-ocr/tesseract/releases), либо использую команду: 
```
apt-get install tesseract-ocr-rus
```
3. Установить пакеты для Python:  
    * pytesseract - для взаимодействия между python и tesseract;
    ```
    pip install pytesseract
    ```
    * tensorflow - для использования нейросетевой модели и детектирования авто номеров; 
    ```
    pip install tensorflow
    ```
    * pycocotools - необходим для *tensorflow models* 
        * [для Windows](https://github.com/philferriere/cocoapi) использовать команду:
         на Windows, вы должны иметь Visual C++ 2015 build tools в переменной PATH. Если у нет, установите [по сслылке](https://go.microsoft.com/fwlink/?LinkId=691126):


        ```
        pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
        ```
        * для Linux скачать [репозиторий](https://github.com/cocodataset/cocoapi) и сделать  "make" under coco/PythonAPI;  

    * Cython, Numpy, Opencv, Matplotlib
    ```
    pip install cython numpy matplotlib opencv-python
    ```

    ## **Готово! Можно запускать!**


### Пример использования:
В командной строке либо терминале ввести:  
Для изображений:
```
python car_number_detector_ssd.py -m 0 -i image_to_detection.jpg
```
Для видео: 
```
python car_number_detector_ssd.py -m 1 
```

### Возможные варианты улучшения:
* улучшение OCR функции - на данный момент используется tesseract;
* улучшение предобработки изображения перед OCR;
* выравнивание номера перед подачей на OCR, удаление искажений.

### Возможные иные пути решения задачи:
* Использовать детектор граней из opencv для нахождения прямоугольной рамки номеров (распространенное решение в сети), но если на номерахблики или грязь то работать не будет.
* Использовать каскады Хаара для детектирования номеров - каскады шумят (много ложных детекций).
* Использование кастомного OCR решения.

