# ОРО 

## Лабораторна робота  № 1
**Дано**:
1) черно-белое изображение: размер 600×600, формат .jpg;
2) изображение выбирается по номеру варианта Лб (папка: ОРО_Лб_1_jpg);
3) фон изображения – белый;
4) цвет фигур – черный;
5) на изображении расположены объекты (геометрические фигуры) в произвольном порядке.

**Найти**:
1) определить количество объектов, расположенных на изображении;
2) выделить габаритные размеры и определить центры объектов;
3) для каждого объекта отобразить т.н. окно "захвата" объекта и центр объекта;
4) в таблицу вывести значения площадей S и центров (x, y) объектов;
5) определить следующие параметры:
   - минимальное значение площади S объекта O(min) и его центр,
   - максимальное значение площади S объекта O(max) и его центр.

**Результат:**


## Лабораторна робота  № 2
**Дано**:
1) данные Лб № 1.
  Найти:
1) выполнить растяжение минимального объекта O(min) по критерию S:
2) выполнить сжатие максимального объекта O(max) по критерию S:
    - новое значение площади преобразованного объекта,
    - k – коэффициент, значение которого округляется в меньшую сторону до целого числа;
3) центры преобразованных объектов не изменяются.

**Результат:**



## Лабораторная робота № 3
**Дано**:
1) данные Лб № 1-2.

  **Найти**:
1) выполнить перемещение нового (преобразованного) O(min) по критерию: центр O(min) равен центру нового O(max);
2) выполнить перемещение нового O(max) по критерию: центр O(max) равен центру нового O(min);
3) выполнить вращение нового O(max) по критерию: новый O(max) вращается относительно центра нового O(min):
    + на угол 90 градусов по часовой стрелке,
    + на угол 90 градусов против часовой стрелки,
- при этом объекты могут накладываться друг на друга.

**Результат:**



## Лабораторная робота № 4
**Дано**:
1) цветное изображение клетки крови, формат .jpg;
2) изображение выбирается по номеру варианта Лб (папка: ОРО_Лб_4_jpg).

**Найти**:
1) выполнить преобразование изображения в импульсный вид.

**Результат:**

![lab4](https://github.com/Stardusted1/ORO/blob/master/Lab_4_impulse.png)

## Лабораторная робота № 5
**Дано**:
1) данные Лб № 4.

**Найти**:
1) как можно точнее (по возможности) выделить габаритные размеры и определить центр объекта;
2) для объекта отобразить окно "захвата" объекта и центр объекта.

**Результат:**

![lab5](https://github.com/Stardusted1/ORO/blob/master/Lab_5_impulse.png)

