# Введение

Работа разделена на 2 тетрадки, в первой происходит визуализация, а во второй ML

## **Основные цели, которые лежат у истоков этого проекта:** 
 
1. Работа с Dockerom на локальном хосте; 
2. Сделать прогноз количества заказов на следующий час

## Промежуточные задачи:

1. Изучить информацию о происхождении данных и ознакомится непосредственно с самим датафреймом;
2. Провести первичную обработку;
3. Провести различные группировку и проанализировать полученные данные;
4. Исследовать графики;
5. Обучить различные модели;
6. Сделать предсказание 1 часа.

**Данные таблиц `Taxi Trips - 2022-2023`**

Набор данных о поездках на такси из города Чикаго за 2022-2023 год. Данные собраны с различных агрегаторов с целью проведения совокупного анализа, все пользователи обезличенны. Время округляется до ближайших 15 минут.

Описание содержания столбцов таблицы:
* `Trip ID` - уникальный идентификатор поездки;
* `Taxi ID` - уникальный идентификатор такси;
* `Trip Start Timestamp` - время начала поездки округляется до ближайших 15 минут;
* `Trip End Timestamp` - время завершения поездки, округляется до ближайших 15 минут;
* `Trip Seconds` - время поездки в секундах;
* `Trip Mile` - расстояние поездки в милях;
* `Pickup Census Trac` - начало поездки;
* `Dropoff Census Trac` - завершение поездки;
* `Pickup Community Are` - общественная зона начала поездки;
* `Dropoff Community Are` - общественная зона завершения поездки;
* `Far` - стоимость проезда;
* `Tip` - чаевые; 
* `Toll` - налог;
* `Extra` - дополнительные расходы на поездку;
* `Trip Tota` - общая стоимость поездки;
* `Payment Typ` - способ оплаты;
* `Compan` - компания такси;
* `Pickup Centroid Latitud` - широта (за пределами Чикаго запись отсутствует);
* `Pickup Centroid Longitud` - долгота (за пределами Чикаго запись отсутствует);
* `Pickup Centroid Locatio` - местоположение общественной территории; 
* `Dropoff Centroid Latitud` - широта местоположение общественной территории;
* `Dropoff Centroid Longitud` - долгота местоположение общественной территории;
* `Dropoff Centroid Locatio` - высодка местоположени;

Данные представлены в двух таблицах, за один год и половину года. Таблицы друг с другом соединятся с сохранением временной последовательности. 

В первую очередь стоит ознакомится с тетрадкой *1_visual_analysis_1*, а потом уже с *2_ml_analysis*.


## Используемые библиотеки
*Импорт стандартных библиотек*
import os
import numpy as np
import pandas as pd

*Импорт библиотек Spark*
import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import *
from pyspark.ml.linalg import VectorUDT
import pyspark.sql.functions as F
from pyspark.sql.functions import udf, col
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator

*Импорт библиотек для визуализации*
import seaborn as sns
import matplotlib.pyplot as plt

*Вспомогательные настройки для визуализации*
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_colwidth', 480)
sns.set(context='notebook', style='whitegrid', rc={'figure.figsize': (18,4)})
rcParams['figure.figsize'] = 18,4
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

*Импорт функций и оконных операций Spark SQL*
from pyspark.sql.functions import (to_timestamp, date_trunc, month, 
                                   year, dayofweek, avg, hour, 
                                   dayofmonth, lag, expr,
                                   sin, cos, lit)
from pyspark.sql.window import Window

*Установка случайного семени (seed) для воспроизводимости результатов*
rnd_seed = 23
np.random.seed = rnd_seed
np.random.set_state = rnd_seed