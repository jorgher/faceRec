# Reconocimiento de emociones.

## Este proyecto tiene por objetivo identificar emociones utilizando características(features) geométricas.

Para lograr lo anterior se utilizará la base de datos de imágenes publicada por el Departmento de Ingeniería Eléctrica y Computación y del Centro de Ciencias del Conocimiento y el Cerebro de la Universidad de Ohio State, Columbus, OH 43210.
En ésta, se incluyen imágenes asociadas a 22 emociones, del total de 22 para el proyecto se tomarán 7. "Neutral, Happy, sad, fearful, angry, surprised y disgusted" de las que se identificarán caracteristicas geométricas con las cuales se pretende alimentar diferentes clasificadores y comprobar cuál de ellos permite el mejor reconocimiento de la emoción.

En el artículo "Compound facial expressions of emotion, Shichuan Du, Yong Tao, and Aleix M. Martinez", se explica la metodología para la generación de estas imágenes.

Una vez obtenidas la imágenes se calculan las características, se incluyen en una base de datos que a su vez se divide para la casificación en dos secciones una de entrenamiento y otra de prueba que a su vez se normaliza para evitar disperciones innecesarias.

Se comparan diferentes clasificadores "Naive Bayes", "QDA","Decision Tree", "Random Forest", "Neural Net", "AdaBoost","Nearest Neighbors", "Linear SVM", "RBF SVM", para verificar sus resultados quedando éstos de a siguiente manera.

Naive Bayes
[ 0.64423077  0.60576923  0.625       0.63461538  0.56730769  0.60576923
  0.67307692  0.66346154  0.65384615  0.65384615]
Accuracy: 0.63 (+/- 0.06)

QDA
[ 0.66346154  0.54807692  0.61538462  0.57692308  0.57692308  0.65384615
  0.625       0.61538462  0.55769231  0.53846154]
Accuracy: 0.60 (+/- 0.08)

Decision Tree
[ 0.50961538  0.50961538  0.50961538  0.54807692  0.46153846  0.55769231
  0.55769231  0.55769231  0.625       0.5       ]
Accuracy: 0.53 (+/- 0.09)

Random Forest
[ 0.63461538  0.63461538  0.54807692  0.57692308  0.56730769  0.56730769
  0.63461538  0.68269231  0.625       0.59615385]
Accuracy: 0.61 (+/- 0.08)

Neural Net
[ 0.75        0.66346154  0.67307692  0.66346154  0.63461538  0.66346154
  0.71153846  0.67307692  0.67307692  0.67307692]
Accuracy: 0.68 (+/- 0.06)

AdaBoost
[ 0.41346154  0.59615385  0.55769231  0.53846154  0.48076923  0.56730769
  0.55769231  0.55769231  0.5         0.39423077]
Accuracy: 0.52 (+/- 0.13)

Nearest Neighbors
[ 0.60576923  0.53846154  0.53846154  0.64423077  0.55769231  0.59615385
  0.59615385  0.69230769  0.66346154  0.63461538]
Accuracy: 0.61 (+/- 0.10)

Linear SVM
[ 0.67307692  0.66346154  0.65384615  0.625       0.61538462  0.65384615
  0.70192308  0.68269231  0.625       0.64423077]
Accuracy: 0.65 (+/- 0.05)

RBF SVM
[ 0.21153846  0.16346154  0.20192308  0.26923077  0.25        0.20192308
  0.25        0.18269231  0.18269231  0.19230769]
Accuracy: 0.21 (+/- 0.07)

Al observar los valores se ve que Neural Nets es el mejor clasificador pero todavía con una precisión que es posible mejorar.

En trabajo posterior se evaluarán métodos que nos conduzcan a la mejora sustancial de la precisión del algorítmo de clasificación.

#Desarrollo con Redes Convolucionales.

