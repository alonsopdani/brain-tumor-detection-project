#**Detección de tumores cerebrales en imágenes de resonancias magnéticas:**

El proyecto se trata de un programa que, a partir de una imagen de resonancia magnética de cerebro, da un diagnóstico cuyo resultado puede ser que existe o no un tumor en ese cerebro.

##**Objetivos**

El resultado al pasarle una imágen es un porcentaje de que exista tumor en ese cerebro, por tanto, los médicos podrían ahorrarse el tiempo invertido en ver todas las resonancias magnéticas e irse directamente a las que tienen un cierto porcentaje mínimo de tener un tumor, por ejemplo, > 10%.

Ayuda de la máquina a detectar tumores. Si se perfecciona este programa hasta el punto de que el porcentaje de acierto es muy alto, la tarea de detectar tumores puede pasar del médico a la máquina, ya que esta podría tener más memoria y capacidad para hacerlo que una persona humana. Este sería un caso claro de la cooperación entre humanos e inteligencia artificial.
    
##**El dataset**

El dataset es un conjunto de MRIs, una carpeta con tumores y otra sin tumores.

##**Procedimiento técnico**

El procedimiento técnico es pasarle a un modelo de redes neuronales el 80% de las imágenes con su diagnóstico (tras haberlas tratado) para que entrene con ellas y luego sea capaz de reconocer si existe o no un tumor en una imagen que nunca ha visto.

El 20% restante de las imágenes nos va a servir para valorar la calidad del modelo. Se las vamos a pasar, el modelo nos dará un resultado y compararemos este con el diagnóstico real de esas imágenes.

##**Pasos**

###Feature engineering (tratamiento de imágenes):

Las imágenes son datos, ya que cada píxel contiene información acerca de su color. Por ello, esta información se puede modificar para que la imagen cambie. Esto es lo que pasa, por ejemplo, cuando aplicáis filtros a una foto que vais a subir a Instagram.

En nuestro caso, a partir de varias funciones, se han tratado las imágenes para que sean un input más legible para el modelo y este sea capaz de predecir con mayor precisión.

Imágenes originales
Hacerlas cuadradas
Redimensionarlas
Pasarlas a escala de grises
Aplicarles un filtro de mediana
Pasarlas a blanco y negro










Después de esto, se convirtió el conjunto de imágenes en un dataframe de Pandas, haciendo que cada fila del mismo estuviera compuesta por los valores de cada pixel de la imagen. Dividí el dataset en dos partes: una para entrenar el modelo (80%) y otra para el test, es decir, para comprobar su calidad (20%).
        
###Algoritmo de Machine Learning

Es una red neuronal que recorre los datos 10 veces para aprender de ellos. Lo evaluamos a partir del score (80%-85%) y la confusion matrix:



En ella podemos ver el diagnóstico real en el eje de ordenadas y el diagnóstico predecido en el eje de abscisas. Vemos que, de los 32 tumores reales, el modelo ha acertado 30 y ha predicho 2 como no tumor, y de los 19 no tumores, ha acertado 13 y ha fallado 6, es decir, que ha dado por tumores 6 casos que no lo son.

Lo más grave es que pase por alto tumores, sin embargo, lo que más le cuesta es acertar los no tumores.

##**Próximos pasos**

Llevar a cabo un tratamiento de las imágenes más profundo y aplicar otros modelos de Machine Learning para intentar que el modelo aumente su porcentaje de acierto.

