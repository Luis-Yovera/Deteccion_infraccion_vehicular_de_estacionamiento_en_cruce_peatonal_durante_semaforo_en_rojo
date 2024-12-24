NOTAS DE LA VERSIÓN:

Al realizar la primera prueba, que fue al dejar correr el programa con los dos primeros minutos (o menos) de los videos "Prueba_1080.mp4" y "Semáforo.mp4", nos pudimos dar cuenta de lo siguiente:

Aunque si se realiza la captura de placas, esta no es precisa debido a que la calidad del video, y de cada frame, es baja, siendo esta 1080p. EasyOCR necesita una calidad alta de video para tener una mejor precisión del la lectura de caracteres.

Debido a esto se grabó un segundo video en calidad 4k, denominado "prueba_4k" al que se le empató el video del semáforo para que ambos originales sean simultáneos. Al dejar correr hasta que durante un rojo detecte una placa, se obtuvieron valores más precisos de la lectura. (resultados entregados en la presente carpeta).

--------------------------------------------------------------------------

Se pudo notar al dejar correr el programa durante lo cuatro minutos enteros del video los siguientes errores:

1. Debido a que la lectura de las placas se realiza cuadro por cuadro, el video de prueba (el direccionado a la berma) disminuye su velocidad para el procesamiento, mientras que el video del semaforo mantienen su velocidad durante toda la ejecución del programa, resultando en un descuadre al aparecer y desaparece la región de interés (ROI)

2. Ante el problema anterior, se genera un segundo problema, al terminar el video del semáforo antes que el video de la berma, se genera un error en el código que impide el procesamiento del video resultante (denominado: "traffic_detection_result") y la generación del archivo .CSV con los datos del número de frame analizado, coordenadas de la caja que encierra la placa, el texto leído de la placa y confidence.

Ante esto, para poder observar el funcionamiento se recomienda lo siguiente:

Para el video prueba de 1080p:

* Recomendamos detener el video (presionar tecla q) durante el segundo rojo, ya que al inicio del video que es en rojo si se puede ver la lectrua de placas 

Para el video prueba de 2k: 

* Al ser un video de mayor resolución, recomendamos detener el video después del primer rojo, debido a que la reoslución del video es mayor y el análisis de cada frame es mas lento, aumentando la diferencia de velocidad entre el video del semáforo y el video prueba.