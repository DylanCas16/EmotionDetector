# EmotionDetector

## Dataset

El dataset utilizado se puede encontrar y descargar a través del siguiente [enlace](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions/data)

## Especificaciones del equipo utilizado

Todo el desarrollo de la práctica se realizó en un portátil con las siguientes características:

- CPU: AMD Ryzen 7 7735HS with Radeon Graphics.
- GPU: NVIDIA GeForce RTX 4050 Laptop, CUDA 12.6.
- RAM: 16GB DDR5 4800 MT/s

## Detección de emociones

### Entorno de Anaconda utilizado

Existen tres posibles entornos para crear según las necesidades y preferencias del usuario. El primer entorno y más sencillo se crea con el siguiente comando. No obstante, este solo permite realizar una ejecución con la CPU debido a que la versión utilizada para `TensorFlow` (2.21) no cuenta con soporte nativo para la GPU en Windows (a partir de la versión 2.11), por lo que el tiempo de ejecución aumenta considerablemente. 

```bash
conda create --name VC_Emotion python=3.11
conda activate VC_Emotion

pip install tensorflow
pip install matplotlib mtcnn deepface tf-keras opencv-python==4.12.0.88 scikit-learn ipykernel
```

Para solucionar este problema existen dos vías: 

#### Instalación para el uso de GPU en Windows

Este método consiste en crear un entorno con versiones anteriores que sí son compatibles con GPU nativo en Windows, haciendo uso además del plugin `DirectML`. además, será necesario crear este nuevo entorno con una versión de Python anterior, ya que la 3.11 no soporta `TensorFlow 2.10`. El proceso de creación del nuevo entorno es el siguiente:

```bash
conda create --name VC_Emotion_gpu python=3.10
conda activate VC_Emotion_gpu

pip install "tensorflow-cpu==2.10.0" "tensorflow-directml-plugin" "keras>=2.10.0,<2.11" "protobuf>=3.9.2,<3.20" "numpy<2"
pip install matplotlib opencv-python==4.9.0.80 scikit-learn ipykernel mtcnn
pip install deepface "numpy<2"
```

Cabe mencionar el motivo del primer lote de dependencias y el de `deepface`, ya que hacerlo de otra manera es propenso a errores y no me permitía instalar correctamente todas. Al intentar instalar las depedencias `keras` y `deepface` sin tener en cuenta sus versiones, se produce un conflicto de versiones incompatibles entre estas dos y `protobuf`. La solución más sencilla es indicar los rangos de versiones en los que estos son compatibles, e instalar `deepface` sin dependencias extra para que no actualice ni corrompa las demás. No obstante, esto supone utilizar `Keras 2` en lugar de `Keras 3`, esto supone tener que instalar una versión de `numpy` anterior a la `2` para que no se rompa la ejecución.

Otro inconveniente con este entorno es que cuenta con ciertas funciones que no soporta nativamente en GPU, por lo que las ejecutará en CPU sin avisar y tendrá un rendimiento inferior a realizar la instalación mediante `WSL`. No obstante, no se considera imprescindible el rendimiento y velocidad de ejecución para esta práctica, por lo que se sacrificó a cambio de una implementación más simple.

#### Instalación con `WSL` para soporte de GPU nativo

Este método es mucho más largo, consiste en instalar `WSL` y montar el entorno allí. El primer paso es instalar `WSL2` en caso de no tenerlo, para ello existe el siguiente comando que debe ser ejecutado en `PowerShell` como administrador:

```bash
wsl --install
```

En caso de tener la versión `WSL1`, utilizar este comando para actualizarlo:

```bash
wsl --set-default-version 2
```

Es importante tener actualizado los drivers de NVIDIA, si ya están en Windows no es necesario volver a instalarlos para Linux ya que tiene soporte nativo con `WSL2`. Una vez instalado en su versión correcta, se abre la terminal `WSL` y se instala `Miniconda` a través del siguiente comando:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Durante el proceso de instalación aparecerán distintos mensajes de confirmación, es necesario aceptar todos. En caso de rechazar el último, que pregunta si iniciar automáticamente conda al abrir una terminal, se puede iniciar de dos formas:

- Inicio temporal: este comando inicia `Miniconda` en esa sesión exclusivamente.

```bash
eval "$(/home/user/miniconda3/bin/conda shell.bash hook)"
```

- Inicio permanente: inicia `Miniconda` en todas las sesiones (es necesario reiniciar la terminal para que tenga efecto):

```bash
/home/user/miniconda3/bin/conda init bash
```

En ambos casos es necesario localizar la ruta completa de `conda`.

Una vez instalado e iniciado `Miniconda`, se instala `TensorFlow` con el soporte nativo de GPU junto con el resto de dependencias de la tarea:

```bash
pip install "tensorflow[and-cuda]"
pip install matplotlib mtcnn deepface tf-keras opencv-python==4.12.0.88 scikit-learn ipykernel
```

Finalmente, se registra este entorno en `jupyter` para poder utilizarlo:

```bash
python -m ipykernel install --user --name=vc_emotion_wsl
```

Una vez montado el entorno en `WSL`, se procede con el montaje del repositorio dentro de Linux para un rendimiento nativo, en lugar de estar realizando en todo momento una conversión Windows a Linux. Para ello se copia el repositorio (generalmente se encuentra en el directorio `/mnt`) y se abre con VS Code desde la propia terminal `WSL` o con el plugin del mismo nombre en el VS Code de Windows, permitiendo así ejecutar tanto el repositorio como el entorno desde la propia Linux.


### Procesamiento del dataset y preparación para los modelos de entrenamiento (`MTCNN`)

#### Definición de variables

Una vez instalado el entorno deseado e importado el dataset bruto en un directorio llamado `dataset/`, es necesario ajustarlo para que los modelos puedan utilizarlo sin problemas. Para ello se hace uso de `MTCNN`, con el que se detectarán todas las caras del dataset y se normalizarán para que los modelos puedan procesarlas. 

Posteriormente, se irán añadiento a un nuevo directorio llamado `dataset_processed/` en un reparto de 80% para entrenamiento y 20% para test (este último se define en la variable `TEST_SIZE`). 

Para evitar desbordamientos en la memoria debido al tamaño de alguna imagen, se incluyó una variable llamada `MAX_DIM`, más adelante se explicará su función con más detalle. 

Sumado a esto, la variable `RANDOM_STATE` indica la semilla reproducible para la repartición de los directorios `train/` y `test/`. 

Por último, la variable `MARGIN` genera un espacio extra fuera del area de la cara que detecte `MTCNN` para evitar realizar recortes demasiado ajustados.

La siguiente celda está dedicada completamente al reparto de las imagenes para el entrenamiento y validación, indicando las imágenes totales del dataset y las resultantes de cada directorio. Para la creación de las parejas simplemente se cogió el nombre de la imagen y del directorio en el que se encuentra, ya que todas las imagenes de cada clase se encontraban en un mismo directorio con el nombre de esta.

#### Funciones auxiliares

Posteriormente, se procede con las funciones de detección, procesamiento y recorte de las imágenes. La función que engloba todo se llama `align_and_crop()` y consta de varias partes:

- **Detección de la cara y los ojos**. Para ello se obtiene los `keypoints` de la cara que haya obtenido el modelo y se extraen los puntos que definen los ojos, una variable para cada uno.

- **Alineamiento de los ojos**. Con las coordenadas de los ojos, se procede a alinearlos rotando la imagen hasta que se encuetren en la misma altura.

- **Creación de la caja de la cara**. Se obtiene una nueva caja para las caras a partir de la imagen rotada con los ojos alineados. Posteriormente, se expande un poco cada una en proporción al margen indicado en la segunda celda.

- **Reescalado e imagen final**. Por último, se realiza el reescalado de la imagen a los píxeles indicados en la segunda celda y se devuelve.

En la siguiente celda se encuentra la función `process_split()`

El dataset final obtenido se puede comprobar a través de la última celda, que muestra los siguientes resultados:

| Emoción  | Fallos (train) | Fallos (test) | Total |
|----------|----------------|----------------|-------|
| Angry    | 1330           | 306            | 1636  |
| Fear     | 1600           | 398            | 1998  |
| Happy    | 1965           | 471            | 2436  |
| Sad      | 2373           | 603            | 2976  |
| Surprise | 911            | 217            | 1128  |

**Total de imágenes:** 59099
**Total de fallos de detección MTCNN:** 10174
**Porcentaje de fallo:** 17.22%
