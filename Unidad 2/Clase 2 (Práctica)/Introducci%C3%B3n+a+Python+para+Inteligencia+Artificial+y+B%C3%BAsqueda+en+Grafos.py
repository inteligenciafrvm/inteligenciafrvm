
# coding: utf-8

# # Introducción a Python para Inteligencia Artificial y Búsqueda en Grafos
# 
# 5to año - Ingeniería en Sistemas de Información
# 
# Facultad Regional Villa María

# ## Introducción
# 
# Python es un lenguaje de software libre, no tipado, interpretado y orientado a objetos. Se presenta como principal elección entre los lenguajes utilizados para aplicaciones de inteligencia artificial, particularmente de machine learning, junto con R, Matlab y Octave (dependiendo la comunidad que lo use).
# 
# Posee un enorme ecosistema de librerías de machine learning. Esto sumado a su flexibilidad y su simpleza de uso hacen que sea uno de los lenguajes más populares para la computación científica.
# 
# Algunos IDEs: Jupyter (web), Pycharm, Rodeo.
# 

# ### jupyter notebook (próximamente será reemplazado por jupyter lab)
# 
# Aplicación web de código abierto que permite crear y compartir documentos (notebooks) mostrando iterativamente el flujo de código en lenguajes como python y R, permitiendo incluir gráficos, explicaciones en Markdown y fórmulas en LaTeX.
# 
# Resulta un gran lenguaje para aplicaciones pequeñas de machine learning ya que permite utilizar el código de a fragmentos. Cada fragmento puede ser de texto o de código (dependiendo la selección del combo box superior: si está "Markdown" seleccionado tomará a la celda como texto - de estar "Code" tomará el fragmento como código).

# ¿Cómo instalarlo?
# 
# - Descargar la plataforma para DataScience "Anaconda" desde
# 
# 	https://docs.anaconda.com/anaconda/install/
# 
# 	Se deberá seleccionar la versión del intérprete Python (3.6 recomendada).
# 
# 
# - Seguir las intrucciones de instalación de Anaconda según el sistema operativo.
# 
# ¿Cómo usarlo?
# 
# * En Linux: abrir la terminal, situarse en el directorio donde está el notebook, ejecutar 
#         jupyter notebook 
#     y abrir el notebook desde la interfaz web.
# 
# 
# * En Windows: abrir la consola de windows (cmd), ejecutar 
#         jupyter notebook 
#     y abrir el notebook desde la interfaz web.
# 
# Prueben ejecutar y mirar el código del notebook de esta clase! Más notebooks de prueba: [https://try.jupyter.org/](https://try.jupyter.org/).
#     
# Algunas extensiones interesantes para notebook (incluídas variables de entorno): https://ndres.me/post/best-jupyter-notebook-extensions/. 
# También está disponible como Beta el ambiente que va a reemplazar a jupyter notebook, https://github.com/jupyterlab/jupyterlab. Mejora varios aspectos, pero todavía no cuenta con extensiones.

# Info adicional:
# 
# * [Markdown](https://daringfireball.net/projects/markdown/) es un conversor de texto plano a HTML. En los notebooks funciona como suerte de versión "liviana" de LaTex.
# * [LaTex](http://www.latex-project.org/) es un sistema de tipografía de alta calidad, que permite armar documentos con una sintaxis predefinida. Su sintaxis es mucho más verborrágica comparado con las herramientas típicas de ofimática, pero es mucho más flexible. Para armar documentos completos en LaTex de forma sencilla recomendamos [Overleaf](https://www.overleaf.com/) (editor LaTex Web). Por su parte, a LaTex se lo utiliza principalmente para fórmulas en los notebooks.
# 
# Para sus explicaciones en texto, jupyter por defecto utiliza Markdown. Para utilizar LaTex, debe encerrarse la sentencia en \$ ecuación \$ para ecuaciones en la misma línea o bien entre \$\$ ecuación \$\$ para ecuaciones en una nueva línea.
# 
# Ejemplo $$\lim_{h \to 0} \frac{f(a+h)-f(a)}{h}$$

# Veamos algo de código al ver una de las estructuras principales de Python...

# In[1]:

# estructura fundamental de python: listas

lista = [1,2,3,6]  # encerrar la estructura con [] es lo que denota que es una lista
print(lista)


# In[2]:

lista2 = [1,2] + [3,6]  # principal característica: listas completamente flexibles
print(lista2)


# In[3]:

# cuál es el elemento con índice 1 de la lista?
print(lista[1])  # notar que Python y numpy indexan desde el 0


# In[4]:

print(lista[1:])  # "1:" significa "el elemento de índice 1 y todos los elementos siguientes"


# In[5]:

# de manera similar...

print(lista[:4])  #(todos los elementos de la lista hasta aquel de índice 4, sin incluírlo)


# In[6]:

# o también...

print(lista[:])  # todos los elementos de la lista


# In[7]:

lista.append(5) 
lista.append('a')  # las listas nativas de python no se establecen para un tipo de dato en concreto
print(lista)


# In[8]:

matriz_1 = [[1,2], [3,4], [5,6]]  # una matriz es una lista de listas. Matriz de 3x2
print(matriz_1)


# In[9]:

# elemento de la fila 1, columna 1:
print(matriz_1[1][1])


# ### numpy
# 
# Librería open-source que dota a Python de manejo de vectores y matrices, junto con un conjunto de librerías para procesarlas, asemejando a python al manejo estilo Matlab.
# 
# El bloque básico de numpy es el ndarray, un array multidimensional de elementos homogéneos y de tamaño fijo. Estos objetos nunca son creados directamente, sino por un método destinado a tal efecto.
# 

# In[10]:

import numpy as np

# una forma fácil de crear un array de numpy es hacer una lista estándar de python
# y usarla como parámetro para ejecutar np.array sobre ella.
array = np.array([1,2,3,4,5])
print(array)


# In[11]:

# cuidado con la diferencia entre una lista "estándar" de python y un array de numpy!
print(type([1,2,3,4,5]), type(array))


# In[12]:

print(array[0])


# In[13]:

matriz = np.array([[1,2],[3,4],[5,6],[7,8]]) # para numpy, una matriz también es una lista de listas
print(matriz)


# In[14]:

matriz_2 = np.array(matriz_1)
print(matriz_2)


# In[15]:

# similarmente, no confundir la matriz que generamos con la lista de listas de python!!!
# son de dos tipos distintos
print(type(matriz_1), type(matriz_2))


# In[16]:

# Principal ventaja de numpy: nos permite realizar fácil y eficientemente operaciones a nivel de lista

matriz_2 = 3 * matriz # una de las muchas operaciones que podemos hacer con la lista como un todo
print(matriz_2)


# In[17]:

matriz_3 = matriz ** 2  # elevamos al cuadrado todos los elementos de matriz
print(matriz_3)


# In[18]:

print(matriz[1,1])


# In[19]:

print(matriz[1,:]) # pedimos todos los elementos de la segunda fila


# In[20]:

# pedimos los elementos 2 y 3 de la segunda columna
# (Notar que 2:4 equivale al intervalo [2,4), es decir que el elemento de índice 4 está excluído)
print(matriz[2:4, 1])


# In[21]:

print(np.shape(matriz)) # función muy útil, nos permite ver el orden de la estructura con la que trabajamos


# In[22]:

print(np.linspace(1,2,5)) # método muy útil para crear arrays de una dimensión


# In[23]:

# comando muy útil para obtener ayuda sobre paquetes, métodos y demás

help(np.array)

# También puede utilizarse sin argumentos, lo cual invoca a la ayuda interactiva del sistema


# ### matplotlib
# 
# matplotlib es la librería estándar para la visualización de datos, algo que resulta fundamental para machine learning. Es muy flexible y simple de usar. Se integra con jupyter, lo cual permite ver gráficos en un notebook.
# 
# La función principal de la misma es plot(), la cual toma un número variable de argumentos y grafica de forma tan simple como escribiendo plot(x,y)

# In[24]:

import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100) # x= 0.0, 0.101, 0.202,..., 10.0
plt.plot(x, np.sin(x))
plt.show()


# In[25]:

x = np.random.normal(size=500)
y = np.random.normal(size=500)
plt.scatter(x, y); # plot de puntos
plt.show()


# ### Aplicación de python: Búsqueda en grafos
# 
# Para adquirir práctica en el lenguaje python, vamos a empezar con una aplicación básica antes de utilizar librerías de alto nivel: implementando un algoritmo de búsqueda en grafos. Parte de esta sección está basada en el curso [Introduction to AI 2017](https://materiaalit.github.io/intro-to-ai-17/), dictado en la Universidad de Helsinki, Finlandia.

# ### Repaso:
# 
# Problema de búsqueda: informalmente, es un problema en donde se parte de un estado inicial (representado por un nodo) y debe encontrarse la forma de llegar a un estado objetivo, dado un espacio de estados posibles y conexiones entre los mismos (representadas por arcos).
# 
# Se define por:
# 
# * Un **conjunto de nodos o estados**.
# * Un **nodo inicial**.
# * Un conjunto de **nodo objetivo**.
# * Una **función** que nos indica si un nodo es o no objetivo.
# * Una **función sucesora**, que mapea un nodo a un conjunto de nuevos nodo.

# Además
# 
# * Un **camino** es una secuencia de nodos.
# * Una **solución** es un camino que lleva del estado inicial a un estado final.

# ¿Cómo obtener una solución? Repasamos dos algoritmos básicos
# 
# * Búsqueda en anchura - Breadth First Search (BFS)
# 
# ![Grafo](images/graph_tree_breadth.png)
# 
# Fuente: https://en.wikipedia.org/wiki/File:Breadth-first-tree.svg

# * Búsqueda en profundidad - Depth First Search (DFS)
# 
# 
# ![Grafo](images/graph_tree_depth.png)
# 
# Fuente: https://en.wikipedia.org/wiki/File:Depth-first-tree.svg

# Pseudocódigo de ejemplo:
# 
#         función búsqueda_básica(nodo_inicial)
#             lista_nodos = Lista()
#             nodos_visitados = Lista()
#             
#             # lista_nodos es la lista de nodos pendientes de recorrer
#             lista_nodos.agregar(nodo_inicial)
#             
#             si lista_nodos no está vacía hacer:
#                 nodo = lista_nodos.primer_elemento()  # obtenemos el 1er elemento de la lista
#                 remover nodo de lista_nodos
#                 
#                 si nodo no está en nodos_visitados hacer:
#                     nodos_visitados.agregar(nodo)
#                     
#                     si es_nodo_objetivo(nodo)
#                         devolver solución     # se alcanzó un estado objetivo
#                     fin si
#                     lista_nodos.agregar(obtener_nodos_vecinos(nodo))
#                 fin si
#             fin si
#             
#             devolver []                       # (si ningún estado objetivo pudo ser alcanzado)
#         
#         fin función
# 

# Vamos a implementar el primer grafo del ejemplo anterior:
# 
# ![Grafo](images/graph_tree_breadth.png)
# 
# Primero vamos a declarar algunas variables principales...

# In[26]:

nodos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
nodo_de_inicio = 1
nodos_objetivo = [11]

# estructura de mapeo en python: dict mapea valores a objetos
mapeos = {  # la estructura encerrada con llaves '{}' denota que la misma es un diccionario
    1: [2, 3, 4], #1 es la clave y [2, 3, 4] son los valores asociados a la misma
    2: [1, 5, 6],
    3: [1],
    4: [1, 7, 8],
    5: [2, 9, 10],
    6: [2],
    7: [4, 11, 12],
    8: [4],
    9: [5],
    10: [5],
    11: [7],
    12: [7],
}



# Considerando las funciones internas de Python, nos restarían de implementar dos funciones antes de pasar al algoritmo principal:
# 
#         obtener_nodos_vecinos(nodo)
#         es_nodo_objetivo(nodo)
#         
# * Implementar la primera función depende de cómo están implementados los nodos. En nuestro caso vamos a usar listas de números para representarlos.
# * Para la segunda función, necesitamos haber definido algún criterio para decidir qué hace que un nodo sea objetivo.

# In[27]:

# implementamos la primer función pendiente: obtener_nodos_vecinos(nodo)

def obtener_nodos_vecinos(nodo):
    # mapeos[nodo] va a devolver el objeto asociado a la clave <nodo>, en este caso la lista de nodos
    return mapeos[nodo]


# In[28]:

# implementamos la segunda función

def es_nodo_objetivo(nodo):
    if nodo in nodos_objetivo:
        return True
    else:
        return False


# In[29]:

# Ahora ya podemos implementar el algoritmo básico de búsqueda
def busqueda_basica(nodo_inicial):  # def en Python declara una función
    lista_nodos = []
    nodos_visitados = []

    lista_nodos.append(nodo_inicial)  # el método append agrega un elemento a una lista

    while lista_nodos is not []:  # verifica si la lista de nodos esta vacía
        nodo = lista_nodos[0]  # toma el primer elemento de la lista de nodos
        lista_nodos.pop(0)  # pop(<indice>) elimina de la lista el elemento del nodo con índice <indice>

        # se comprueba si el nodo no ha sido visitado previamente, para no repetir cálculos
        if nodo not in nodos_visitados:

            nodos_visitados.append(nodo)  # inserta el nodo en la posición
            if es_nodo_objetivo(nodo):  # función que determina si un nodo es objetivo
                return nodos_visitados

            # obtener_nodos_vecinos: función que, dado un nodo, obtiene los nodos conectados con él
            nodos_vecinos = obtener_nodos_vecinos(nodo)

            for vecino in nodos_vecinos:
                if vecino not in nodos_visitados:
                    lista_nodos.append(vecino)

    return None


# In[30]:

# vamos a probar llamarlo, y compararlo con el grafo:
print(busqueda_basica(nodo_de_inicio))


# ¿Qué algoritmo de búsqueda está implementando?

# ![Grafo](images/graph_tree_breadth.png)

# ### Búsqueda Best-First, informada y A*.
# 
# ¿Qué ocurriría si los caminos a recorrer tuvieran un costo? Por ejemplo, ir de un nodo a otro podría insumir una cierta cantidad de tiempo, diferente a la de ir hacia otro nodo vecino.
# 
# Una opción para tratar este problema es la de utilizar el criterio Best-First para ordenar los nodos.
# 
# * En tal caso, a nivel de implementación sería conveniente cambiar la estructura de datos en *lista_nodos* de lista a [cola de prioridades](https://es.wikipedia.org/wiki/Cola_de_prioridades), que ordenaríamos con un determinado criterio, por ejemplo la distancia en metros de un nodo a otro.
# * Esto se conoce como algoritmo de Dijkstra.
# * Si todos los nodos tuvieran el mismo costo, entonces el algoritmo de Dijkstra equivaldría a BFS.

# Problema con el algoritmo de Dijkstra o BFS: la búsqueda se extiende simétricamente en todas las direcciones, sin tender hacia donde se encuentra el estado objetivo.
# 
# * ¿Cómo solucionarlo? Un enfoque propuesto involucra emplear una heurística (estrategia para resolver un problema) que nos permita tomar en cuenta información sobre el nodo objetivo, para poder guiar la búsqueda hacia él partiendo desde el nodo en el que estamos.
# * Al utilizar búsqueda informada con una heurística, estamos utilizando el algoritmo A\*, donde el mismo ordena la cola de acuerdo a quien minimiza:
# 
# $$f(nodo) = c(nodo) + h(nodo)$$
# 
# donde $c(nodo)$ es el costo de llegar desde el nodo inicial hacia un determinado nodo, mientras que $h(nodo)$ es una heurística que estima el costo insumido en ir desde el nodo hacia el estado final.
# 

# ### **Trabajo Práctico 2**
# 
# Para los primeros ejercicios básicos de programación nos vamos a enfocar en empezar a practicar con cuestiones básicas de python y numpy, en particular motivando el manejo de listas y matrices (listas de listas), para adquirir práctica de cara a los próximos TPs. Algunas de las operaciones que aquí se piden ya fueron definidas al presentar la clase, se motiva a que busquen por internet las demás (por ejemplo, cómo implementar un *for* en python).
# 
# #### Ejercicios básicos
# 
# 1. Considerando el diccionario *mapeos* de nodos y sus mapeos, implementar una matriz de 12 filas y 3 columnas de modo que la fila 0 represente las conexiones del nodo 1, siendo la columna 0 la primera conexión, la columna 1 la segunda conexión, y la columna 2 la tercera conexión. Como consideración, si un nodo tiene menos de tres conexiones, en la fila y columna donde no tenga conexión debe ir un -1. Es decir,
# $$M = 
#  \begin{pmatrix}
#   2 & 3 & 4 \\
#   1 & 5 & 6 \\
#   1 & -1 & -1 \\
#   \vdots  & \vdots  & \vdots
#  \end{pmatrix}
# $$
# 
# 2. Obtener dos submatrices $M_1$ y $M_2$ a partir de la matriz $M$ del ejercicio anterior: una que considere todas las filas y la columa 0 y 1 además de otra que considere todas las filas y la columna 1 y 2. Es decir,
# $$M_1 = 
#  \begin{pmatrix}
#   2 & 3 \\
#   1 & 5 \\
#   1 & -1 \\
#   \vdots  & \vdots
#  \end{pmatrix}
# M_2 = 
#  \begin{pmatrix}
#   3 & 4 \\
#   5 & 6 \\
#   -1 & -1 \\
#   \vdots  & \vdots
#  \end{pmatrix}
# $$
# 
# A partir de $M_1$ y $M_2$, obtener el producto matricial $R = M_1 \times (M_2)^T$ e imprimir, como resultado, las filas 5 a 8 (fila 8 inclusive) y todas las columnas de la matriz resultante $R$. Ayuda: para el producto matricial y la transpuesta, buscar los métodos a tal efecto de la librería *numpy*.

# #### Ejercicios complementarios
# 
# 1. Cargar la matriz del ejercicio 1 de forma iterativa o de alguna forma que no involucre especificar manualmente los valores para cada (fila, columna) sino, por ejemplo, usando un *for* para recorrer los elementos del diccionario. Si esto ya está implementado en el ejercicio básico 1.1, ya se considera como resuelto. Ayuda: la forma más sencilla de recorrer un dict es mediante *for key, value in dict*, donde *key*, *value* y *dict* son las variables que representan la clave, su correspondiente valor y el diccionario, respectivamente.
# 2. Modificar la función *busqueda_basica* para que la misma implemente el algoritmo de búsqueda primero en profundidad (DFS). Hacer una prueba de escritorio simple (con un *print* como se hizo para verificar BFS) para determinar si la salida es correcta.
# 
# 
# #### Ejercicios extras
# 
# 1. Modificar la función *busqueda_basica* para que la misma implemente el algoritmo de búsqueda A\* de acuerdo a alguna heurística a elección que considere al nodo objetivo. Por ejemplo: cantidad de nodos que separan al nodo actual del nodo objetivo. Hacer una prueba de escritorio simple (con un *print* como se hizo para verificar BFS) para determinar si la salida es correcta.
# 
# **Fecha de entrega: 17/04/2018 23:55hs.**
# 
# Formato para entregar los ejercicios: los ejercicios deben ser cargados al campus en un archivo comprimido con el código (por ejemplo en un archivo de script de python \*.py, o en un archivo de jupyter notebook \*.ipynb). Antes de entregar se recomienda hacer un reinicio completo del ambiente de ejecución y correr nuevamente el código completo (especialmente si el trabajo está hecho en un notebook, tal reinicio puede hacerse desde *Kernel -> Restart Kernel and Run All Cells...*).
# 
# Recomendación: Para hacer los ejercicios, es conveniente exportar el notebook como archivo \*.py para así poder hacer un debug paso a paso en un editor como Pycharm. Para ello tienen que abrir el notebook desde el navegador, e ir a *File -> Download as -> Python (.py)* (si están utilizando *jupyter lab*, la exportación se hace desde *File -> Export Notebook As -> Executable Script*).
# 
# Recordatorio: la resolución de los ejercicios es **individual**. Está permitida la reutilización del código del notebook (por ejemplo para utilizar el código provisto).

# In[ ]:



