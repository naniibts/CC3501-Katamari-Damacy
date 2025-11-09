# CC3501-Katamari-Damacy
# Katamari Damacy - Tarea 3 Gráfica

Un juego inspirado en Katamari Damacy implementado con OpenGL y Pyglet, donde controlas una esfera que recolecta objetos en su camino.

## Descripción
Este proyecto recrea la mecánica principal de Katamari Damacy: se controla una esfera mágica que rueda por un campo y va "pegando" objetos a medida que los toca.
Los objetos recolectados quedan adheridos a la esfera, creando un efecto visual similar al juego original.

## Requisitos
python >= 3.8
pyglet
numpy
trimesh
pillow
OpenGL

## Características

1. **Sistema de Recolección**: Detecta colisiones entre la esfera y los objetos coleccionables
2. **Anclaje Dinámico**: Los objetos recolectados se posicionan aleatoriamente alrededor de la esfera
3. **Múltiples Vistas de Cámara**:
    -Vista aérea fija (Tecla 1)
    -Primera persona (Tecla 2)
    -Tercera persona (Tecla 3)


4. **Iluminación**: Sistema de iluminación direccional para dar profundidad a la escena
5. **Modelos 3D**: Incluye personajes y objetos de elección personal 

## Controles
Movimiento de la Esfera

**W** - Mover hacia adelante
**S** - Mover hacia atrás
**A** - Mover hacia la izquierda
**D** - Mover hacia la derecha

## Cámaras

1 - Vista aérea (cenital)
2 - Primera persona (Movimiento del mouse - Rotar la cámara (yaw y pitch))
3 - Tercera persona


## Instalación

Clona el repositorio:
```
bashgit clone [url-del-repositorio]
cd Tarea_3_Daniela_Olave/Tarea3
```

Instala las dependencias:
```
bashpip install pyglet numpy trimesh pillow PyOpenGL networkx
```
Ejecuta el juego:
```
bashpython Tarea_3.py
```


## Objetos Recolectables
El juego incluye 9 objetos coleccionables distribuidos por el campo:

**2x Cinnamoroll** (Personaje)
**2x Corazones**
**2x Togepi** (Pokémon)
**3x Fresas**

Cada objeto tiene su posición única en el mundo y puede ser recolectado al hacer contacto con la esfera.


## Grafo de Escena
Utiliza un sistema de grafo de escena para manejar las jerarquías de objetos:

Los objetos recolectados se anclan como hijos de la esfera
Las transformaciones se propagan jerárquicamente
Miku (personaje principal) está anclado permanentemente a la esfera

## Mecánicas del Juego

**Inicio**: La esfera comienza en el centro del campo con un personaje haciendo el gesto de empujar
**Exploración**: Mueve la esfera con WASD para buscar objetos
**Recolección**: Al tocar un objeto, este se "pega" a la esfera
**Crecimiento**: Los objetos recolectados orbitan alrededor de la esfera
**Cámara**: Cambia entre vistas para mejor control y visualización


