import pyglet
from pyglet.gl import *
from pyglet.math import Mat4, Vec3
from pyglet.window import key
from pyglet.graphics.shader import Shader, ShaderProgram
import numpy as np
import os

# Rutas
root = (os.path.dirname(__file__))

# Librerías propias
from librerias.scene_graph import *
from librerias.helpers import mesh_from_file
from librerias.drawables import Model
from librerias import shapes

# librerias adicionales t3
from utils.camera import FreeCamera, OrbitCamera
from utils.helpers import init_axis
from pyglet import math
import random
from utils.drawables import DirectionalLight

# --------------------------
# Ventana principal
# --------------------------
class Controller(pyglet.window.Window):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.time = 0.0
        self.gameState = 0
        # Agregar más atributos si se necesita (como cámara actual, inputs, etc.)

#CAMARA definida en una clase
class MyCam(FreeCamera):
    def __init__(self, position=np.array([0, 0, 0]), camera_type="perspective"):
        super().__init__(position, camera_type)
        self.direction = np.array([0,0,0])
        self.speed = 2

    def time_update(self, dt):
        self.update()
        dir = self.direction[0]*self.forward + self.direction[1]*self.right
        dir_norm = np.linalg.norm(dir)
        if dir_norm:
            dir /= dir_norm
        self.position += dir*self.speed*dt
        self.focus = self.position + self.forward

WIDTH = 1000
HEIGHT = 1000
window = Controller(WIDTH, HEIGHT, "Template T3")

# --------------------------
# Shaders
# --------------------------
if __name__ == "__main__":
    window.set_exclusive_mouse(True)

# Shader con textura
    vert_source = """
#version 330

in vec3 position;
in vec2 texCoord; 
in vec3 normal;

out vec2 fragTexCoord; 
out vec3 fragNormal;

uniform mat4 u_model = mat4(1.0);
uniform mat4 u_view = mat4(1.0);
uniform mat4 u_projection = mat4(1.0);

void main() {
    fragTexCoord = texCoord;
    fragNormal = mat3(u_model) * normal;
    gl_Position = u_projection * u_view * u_model * vec4(position, 1.0f);
}
    """
    frag_source = """
#version 330
in vec2 fragTexCoord;
in vec3 fragNormal;

uniform sampler2D u_texture;
uniform vec3 lightDir;
uniform vec3 lightDiffuse;
uniform vec3 lightAmbient;

out vec4 outColor;

void main() {
    vec3 normal = normalize(fragNormal);
    float diff = max(dot(normal, -lightDir), 0.0);
    vec3 diffuse = lightDiffuse * diff;
    vec3 ambient = lightAmbient;

    vec4 texColor = texture(u_texture, fragTexCoord);
    vec3 finalColor = texColor.rgb * (ambient + diffuse);

    outColor = vec4(finalColor, texColor.a);
}
    """

# Shader con color sólido
    color_vert = """
#version 330
in vec3 position;

uniform mat4 u_model = mat4(1.0);
uniform mat4 u_view = mat4(1.0);
uniform mat4 u_projection = mat4(1.0);

void main() {
    gl_Position = u_projection * u_view * u_model * vec4(position, 1.0f);
}
"""

    color_frag = """
#version 330
uniform vec4 u_color;
out vec4 outColor;

void main() {
    outColor = u_color;
}
"""

    # Pipelines 
    pipeline = ShaderProgram(Shader(vert_source, "vertex"), Shader(frag_source, "fragment"))
    color_pipeline = ShaderProgram(Shader(color_vert, "vertex"), Shader(color_frag, "fragment"))

    pipeline["lightDir"] = np.array([1, -1, -1], dtype=np.float32) / np.linalg.norm([1, -1, -1])
    pipeline["lightDiffuse"] = np.array([1, 1, 1], dtype=np.float32) 
    pipeline["lightAmbient"] = np.array([0.1, 0.1, 0.1], dtype=np.float32)

    # --------------------------
    # Cargar modelos
    # --------------------------
    #Planicie
    grass = Texture(root + "/assets/grass.png", minFilterMode=GL_NEAREST, maxFilterMode=GL_NEAREST)
    face_uv = [0, 0, 1, 0, 1, 1, 0, 1]
    texcoords = face_uv * 6
    cube = Model(shapes.Cube["position"], texcoords, index_data=shapes.Cube["indices"], 
                 normal_data=shapes.Cube["normal"])

    #Esfera
    sphere = mesh_from_file(root + "/assets/sphere.obj")[0]['mesh']

    #Cinamoroll
    cinamon = mesh_from_file(root + "/assets/CinnamorollStand.obj")
    cinamon_texture = Texture(root + "/assets/Cinnamaroll_01.png", minFilterMode=GL_NEAREST, maxFilterMode=GL_NEAREST)
    
    #Miku
    miku = mesh_from_file(root + "/assets/plMikuV2.obj")
    miku_texture = Texture(root + "/assets/plMikuV2_Alb.png", minFilterMode=GL_NEAREST, maxFilterMode=GL_NEAREST)

    #corazon
    heart = mesh_from_file(root + "/assets/ItemHeart_Prefab.obj")[0]['mesh']
    heart_texture = Texture(root + "/assets/T_ItemHeart_Alb.png", minFilterMode=GL_NEAREST, maxFilterMode=GL_NEAREST)

    #togepiii
    togepi = mesh_from_file(root + "/assets/togepiidoll.obj")
    togepi_texture = Texture(root + "/assets/togepiidoll.png", minFilterMode=GL_NEAREST, maxFilterMode=GL_NEAREST)

    #frutilla
    strawberry = mesh_from_file(root + "/assets/Strawberry.obj")[0]['mesh']
    strawberry_texture = Texture(root + "/assets/Strawberry.png", minFilterMode=GL_NEAREST, maxFilterMode=GL_NEAREST)

    
    # --------------------------
    # Cámara
    cam = OrbitCamera(distance=4.0)
    cam.phi = 0
    cam.theta = 0.0001
    cam.update()
    cam_mode = 1
    # --------------------------

    axis = init_axis(cam)

    world = SceneGraph(cam)
    world.add_node("scene")

    # --------------------------
    # Grafo de escena
    # --------------------------

    #Planicie
    tile_size = 1.0
    grid_size = 10  # 10x10
    for i in range(-grid_size // 2, grid_size // 2):
        for j in range(-grid_size // 2, grid_size // 2):
            tile_name = f"grass_{i}_{j}"
            world.add_node(tile_name, mesh=cube, texture=grass, pipeline=pipeline)
            world[tile_name]["position"] = [i * tile_size, -0.01, j * tile_size]
            world[tile_name]["scale"] = [tile_size, 0.01, tile_size]

    #Esfera
    world.add_node("sphere", attach_to="scene",
                   mesh=sphere,
                   color=[*shapes.MAGENTA, 1.0],
                   pipeline=color_pipeline,
                   position=[0, 0.25, 0],
                   scale=[0.5, 0.5, 0.5])
    

    # Agregar objetos 2 y 3 con sus posiciones y texturas
    world.add_node("cinamon_1",
                   mesh=cinamon[0]['mesh'], 
                   texture=cinamon_texture,
                   pipeline=pipeline,
                   position=[2.5, 0.07, -1.7],
                   scale=[0.2, 0.2, 0.2])
    
    world.add_node("cinamon_2",
                   mesh=cinamon[0]['mesh'], 
                   texture=cinamon_texture,
                   pipeline=pipeline,
                   position=[-2.8, 0.07, 1.9],
                   scale=[0.2, 0.2, 0.2])
    
    
    world.add_node("heart_1",
                   mesh=heart, 
                   texture=heart_texture,
                   pipeline=pipeline,
                   position=[1.8, 0.1,  1.0],
                   scale=[0.2, 0.2, 0.3])
    
    world.add_node("heart_2",
                   mesh=heart, 
                   texture=heart_texture,
                   pipeline=pipeline,
                   position=[-2.5, 0.1, -1.0],
                   scale=[0.2, 0.2, 0.3])
    
    world.add_node("togepi_1",
                   mesh=togepi[0]['mesh'], 
                   texture=togepi_texture,
                   pipeline=pipeline,
                   position=[1.5, 0.06,  1.8],
                   scale=[0.1, 0.06, 0.1])
    
    world.add_node("togepi_2",
                   mesh=togepi[0]['mesh'], 
                   texture=togepi_texture,
                   pipeline=pipeline,
                   position=[-1.0, 0.06,  2.1],
                   scale=[0.1, 0.06, 0.1])
    
    world.add_node("strawberry_1",
                   mesh=strawberry, 
                   texture=strawberry_texture,
                   pipeline=pipeline,
                   position=[1.2, 0.1, -2.0],
                   scale=[0.2, 0.2, 0.2])
    
    world.add_node("strawberry_2",
                   mesh=strawberry, 
                   texture=strawberry_texture,
                   pipeline=pipeline,
                   position=[-1.5, 0.1, -1.9],
                   scale=[0.2, 0.2, 0.2])
    
    world.add_node("strawberry_3",
                   mesh=strawberry, 
                   texture=strawberry_texture,
                   pipeline=pipeline,
                   position=[1.9, 0.1, -1.0],
                   scale=[0.2, 0.2, 0.2])
        
    # Si agregas el personaje recuerda que se mueve junto con la escena
    # Añadir personaje Miku
    world.add_node("miku",
                   attach_to="sphere",
                   mesh=miku[0]['mesh'], 
                   texture=miku_texture,
                   pipeline=pipeline,
                   position=[0, 0.04, 0.7],
                   rotation=[0, 3, 0, 1],
                   scale=[0.5, 0.5, 0.5])
    
    #lista de objetos recolectables
    objetos_recolectables = [
        "cinamon_1",
        "cinamon_2",
        "heart_1",
        "heart_2",
        "togepi_1",
        "togepi_2",
        "strawberry_1",
        "strawberry_2",
        "strawberry_3"
    ]

    recolectados = {}

    #iluminacion
    sun = DirectionalLight(
        ambient=[0.3, 0.3, 0.3], 
        diffuse=[0.9, 0.9, 0.9],  
        specular=[0.3, 0.3, 0.3]      
    )

    world.add_node(
        "sun",
        light=sun,
        pipeline=pipeline,  # O color_pipeline, si es el que usas
        position=[0, 5, 0],  # solo visual, la luz direccional no depende de posición
    )

    #funcion colisiones
    def colision(obj_name, sphere_name="sphere", dist=0.4):
        #posiciones de los objetos
        position_obj = world.find_position(obj_name)
        position_sphere = world.find_position(sphere_name)
        #no hay colision
        if position_obj is None or position_sphere is None:
            return False
        #colision, retorna True
        return np.linalg.norm(position_obj - position_sphere) < dist

    # reubicacion en la esfera
    def random_pos(radio=0.3, extra= 0.02):
        #genera una posición aleatoria dentro de un radio alrededor de la esfera
        theta = random.uniform(0, 2 * np.pi)
        phi = random.uniform(0, np.pi / 2)
        x = radio * np.cos(phi) * np.sin(theta) 
        y = radio + extra
        z = radio * np.sin(phi) * np.sin(theta)
        return np.array([x, y, z])

    def sphere_update(dt):
        esfera_speed = 0.09
        esfera_pos = [0,0,0]

        #movimiento de la esfera
        if key.W in pressed_keys:
            esfera_pos[2] -= esfera_speed
        if key.S in pressed_keys:
            esfera_pos[2] += esfera_speed
        if key.A in pressed_keys:
            esfera_pos[0] -= esfera_speed
        if key.D in pressed_keys:
            esfera_pos[0] += esfera_speed

        if esfera_pos != [0, 0, 0]:
            pos = world["sphere"]["position"]
            world["sphere"]["position"] = [
                pos[0] + esfera_pos[0],
                pos[1],
                pos[2] + esfera_pos[2]
            ]

        
        #anclaje
        esfera_actual = np.array(world["sphere"]["position"])
        for obj in objetos_recolectables:
            if obj not in recolectados and colision(obj):
                #offset aleatorio sobre la esfera 
                recolectados[obj] = random_pos()

        # Actualiza la posición de los objetos anclados para que sigan a la esfera
        for obj, offset in recolectados.items():
            world[obj]["position"] = esfera_actual + offset

    def camera_update(dt):
        global cam_mode, cam
        if cam_mode == 1:
            # Vista aérea fija
            if isinstance(cam, OrbitCamera):
                cam.phi = 0
                cam.theta = 0.0001
                cam.update()
        elif cam_mode == 2:
            # Primera persona
            sphere_pos = world["sphere"]["position"]
            if isinstance(cam, FreeCamera):
                cam.position = np.array([sphere_pos[0], sphere_pos[1] + 0.4, sphere_pos[2]])
                cam.update()
        elif cam_mode == 3:
            # Tercera persona
            sphere_pos = world["sphere"]["position"]
            if isinstance(cam, FreeCamera):
                cam.position = np.array([sphere_pos[0], sphere_pos[1] + 1.0, sphere_pos[2] + 2.0])
                cam.focus = np.array([sphere_pos[0], sphere_pos[1] + 0.4, sphere_pos[2]])
                cam.update()

        pipeline["u_view"] = cam.get_view()
        pipeline["u_projection"] = cam.get_projection()

    # --------------------------
    # Update y render
    # --------------------------
    pressed_keys = set()  # Conjunto para rastrear las teclas presionadas

    @window.event
    def on_key_press(symbol, modifiers):
        pressed_keys.add(symbol)

        global cam_mode, cam
        if symbol == key._1:
            cam_mode = 1
            cam = OrbitCamera(distance=4.0)
            cam.phi = 0
            cam.theta = 0.0001
            cam.update()
            world.camera = cam

        elif symbol == key._2:
            cam_mode = 2
            sphere_pos = world["sphere"]["position"]
            cam = FreeCamera(position=[sphere_pos[0], sphere_pos[1] + 0.4, sphere_pos[2]])
            cam.update()
            world.camera = cam

        elif symbol == key._3:
            cam_mode = 3
            sphere_pos = world["sphere"]["position"]
            cam = FreeCamera(position=[sphere_pos[0], sphere_pos[1] + 1.0, sphere_pos[2] - 2.0])
            cam.focus = np.array([sphere_pos[0], sphere_pos[1] + 0.4, sphere_pos[2]])
            cam.update()
            world.camera = cam

    @window.event
    def on_key_release(symbol, modifiers):
        if symbol in pressed_keys:
          pressed_keys.remove(symbol)


    @window.event
    def on_mouse_motion(x, y, dx, dy):
        if cam_mode != 2:
            return
        
        cam.yaw += dx * .001
        cam.pitch += dy * .001
        cam.pitch = math.clamp(cam.pitch, -(np.pi/2 - 0.01), np.pi/2 - 0.01)

    def update(dt):
        world.update()
        
        axis.update()
        sphere_update(dt)
        camera_update(dt)
        

        window.time += dt

    @window.event
    def on_draw():
        glEnable(GL_DEPTH_TEST) 
        glEnable(GL_CULL_FACE)
        glClearColor(0.63, 0.6, 0.8, 0.0) # Color fondo
        window.clear()
        world.draw()

        

    pyglet.clock.schedule_interval(update, 1/60)
    pyglet.app.run()