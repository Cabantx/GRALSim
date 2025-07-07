import bpy
import numpy as np
from numpy.fft import fftfreq, ifft2, ifftshift
import os
import math

# -------------------------------------------------------------------------------#
#        ESZENAREN PROPIETATEAK ETA INTERFAZEAREN EGUNERAKETA FUNTZIOAK          #
# -------------------------------------------------------------------------------#

def update_ocean_simulation(self, context):
    
    #Funtzio hau deitzen da UIko balore bat aldatzen den bakoitzean. Simulazioaren parametroak eguneratzen ditu eta bistaren "refresh" bat egiten du
    
    scene = context.scene
    
    #Simulazioaren parametroak eguneratu
    try:
        wind_dir_x = math.cos(math.radians(scene.ocean_props.wind_direction_angle))
        wind_dir_y = math.sin(math.radians(scene.ocean_props.wind_direction_angle))
        
        ocean_sim.wind_speed = scene.ocean_props.wind_force
        ocean_sim.wind_direction = np.array([wind_dir_x, wind_dir_y])
        ocean_sim.phillips_amplitude = scene.ocean_props.wave_height
        
        ocean_sim.initialize_spectrum()
        
    except NameError:
        pass
    
    #Eguzki-argiaren intentsitatea eguneratu
    try:
        if sun_obj_global:
            sun_obj_global.data.energy = scene.ocean_props.light_strength
    
    except (NameError, ReferenceError):
        pass
    
    if scene.frame_current == bpy.context.scene.frame_current:
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        update_ocean(scene)
        
#Interfazean bistaratuko diren propietateak hasieratu
class OceanSceneProperties(bpy.types.PropertyGroup):
    wind_direction_angle: bpy.props.FloatProperty(
        name="Haizearen norabidea",
        description="Haizearen norabidearen angelua (gradutan)",
        min=0,
        max=360,
        default=30,
        update=update_ocean_simulation
    )
    wind_force: bpy.props.FloatProperty(
        name="Haizearen indarra",
        description="Haizearen indarra, olatuen altueran eragina du",
        min=0.1,
        max=50.0,
        default=25.0,
        update=update_ocean_simulation
    )
    wave_height: bpy.props.FloatProperty(
        name="Olatuen altuera",
        description="Olatuen anplitudea",
        min=0.0,
        max=2.0,
        default=0.8,
        update=update_ocean_simulation
    )
    light_strength: bpy.props.FloatProperty(
        name="Argiaren indarra",
        description="Eguzkiaren argiaren intentsitatea",
        min=0.0,
        max=50.0,
        default=5.0,
        update=update_ocean_simulation
    )
    
# -------------------------------------------------------------------------------#
#                             INTERFAZEAREN PANELA                               #
# -------------------------------------------------------------------------------#

class OCEAN_PT_controls(bpy.types.Panel):
    #3D bistaraketaren alde batean panela sortzen du
    bl_label = "UIaren kontrolak"
    bl_idname = "OBJECT_PT_ocean_controls"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Simulatzailearen parametroak"
    
    def draw(self, context): 
        layout = self.layout
        scene = context.scene
        ocean_props = scene.ocean_props
        
        box = layout.box()
        box.label(text="Simulazioaren parametroak:")
        box.prop(ocean_props, "wind_direction_angle")
        box.prop(ocean_props, "wind_force")
        box.prop(ocean_props, "wave_height")
        box.prop(ocean_props, "light_strength")
        

# -------------------------------------------------------------------------------#
#                SKYBOX-AREN SIMULAZIOA ETA EGUZKIA EZARRI                       #
# -------------------------------------------------------------------------------#

def setup_skybox_hdri(image_path, initial_sun_rotation_z=0, initial_sun_elevation=45):
    
    #HDRI irudi bat katgatu. Honek argiztapen globala emango du fondoaz aparte.

    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worllds.new("World")
        bpy.context.scene.world = world
        
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    
    for node in nodes:
        nodes.remove(node)
        
    background_node = nodes.new(type='ShaderNodeBackground')
    background_node.location = (0, 0)
    
    env_texture_node = nodes.new(type='ShaderNodeTexEnvironment')
    env_texture_node.location = (-300, 0)
    
    #HDRI irudia kargatu
    try:
        image = bpy.data.images.load(image_path)
        env_texture_node.image = image
    except RuntimeError:
        print(f"Errorea: Ezin da irudia kargatu {image_path} direktoriotik.")
        return
    
    tex_coords_node = nodes.new(type='ShaderNodeTexCoord')
    tex_coords_node.location = (-600, 0)

    mapping_node = nodes.new(type='ShaderNodeMapping')
    mapping_node.location = (-450, 0)
    mapping_node.vector_type = 'TEXTURE'
    
    mapping_node.inputs['Rotation'].default_value.z = math.radians(initial_sun_rotation_z)

    # Nodoak konektatu: Generated -> Vector (TexCoord) -> Vector (Mapping) -> Vector (EnvTexture)
    links.new(tex_coords_node.outputs['Generated'], mapping_node.inputs['Vector'])
    links.new(mapping_node.outputs['Vector'], env_texture_node.inputs['Vector'])

    links.new(env_texture_node.outputs['Color'], background_node.inputs['Color'])

    world_output_node = nodes.new(type='ShaderNodeOutputWorld')
    world_output_node.location = (200, 0)
    links.new(background_node.outputs['Background'], world_output_node.inputs['Surface'])

    print(f"Skybox HDRI konfiguratuta '{image_path}' irudiarekin, {initial_sun_rotation_z} graduko hasierako rotazioarekin.")
    

def setup_sun_light(initial_sun_rotation_z=0, initial_sun_elevation=45):
    
    #BLENDER-en eszenan 'SUN' motako argiztapen iturri bat konfiguratu
    
    sun_light_obj = None
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT' and obj.data.type == 'SUN':
            sun_light_obj = obj
            break

    #Ez bada 'SUN' motako argiztapen objektu bat aurkitu, berri bat sortu
    if not sun_light_obj:
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 0))
        sun_light_obj = bpy.context.active_object
        sun_light_obj.name = "SceneSunLight"
        sun_light_obj.data.energy = 5.0 # Argiaren intentsitatea
        sun_light_obj.data.color = (1.0, 1.0, 0.9) #Kolore pixkat horia
        print("'SceneSunLight' eguzki argia sortuta.")
    else: print("'SceneSunLight' existitzen da.")
    
    # Eguzkiaren hasierako posizioa ezarri
    sun_light_obj.rotation_euler.z = math.radians(initial_sun_rotation_z)
    sun_light_obj.rotation_euler.x = math.radians(90 - initial_sun_elevation)

    print(f"Eguzkiaren posizioa Z ardatzean: {initial_sun_rotation_z}°, Altuera: {initial_sun_elevation}°.")
    return sun_light_obj

# ---------------------------------------------------------------------#
#                      OCEAN SIMULATOR KLASEA                          #
# ---------------------------------------------------------------------#

class OceanSimulator:
    
    #Phillipsen espektroa erabiliz ozeanoaren gainazala simulatzeko klasea
    #Altuera- eta desplazamendu-mapak sortzen ditu
    
    def __init__(self, resolution, wind_speed, wind_direction, domain_size, phillips_amplitude):
        self.N = resolution
        self.domain_size = domain_size
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction / np.linalg.norm(wind_direction)
        self.phillips_amplitude = phillips_amplitude
        self.initialize_spectrum()

    def phillips_spectrum(self, kx, ky):
        
        #Phillipsen espektrua kalkulatu (kx,ky) maiztasun baterako
        #Espektro honek energiaren hedapena deskribatzen du olatuetan zehar
        
        g = 9.81
        k = np.sqrt(kx**2 + ky**2)
        k[k == 0] = 1e-6 #Zerorekin ez zatitzeko

        L = (self.wind_speed**2) / g
        k_dir = np.stack([kx / k, ky / k], axis=-1)
        dot_product = np.sum(k_dir * self.wind_direction, axis=-1)

        #Phillipsen espektrua norabide eta arintzearekin
        spectrum = self.phillips_amplitude * (np.exp(-1.0 / (k**2 * L**2)) * (dot_product**2) / (k**4))
        spectrum *= np.exp(-k**2 * 1e-4)

        spectrum[k < 1e-3] = 0 
        
        return spectrum
    
    def initialize_spectrum(self):
        
        #Fourierren anplitudeak konplexuak hasieratu Phillipsen espektroan eta zarata gaussierrean oinarrituta
        
        #Maiztasunen taula
        kx = fftfreq(self.N, d=self.domain_size/self.N) * 2 * np.pi
        ky = fftfreq(self.N, d=self.domain_size/self.N) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(kx, ky)
        
        spectrum = self.phillips_spectrum(self.KX, self.KY)
        
        rng = np.random.default_rng()
        self.A = (np.sqrt(spectrum / 2.0) * (rng.normal(size=spectrum.shape) + 1j * rng.normal(size=spectrum.shape)))

    def generate_heightmap(self, time):
        
        #Gainazalaren altuera-mapa kalkulatzen du emandako momentu puntualean
        
        #Denboraren bilakaera fourierren espazioan
        omega = np.sqrt(9.81 * np.sqrt(self.KX**2 + self.KY**2))
        h_t = self.A * np.exp(1j * omega * time) + np.conj(self.A) * np.exp(-1j * omega * time)
        return ifft2(h_t).real
    
    def generate_displacements(self, time):
        
        #Desplazamendu-mapak kalkulatzen dira emandako momentu batean
        
        g = 9.81
        k_magnitude = np.sqrt(self.KX**2 + self.KY**2)
        omega = np.sqrt(g * k_magnitude)
        phase = self.KX * 0.0 + self.KY * 0.0 + omega * time

        zero_k_mask = (k_magnitude == 0)

        displacement_x_fft = np.zeros_like(self.A, dtype=complex)
        displacement_y_fft = np.zeros_like(self.A, dtype=complex)
        
        displacement_x_fft[~zero_k_mask] = 1j * (self.KX[~zero_k_mask] / k_magnitude[~zero_k_mask]) * self.A[~zero_k_mask]
        displacement_y_fft[~zero_k_mask] = 1j * (self.KY[~zero_k_mask] / k_magnitude[~zero_k_mask]) * self.A[~zero_k_mask]

        displacement_x = ifft2(ifftshift(displacement_x_fft)).real
        displacement_y = ifft2(ifftshift(displacement_y_fft)).real
        return displacement_x, displacement_y
    
# ------------------------------------------------------ #
#      ERPINAK ETA TRIANGELUAK SORTZEKO FUNTZIOAK        #
# ------------------------------------------------------ #

def generate_vertices(heightmap, displacements_x, displacements_y, size, domain_size):
    
    #Erpin lista sortzen du 3dn. Altuera- eta desplazamendu-mapak konbinatzen ditu
    
    vertices = []
    num_y, num_x = heightmap.shape
    
    step_x = domain_size / (num_x - 1)
    step_y = domain_size / (num_y - 1)
    origin_x = -domain_size / 2
    origin_y = -domain_size / 2

    z_scale_factor = 100.0

    for y in range(num_y):
        for x in range(num_x):
            x_coord = origin_x + step_x * x + 2*displacements_x[y][x]
            y_coord = origin_y + step_y * y + 2*displacements_y[y][x]
            z_coord = heightmap[y][x] * z_scale_factor
            vertices.append((x_coord, y_coord, z_coord))
    return vertices

def generate_tris(grid_size):
    
    #Triangelu (edo aldeak) sortzen ditu sarean, sarea karratuen bidez sortzen da eta. Bi triangelu sortzen ditu karratu bakoitzeko
    
    tris = []
    num_x = grid_size[0]
    num_y = grid_size[1]

    for y_idx in range(num_y - 1):
        for x_idx in range(num_x - 1):
            
            #Uneko karratuaren 4 erpinen koordenatuak lortu
            v_bottom_left = y_idx * num_x + x_idx 
            v_bottom_right = y_idx * num_x + (x_idx + 1)
            v_top_left = (y_idx + 1) * num_x + x_idx
            v_top_right = (y_idx + 1) * num_x + (x_idx + 1)

            #Bi triangelu sortu karratu bakoitzeko. Triangelu bikoitiak alde batera orientatuta eta bakoitiak bestera
            if (x_idx + y_idx) % 2 == 0:
                tris.append((v_bottom_left, v_bottom_right, v_top_right))
                tris.append((v_bottom_left, v_top_right, v_top_left))
            else:
                tris.append((v_bottom_right, v_top_right, v_top_left))
                tris.append((v_bottom_right, v_top_left, v_bottom_left))
    return tris

# ------------------------------------------------------ #
#                BLENDERREN SAREA SORTU                  #
# ------------------------------------------------------ #

def create_ocean_mesh(resolution, domain_size):
    
    #Blenderren 'mesh' objektu bat sortzen du ozeanoaren gainazalerako
    
    mesh_name = "OceanMesh"
    obj_name = "OceanObject"
    
    if obj_name in bpy.data.objects: bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)
    if mesh_name in bpy.data.meshes: bpy.data.meshes.remove(bpy.data.meshes[mesh_name], do_unlink=True)
    
    mesh = bpy.data.meshes.new(mesh_name)
    obj = bpy.data.objects.new(obj_name, mesh)
    bpy.context.collection.objects.link(obj)
    
    num_vertices = int(resolution * resolution)
    vertices_temp = [(0, 0, 0)] * num_vertices
    edges = []
    faces = generate_tris((int(resolution), int(resolution)))
    
    mesh.from_pydata(vertices_temp, edges, faces)
    
    obj.data.shade_smooth()
    
    mesh.update()
    
    return obj

def update_ocean(scene):
    
    #Frame bakoitzeko exekutatzen da funtzioa mesh-aren geometria eguneratzeko, simulazioan oinarrituta
    
    frame = scene.frame_current
    fps = scene.render.fps
    time = frame / fps
    
    heightmap = ocean_sim.generate_heightmap(time)
    displacements_x, displacements_y = ocean_sim.generate_displacements(time)
    
    size = (int(ocean_sim.N), int(ocean_sim.N))
    
    vertices = generate_vertices(heightmap, displacements_x, displacements_y, size, ocean_sim.domain_size)
    
    mesh_obj = bpy.data.objects.get("OceanObject")
    if mesh_obj and mesh_obj.type == 'MESH':
        mesh = mesh_obj.data

        if len(mesh.vertices) != len(vertices):
            mesh.vertices.add(len(vertices) - len(mesh.vertices))

        mesh.vertices.foreach_set("co", np.array(vertices).flatten())

        mesh.validate()
        mesh.update()

def register_handlers():
    
    #Update_ocean funtzioa erregistratzen du event handler moduan frame aldaketa bakoitzean exekutatua izateko
    
    if update_ocean in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(update_ocean)
    bpy.app.handlers.frame_change_pre.append(update_ocean)
    
def unregister_handlers():
    
    # Update_ocean funtzioa erregistratu gabeko event handler moduan kentzen du
    
    if update_ocean in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(update_ocean)
    
# ------------------------------------------------------ #
#                  OZEANOAREN MATERIALA                  #
# ------------------------------------------------------ #
    
def add_ocean_material(object):
    
    #PBR (Principled BSDF) material bat gehitzen dio ozeanoaren objektuari, uraren propietateak dituena.
    #Nodo berriak gehitzen ditu urak itxura gardena, islatzailea eta uhin txikiduna izan dezan.
    
    mat_name = "OceanMaterial"
    if mat_name not in bpy.data.materials:
        material = bpy.data.materials.new(name=mat_name)
        material.use_nodes = True
        tree = material.node_tree
        
        for node in tree.nodes: tree.nodes.remove(node)

        principled_bsdf = tree.nodes.new(type='ShaderNodeBsdfPrincipled')
        principled_bsdf.location = (0, 0)
        
        material_output = tree.nodes.new(type='ShaderNodeOutputMaterial')
        material_output.location = (400, 0)
        
        tree.links.new(principled_bsdf.outputs['BSDF'], material_output.inputs['Surface'])

        # Uraren propietateak zehaztu Principled BSDF nodoan
        # Base Color: Oinarrizko kolorea
        principled_bsdf.inputs["Base Color"].default_value = (0.005, 0.02, 0.05, 1)  # Ur pixkat iluna sakonera sentsazioa emateko
        # Roughness: Zenbateko dispertsioa izango duen argiak gainazalean isladatzean
        principled_bsdf.inputs["Roughness"].default_value = 0.02 #Dispertsio gutxi isladapena handia izateko
        # Transmission: Gainazalaren transparentzia
        principled_bsdf.inputs["Transmission"].default_value = 1.0 #Balio altuena, ur garbiaren itxura emateko
        # IOR: Uraren errefrakzio indizea (1.333)
        principled_bsdf.inputs["IOR"].default_value = 1.333
        # Specular Tint: Argi islatuaren kolorea aldatu urarena hobeto simulatzeko
        principled_bsdf.inputs["Specular Tint"].default_value = 0.1
        # Clearcoat eta Clearcoat Roughness: Islapen geruza bat gehiago aplikatzen du
        principled_bsdf.inputs["Clearcoat"].default_value = 0.5
        principled_bsdf.inputs["Clearcoat Roughness"].default_value = 0.03 #Geruza oso leuna
        # Sheen eta Sheen Tint: Distira geruza gehigarri bat
        principled_bsdf.inputs["Sheen"].default_value = 0.05
        principled_bsdf.inputs["Sheen Tint"].default_value = 0.5
        
    else:
        material = bpy.data.materials[mat_name]

    if object.data.materials:
        object.data.materials[0] = material
    else:
        object.data.materials.append(material)
        
        
# ------------------------------------------------------ #
#                  ESZENAREN HASIERAKETA                 #
# ------------------------------------------------------ #

ocean_sim = None
sun_obj_global = None

def main():
    global ocean_sim, sun_obj_global

    scene = bpy.context.scene
    ocean_props = scene.ocean_props
    
    resolution = 256
    domain_size = 100.0
    
    wind_angle_rad = math.radians(ocean_props.wind_direction_angle)
    initial_wind_direction = np.array([math.cos(wind_angle_rad), math.sin(wind_angle_rad)])

    ocean_sim = OceanSimulator(
        resolution=resolution, 
        wind_speed=ocean_props.wind_force,
        wind_direction=initial_wind_direction, 
        domain_size=domain_size,
        phillips_amplitude=ocean_props.wave_height
    )

    ocean_object = create_ocean_mesh(resolution, domain_size)
    add_ocean_material(ocean_object)

    # Skybox HDRIren path-a
    hdri_image_path = "//skyHDRi.exr"

    initial_skybox_rotation_z = 0
    initial_sun_elevation = 45

    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'

    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT' and obj.name != "SceneSunLight":
            bpy.data.objects.remove(obj, do_unlink=True)
        if obj.type == 'MESH' and obj.name != "OceanObject":
            bpy.data.objects.remove(obj, do_unlink=True)

    if os.path.exists(bpy.path.abspath(hdri_image_path)):
        setup_skybox_hdri(hdri_image_path, initial_skybox_rotation_z, initial_sun_elevation)
    else:
        print(f"Abisua: HDRI irudia ez da aurkitu '{hdri_image_path}' helbidean. Skybox-a ez da konfiguratuko.")

    sun_obj_global = setup_sun_light(initial_skybox_rotation_z, initial_sun_elevation)

    register_handlers()

    update_ocean_simulation(None, bpy.context)
    
# ---------------------------------------------------------------------#
#                            KLASEEN KUDEAKETA                         #
# ---------------------------------------------------------------------#

classes = (
    OceanSceneProperties,
    OCEAN_PT_controls,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.ocean_props = bpy.props.PointerProperty(type=OceanSceneProperties)

    main()

def unregister():
    unregister_handlers()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
        
    del bpy.types.Scene.ocean_props

if __name__ == "__main__":
    try:
        unregister()
    except (RuntimeError, AttributeError): pass
    register()