use anyhow::Result;
use perovskite_core::protocol::render::{CustomMesh, TextureReference};

pub fn load_obj_mesh(obj_data: &[u8], texture: impl Into<String>) -> Result<CustomMesh> {
    let object: obj::Obj<obj::TexturedVertex, u32> = obj::load_obj(obj_data)?;
    let mut mesh = CustomMesh::default();
    mesh.texture = Some(TextureReference {
        diffuse: texture.into(),
        rt_specular: String::new(),
        emissive: String::new(),
        crop: None,
    });
    for vertex in object.vertices {
        mesh.x.push(vertex.position[2]);
        // We need vulkan's coordinate system, with Y going down
        mesh.y.push(-vertex.position[1]);
        mesh.z.push(vertex.position[0]);
        mesh.u.push(vertex.texture[0]);
        // OBJ and vulkan have opposite UV Y coordinates
        mesh.v.push(1.0 - vertex.texture[1]);
        mesh.nx.push(vertex.normal[0]);
        mesh.ny.push(vertex.normal[1]);
        mesh.nz.push(vertex.normal[2]);
    }
    mesh.indices = object.indices;
    Ok(mesh)
}
