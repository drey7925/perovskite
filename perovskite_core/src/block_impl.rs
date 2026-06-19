use crate::protocol::{
    blocks::{
        block_type_def::{PhysicsInfo, RenderInfo},
        AxisAlignedBoxRotation, BlockTypeDef, CubeVariantEffect,
    },
    render::TextureReference,
};

impl BlockTypeDef {
    /// Indicates which variant bits need to change for this block to be worth sending to
    /// clients. This is intended as an optimization, and is not yet fully implemented.
    ///
    /// This will return all of the bits that affect the visual appearance, rendering, or client physics
    /// in a way that is visible to the client.
    ///
    /// Note that simply returning all-ones is always valid (just misses the optimization opportunity)
    ///
    /// In the future, there may be broadcasts used for listners _other than_ clients, e.g. state replication
    /// and similar. This mask does _not_ apply to them, it only applies to clients.
    pub fn client_update_mask(&self) -> u16 {
        fn check_tex(tr: &Option<TextureReference>) -> u16 {
            let Some(tx) = tr else {
                return 0;
            };
            let crop_val = match tx.crop {
                Some(c) => {
                    if let Some(d) = c.dynamic {
                        d.x_selector_bits as u16 | d.y_selector_bits as u16
                    } else {
                        0
                    }
                }
                None => 0,
            };
            // so far just crop_val, but structure for flexibility
            crop_val
        }

        // TODO check the render details
        let render_mask = match &self.render_info {
            Some(RenderInfo::Cube(c)) => {
                let var = match c.variant_effect() {
                    CubeVariantEffect::None => 0,
                    CubeVariantEffect::RotateNesw => 3,
                    CubeVariantEffect::Liquid => 15,
                    CubeVariantEffect::CubeVariantHeight => 15,
                };
                var | check_tex(&c.tex_left)
                    | check_tex(&c.tex_right)
                    | check_tex(&c.tex_top)
                    | check_tex(&c.tex_bottom)
                    | check_tex(&c.tex_front)
                    | check_tex(&c.tex_back)
            }
            Some(RenderInfo::PlantLike(pl)) => check_tex(&pl.tex),
            Some(RenderInfo::AxisAlignedBoxes(aabbs)) => {
                let mut acc = 0;
                for aabb in &aabbs.boxes {
                    acc |= aabb.variant_mask as u16;
                    if aabb.rotation() == AxisAlignedBoxRotation::Nesw {
                        acc |= 3;
                    }
                    acc |= check_tex(&aabb.tex_left);
                    acc |= check_tex(&aabb.tex_right);
                    acc |= check_tex(&aabb.tex_top);
                    acc |= check_tex(&aabb.tex_bottom);
                    acc |= check_tex(&aabb.tex_front);
                    acc |= check_tex(&aabb.tex_back);
                    acc |= check_tex(&aabb.plant_like_tex);
                }
                acc
            }
            _ => 0,
        };
        let physics_mask = match &self.physics_info {
            // TBD whether this will truly affect physics later, but it definitely affects rendering so ok to be overly safe here
            Some(PhysicsInfo::Fluid(_)) => 15,
            Some(PhysicsInfo::SolidCustomCollisionboxes(aabbs)) => {
                let mut acc = 0;
                for aabb in &aabbs.boxes {
                    acc |= aabb.variant_mask as u16;
                    if aabb.rotation() == AxisAlignedBoxRotation::Nesw {
                        acc |= 3;
                    }
                }
                acc
            }
            _ => 0,
        };
        let hit_mask = self
            .tool_custom_hitbox
            .as_ref()
            .map(|x| {
                let mut acc = 0;
                for aabb in &x.boxes {
                    acc |= aabb.variant_mask as u16;
                    if aabb.rotation() == AxisAlignedBoxRotation::Nesw {
                        acc |= 3;
                    }
                }
                acc
            })
            .unwrap_or(0);
        render_mask | physics_mask | hit_mask
    }
}
