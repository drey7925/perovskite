use std::collections::HashSet;

use perovskite_core::protocol::blocks::block_type_def::RenderInfo;
use perovskite_core::protocol::render::TextureReference;
use perovskite_game_api::game_builder::GameBuilder;

fn main() {
    let (mut game, _data_dir) = GameBuilder::testonly_in_memory().unwrap();
    perovskite_game_api::configure_default_game(&mut game).unwrap();
    let mut blocks = game
        .run_task_in_server(|gs| {
            anyhow::Ok(
                gs.block_types()
                    .all_types()
                    .map(|x| x.client_info.clone())
                    .collect::<Vec<_>>(),
            )
        })
        .unwrap();
    blocks.sort_by(|a, b| a.short_name.cmp(&b.short_name));
    for block in blocks {
        println!("{} {}", block.short_name, tex_summary(&block.render_info));
    }
}

fn tex_summary(ri: &Option<RenderInfo>) -> String {
    let prefix = match ri.as_ref() {
        Some(RenderInfo::Empty(_)) => "Empty",
        Some(RenderInfo::Cube(_)) => "Cube",
        Some(RenderInfo::PlantLike(_)) => "PlantLike",
        Some(RenderInfo::AxisAlignedBoxes(_)) => "AxisAlignedBoxes",
        None => "None",
    };

    let tex_heuristic = |tex: &Option<TextureReference>| -> HashSet<&'static str> {
        let mut set = HashSet::new();
        match tex {
            None => {
                set.insert("Missing texture");
            }
            Some(tex) => {
                if tex.diffuse.contains("unknown")
                    || tex.diffuse.contains("todo")
                    || tex.diffuse.is_empty()
                {
                    set.insert("Missing/TODO diffuse texture");
                }
                if tex.diffuse.starts_with("generated:solid_css") {
                    set.insert("Placeholder solid color for diffuse");
                }
                if tex.rt_specular.contains("unknown") || tex.rt_specular.contains("todo") {
                    set.insert("TODO specular texture");
                }
                if tex.emissive.contains("unknown") || tex.emissive.contains("todo") {
                    set.insert("TODO emissive texture");
                }
            }
        };
        set
    };

    let mut all_issues = HashSet::new();
    match ri.as_ref() {
        Some(RenderInfo::Empty(_)) => {}
        Some(RenderInfo::Cube(cube)) => {
            all_issues.extend(tex_heuristic(&cube.tex_right));
            all_issues.extend(tex_heuristic(&cube.tex_left));
            all_issues.extend(tex_heuristic(&cube.tex_top));
            all_issues.extend(tex_heuristic(&cube.tex_bottom));
            all_issues.extend(tex_heuristic(&cube.tex_front));
            all_issues.extend(tex_heuristic(&cube.tex_back));
        }
        Some(RenderInfo::PlantLike(plant)) => {
            all_issues.extend(tex_heuristic(&plant.tex));
        }
        Some(RenderInfo::AxisAlignedBoxes(aabbs)) => {
            for aabb in &aabbs.boxes {
                if aabb.plant_like_tex.is_none() {
                    all_issues.extend(tex_heuristic(&aabb.tex_right));
                    all_issues.extend(tex_heuristic(&aabb.tex_left));
                    all_issues.extend(tex_heuristic(&aabb.tex_top));
                    all_issues.extend(tex_heuristic(&aabb.tex_bottom));
                    all_issues.extend(tex_heuristic(&aabb.tex_front));
                    all_issues.extend(tex_heuristic(&aabb.tex_back));
                } else {
                    all_issues.extend(tex_heuristic(&aabb.plant_like_tex));
                }
            }
        }
        None => {
            all_issues.insert("Missing render info");
        }
    }

    if all_issues.is_empty() {
        all_issues.insert("OK");
    }

    let mut all_issues_vec: Vec<&'static str> = all_issues.into_iter().collect();
    all_issues_vec.sort();

    format!("{}: {}", prefix, all_issues_vec.join(", "))
}
