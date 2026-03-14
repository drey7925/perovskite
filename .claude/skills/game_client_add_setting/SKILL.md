---
name: game_client_add_setting
description: Adds a new setting to the game client. Use this when explicitly asked to add a new setting.
---

To add a setting to the game client:

* Read perovskite_client/src/client_state/settings.rs
* Pick a category from GameSettings (unless adding a new category)
* Add a field to the appropriate struct
* If adding a new type, be sure to include all of the derives seen on existing structs/enums
* Give a sensible default value in the default implementation
* In perovskite_client/src/main_menu.rs, find the corresponding function labeled `draw_...` and add an egui control, consistent with existing controls in the file.
* If the setting is likely needed for the Vulkan pipeline or swapchain (i.e. it affects a pipeline, layout, shader, etc), flag it for human attention by printing a message once you're done making changes.

In most cases, unless special migration code is required to infer a new setting from old settings, it's enough to add the setting to the default implementation; the settings file parser will handle the rest.