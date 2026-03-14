# Instructions for AI agents

## Game client: To add a setting

See game_client_add_setting/SKILL.md

## Releasing: To bump the version

* Run `ls -d */` in the root of the repository to see all of the crates. Disregard `target`
* Inside each of these, update Cargo.toml to update the version of the crate, and also any perovskite_ dependencies.

Current dependency order:

* Publish core, then server, then game_api, then client. If any additional crates are found other than these, please notify the user to update CLAUDE.local.md.
* For consistency, recommend using `cargo publish -p [crate name]` instead of changing directory then running `cargo publish`

## Adding a new UI control type

See ui_new_control_type/SKILL.md