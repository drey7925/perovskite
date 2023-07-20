use cuberef_server::game_state::{
    items::{Item, ItemManager, ItemStack},
};
use parking_lot::RwLock;

use crate::maybe_export;

use super::basic_blocks::DIRT_WITH_GRASS;

/// Holds crafting/smelting/etc recipes.
///
/// Recipes can be registered before game startup, or at runtime.
/// Once a recipe is registered, it cannot be removed (in the current impl).
///
/// When the recipe book is sorted, ambiguous matches are tiebroken by most
/// specific to least specific (sort key is number of slots expecting an item
/// group rather than specific items). Beyond this sort key, there is no guarantee
/// the sort is stable.
///
/// When a recipe is added, the recipe book is no longer considered sorted until [`sort()`](#method.sort)
/// is called. When the book is not sorted, ambiguous matches are broken arbitrarily.
pub struct RecipeBook<const N: usize, T: Clone> {
    recipes: RwLock<Vec<RecipeImpl<N, T>>>,
}
impl<const N: usize, T: Clone> RecipeBook<N, T> {
    pub(crate) fn new() -> RecipeBook<N, T> {
        RecipeBook {
            recipes: RwLock::new(Vec::new()),
        }
    }
    /// Sorts the recipe book.
    pub fn sort(&self) {
        self.recipes
            .write()
            .sort_unstable_by_key(RecipeImpl::sort_key)
    }

    pub fn find(
        &self,
        items: &ItemManager,
        stacks: &[&Option<ItemStack>],
    ) -> Option<RecipeImpl<N, T>> {
        if stacks.len() != N {
            log::error!("Invalid stacks length passed to Recipes::find()");
            return None;
        }
        let stacks: Vec<Option<&Item>> = stacks
            .iter()
            .map(|x| x.as_ref().and_then(|y| items.get_item(&y.proto.item_name)))
            .collect();

        let stacks = match stacks.try_into() {
            Ok(x) => x,
            Err(_) => {
                log::error!("Conversion from Vec<Option<&Item>> to [Option<&Item>; N] failed; this should not happen");
                return None;
            }
        };

        self.recipes
            .read()
            .iter()
            .find(|x| x.matches(&stacks))
            .map(|x| (*x).clone())
    }

    maybe_export!(
        /// Adds a recipe to this recipe book
        fn register_recipe(&self, recipe: Recipe<N, T>) {
            self.recipes.write().push(recipe);
        }
    );
}

/// Defines what items match a slot in a recipe.
///
/// TODO(future) add a variant that consumes multiple items from a stack
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecipeSlot {
    /// The slot should be empty
    Empty,
    /// The item should belong to the given group
    Group(String),
    /// The item should be exactly the indicated item
    Exact(String),
}
impl RecipeSlot {
    /// How many items to take from the input. At the moment, this is always 1, since there is not yet a variant
    /// that consumes multiple items from a stack
    pub fn quantity(&self) -> u32 {
        1
    }
}

maybe_export!(use self::RecipeImpl as Recipe);

#[derive(Debug, Clone)]
pub struct RecipeImpl<const N: usize, T> {
    /// The N slots that need to be filled
    pub slots: [RecipeSlot; N],
    /// The result obtained from this crafting recipe
    pub result: ItemStack,
    /// If true, the inputs may be given in any order
    pub shapeless: bool,
    /// Any metadata for this recipe (e.g. time, non-inventory resources, etc)
    pub metadata: T,
}
impl<const N: usize, T> RecipeImpl<N, T> {
    fn sort_key(&self) -> usize {
        self.slots
            .iter()
            .filter(|&x| matches!(x, RecipeSlot::Group(_)))
            .count()
    }
    pub(crate) fn matches(&self, stacks: &[Option<&Item>; N]) -> bool {
        if self.shapeless {
            todo!("shapeless solver not written");
        } else {
            for (j, stack) in stacks.iter().enumerate() {
                match &self.slots[j] {
                    RecipeSlot::Empty => {
                        if stack.is_some() {
                            return false;
                        }
                    }
                    RecipeSlot::Group(group) => {
                        if !stack.is_some_and(|x| x.proto.groups.contains(group)) {
                            return false;
                        }
                    }
                    RecipeSlot::Exact(item) => {
                        if !stack.is_some_and(|x| &x.proto.short_name == item) {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }
}

pub(crate) fn register_test_recipes(game_builder: &mut super::DefaultGameBuilder) {
    use RecipeSlot::*;
    // testonly
    game_builder.register_crafting_recipe(
        [
            Group("testonly_wet".to_string()),
            Exact("default:dirt".to_string()),
            Empty,
            Empty,
            Empty,
            Empty,
            Empty,
            Empty,
            Empty,
        ],
        DIRT_WITH_GRASS.0.to_string(),
        1,
        true,
    );
    game_builder.register_crafting_recipe(
        [
            Group("testonly_wet".to_string()),
            Group("testonly_wet".to_string()),
            Exact("default:dirt".to_string()),
            Exact("default:dirt".to_string()),
            Empty,
            Empty,
            Empty,
            Empty,
            Empty,
        ],
        DIRT_WITH_GRASS.0.to_string(),
        2,
        true,
    );
}
