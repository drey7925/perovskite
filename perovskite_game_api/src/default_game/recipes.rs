use parking_lot::RwLock;
use perovskite_server::game_state::items::{Item, ItemManager, ItemStack};

use crate::maybe_export;

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
    // TODO: This uses a RwLock because we need to be able to fill and sort it.
    // However, it shouldn't be edited after game startup.
    // Can we create some kind of mutex that becomes read-only with no contention/ping-pong
    // once the game starts up?
    // (something like `oncemutex` but allowing one to lock and unlock multiple times before explicitly
    // transitioning to the "very fast concurrent reads after the first lock is over" state???)
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
            return self.matches_shapeless(stacks);
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

    fn matches_shapeless(&self, stacks: &[Option<&Item>; N]) -> bool {
        // Ugh, this is basically a boolean satisfiability problem.

        // small checks first
        // Count non-empty slots and non-empty stacks. If mismatched, false.
        if stacks.iter().flatten().count()
            != self
                .slots
                .iter()
                .filter(|x| **x != RecipeSlot::Empty)
                .count()
        {
            return false;
        }

        let mut remaining_stacks: smallvec::SmallVec<[_; 16]> =
            stacks.iter().filter_map(|x| *x).collect();

        // First, try to match exact matches and remove them from the list
        for slot in &self.slots {
            if let RecipeSlot::Exact(item) = slot {
                if let Some(index) = remaining_stacks
                    .iter()
                    .position(|x| &x.proto.short_name == item)
                {
                    remaining_stacks.swap_remove(index);
                } else {
                    return false;
                }
            }
        }

        let remaining_slots: smallvec::SmallVec<[_; 16]> = self
            .slots
            .iter()
            .filter_map(|x| match x {
                RecipeSlot::Empty => None,
                RecipeSlot::Group(group) => Some(group),
                RecipeSlot::Exact(_) => None,
            })
            .collect();
        assert_eq!(remaining_slots.len(), remaining_stacks.len());

        match remaining_slots.len() {
            0 => true,
            1 => {
                let slot = remaining_slots[0];
                remaining_stacks[0].proto.groups.contains(slot)
            }
            2 => {
                let slot1 = remaining_slots[0];
                let slot2 = remaining_slots[1];
                (remaining_stacks[0].proto.groups.contains(slot1)
                    && remaining_stacks[1].proto.groups.contains(slot2))
                    || (remaining_stacks[0].proto.groups.contains(slot2)
                        && remaining_stacks[1].proto.groups.contains(slot1))
            }
            _ => {
                tracing::error!(
                    "Too many shapeless slots to match to groups, todo implement graph matching here"
                );
                false
            }
        }
    }
}

mod test {
    #[test]
    fn test_shapeless() {
        use perovskite_core::protocol;
        use perovskite_server::game_state::items::Item;

        use crate::default_game::recipes::RecipeImpl;

        use crate::default_game::recipes::RecipeSlot;
        use perovskite_server::game_state::items::ItemStack;
        let item_1 = Item::default_with_proto(protocol::items::ItemDef {
            short_name: "a".into(),
            groups: vec!["a".into()],
            ..Default::default()
        });
        let item_2 = Item::default_with_proto(protocol::items::ItemDef {
            short_name: "b".into(),
            groups: vec!["b".into()],
            ..Default::default()
        });

        let recipe = RecipeImpl {
            slots: [
                RecipeSlot::Group("a".into()),
                RecipeSlot::Exact("b".into()),
                RecipeSlot::Empty,
                RecipeSlot::Empty,
            ],
            result: ItemStack {
                proto: protocol::items::ItemStack {
                    ..Default::default()
                },
            },
            shapeless: true,
            metadata: (),
        };

        assert!(!recipe.matches(&[None, None, None, None]));
        assert!(!recipe.matches(&[Some(&item_1), None, None, None]));
        assert!(!recipe.matches(&[Some(&item_1), Some(&item_1), Some(&item_2), None]));

        assert!(recipe.matches(&[Some(&item_1), Some(&item_2), None, None]));
        assert!(recipe.matches(&[None, Some(&item_2), None, Some(&item_1)]));

        let recipe2 = RecipeImpl {
            slots: [
                RecipeSlot::Group("a".into()),
                RecipeSlot::Group("b".into()),
                RecipeSlot::Empty,
                RecipeSlot::Empty,
            ],
            result: ItemStack {
                proto: protocol::items::ItemStack {
                    ..Default::default()
                },
            },
            shapeless: true,
            metadata: (),
        };

        assert!(!recipe.matches(&[None, None, None, None]));
        assert!(!recipe.matches(&[Some(&item_1), None, None, None]));
        assert!(!recipe.matches(&[Some(&item_1), Some(&item_1), Some(&item_2), None]));

        assert!(recipe.matches(&[Some(&item_1), Some(&item_2), None, None]));
        assert!(recipe.matches(&[None, Some(&item_2), None, Some(&item_1)]));

        assert!(recipe2.matches(&[Some(&item_1), Some(&item_2), None, None]));
        assert!(recipe2.matches(&[None, Some(&item_2), None, Some(&item_1)]));
    }
}
