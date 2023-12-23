// TODO move item stack methods into here so they can be shared with the client
use crate::protocol::items as items_proto;
pub trait ItemStackExt {
    fn stackable(&self) -> bool;
    fn max_stack(&self) -> u32;
}
impl ItemStackExt for crate::protocol::items::ItemStack {
    fn stackable(&self) -> bool {
        matches!(
            self.quantity_type,
            Some(items_proto::item_stack::QuantityType::Stack(_))
        )
    }
    fn max_stack(&self) -> u32 {
        match self.quantity_type {
            Some(items_proto::item_stack::QuantityType::Stack(x)) => x,
            _ => 1,
        }
    }
}
