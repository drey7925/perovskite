use dyn_clone::DynClone;
use std::{any::Any, fmt::Debug};

/// Marker trait for types that can be stored as custom extended data on a block.
/// Automatically implemented for any `T: Any + Send + Sync + Clone + DebugAsString + 'static`.
pub trait CustomDataContents: Any + Send + Sync + DynClone + DebugAsString + 'static {}
dyn_clone::clone_trait_object!(CustomDataContents);

impl<T: Any + Send + Sync + Clone + DebugAsString + 'static> CustomDataContents for T {}
/// A type-erased, cloneable custom data value that can be attached to a block.
/// Use [`CustomDataDowncast`] to recover the concrete type.
pub type CustomData = Box<dyn CustomDataContents>;
impl dyn CustomDataContents {
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        (self as &dyn Any).downcast_ref()
    }
    pub fn downcast_mut<T: Any>(&mut self) -> Option<&mut T> {
        (self as &mut dyn Any).downcast_mut()
    }
}
/// Downcasting helpers for [`CustomData`] (`Box<dyn CustomDataContents>`).
pub trait CustomDataDowncast {
    fn downcast_ref<T: Any>(&self) -> Option<&T>;
    fn downcast_mut<T: Any>(&mut self) -> Option<&mut T>;
    fn downcast_box<T: Any>(self) -> Option<Box<T>>;
}

impl CustomDataDowncast for Box<dyn CustomDataContents> {
    fn downcast_ref<T: Any>(&self) -> Option<&T> {
        (self.as_ref() as &dyn Any).downcast_ref()
    }
    fn downcast_mut<T: Any>(&mut self) -> Option<&mut T> {
        (self.as_mut() as &mut dyn Any).downcast_mut()
    }
    fn downcast_box<T: Any>(self) -> Option<Box<T>> {
        Box::<dyn Any>::downcast::<T>(self).ok()
    }
}

pub trait DebugAsString {
    fn debug_as_string(&self) -> String {
        format!("{}{{ ... }}", core::any::type_name::<Self>())
    }
}
impl<T: Debug> DebugAsString for T {
    fn debug_as_string(&self) -> String {
        format!("{:?}", self)
    }
}
