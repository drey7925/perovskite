use dyn_clone::DynClone;
use std::any::Any;

pub trait CustomDataContents: Any + Send + Sync + DynClone + 'static {}
dyn_clone::clone_trait_object!(CustomDataContents);

impl<T: Any + Send + Sync + Clone + 'static> CustomDataContents for T {}
pub type CustomData = Box<dyn CustomDataContents>;
impl dyn CustomDataContents {
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        (self as &dyn Any).downcast_ref()
    }
    pub fn downcast_mut<T: Any>(&mut self) -> Option<&mut T> {
        (self as &mut dyn Any).downcast_mut()
    }
}
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
