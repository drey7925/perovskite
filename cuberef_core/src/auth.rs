use opaque_ke::CipherSuite;
pub struct CuberefOpaqueAuth;
impl CipherSuite for CuberefOpaqueAuth {
    type OprfCs = opaque_ke::Ristretto255;
    type KeGroup = opaque_ke::Ristretto255;
    type KeyExchange = opaque_ke::key_exchange::tripledh::TripleDh;
    type Ksf = argon2::Argon2<'static>;
}