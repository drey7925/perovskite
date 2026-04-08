use generic_array::{ArrayLength, GenericArray};

pub struct LegacyPerovskiteOpaqueAuth;
impl opaque2::CipherSuite for LegacyPerovskiteOpaqueAuth {
    type OprfCs = opaque2::Ristretto255;
    type KeGroup = opaque2::Ristretto255;
    type KeyExchange = opaque2::key_exchange::tripledh::TripleDh;
    type Ksf = LegacyOpaque2Argon2;
}

pub struct PerovskiteOpaqueAuth;
impl opaque4::CipherSuite for PerovskiteOpaqueAuth {
    type OprfCs = opaque4::Ristretto255;
    type KeyExchange =
        opaque4::key_exchange::tripledh::TripleDh<opaque4::Ristretto255, sha2::Sha512>;
    type Ksf = Opaque4Argon2;
}

#[doc(hidden)]
pub struct Argon2<const MEMORY: u32, const T_COST: u32, const PARALLELISM: u32> {
    inner: argon2::Argon2<'static>,
}
impl<const MEMORY: u32, const T_COST: u32, const PARALLELISM: u32> Default
    for Argon2<MEMORY, T_COST, PARALLELISM>
{
    fn default() -> Self {
        Self {
            inner: argon2::Argon2::new(
                argon2::Algorithm::default(),
                argon2::Version::default(),
                argon2::Params::new(4096, 3, 1, None).unwrap(),
            ),
        }
    }
}

impl opaque4::ksf::Ksf for Opaque4Argon2 {
    fn hash<L: ArrayLength<u8>>(
        &self,
        input: GenericArray<u8, L>,
    ) -> Result<GenericArray<u8, L>, opaque4::errors::InternalError> {
        self.inner.hash(input)
    }
}

impl opaque2::ksf::Ksf for LegacyOpaque2Argon2 {
    fn hash<L: ArrayLength<u8>>(
        &self,
        input: GenericArray<u8, L>,
    ) -> Result<GenericArray<u8, L>, opaque2::errors::InternalError> {
        self.inner.hash(input)
    }
}

type LegacyOpaque2Argon2 = Argon2<4096, 3, 1>;
type Opaque4Argon2 = Argon2<19456, 2, 1>;
