//! Common authentication details for Perovskite. This should be used primarily by the client and server.
//!
//! If you want to implement an API client to access a perovskite server, please reach out - there are better
//! options than OPAQUE + heavy Argon2 parameters for those scenarios (e.g. because high-entropy tokens can be used)
//!
//! Please read PRIMER.md for critical information regarding the project's security posture.

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
                argon2::Algorithm::Argon2id,
                argon2::Version::V0x13,
                argon2::Params::new(MEMORY, T_COST, PARALLELISM, None).unwrap(),
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

// These parameters are relatively large. The rationale is as follows:
// 524288 KiB = 512 MiB. This is a lot of memory, but it's not *that* much for a client that's about to reclaim that memory and begin loading chunks.
// Also, players are unfortunately likely to use weak passwords and reuse passwords; our goal is to not be the party that faciliates a password reuse attack,
// by making brute-force attacks (including offline ones using compromised server databases) prohibitively expensive.
//
// Note that this doesn't affect server costs; this work is done on the client before the OPAQUE protocol runs.
type Opaque4Argon2 = Argon2<524288, 4, 2>;

// Security note: this is comparatively weak, but it's legacy for unmigrated accounts, and no more users will ever register with it.
type LegacyOpaque2Argon2 = Argon2<4096, 3, 1>;
