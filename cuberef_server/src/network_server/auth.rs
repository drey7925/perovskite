// Copyright 2023 drey7925
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use std::net::SocketAddr;

pub enum AuthOutcome {
    /// Successful, here's the token
    Success(String),
    WrongPassword,
    NoSuchAccount,
}

pub enum RegisterOutcome {
    /// Successful, here's a token to use
    Success(String),
    AccountExists,
    RegistrationForbidden,
}

pub enum TokenOutcome {
    Success,
    Failure,
}

/// Service for authenticating users and their tokens.
/// Users have long-lived usernames and passwords, and short-lived tokens that are valid
/// while a cuberef server is running.
///
/// For security, we will use a PAKE algorithm like docs.rs/opaque-ke
/// THIS IS NOT YET IMPLEMENTED
/// THE API HERE WILL CHANGE WHEN IT IS
pub trait AuthService: Send + Sync {
    // Takes uername, password hash, and remote IP. Adds an account to the database, or returs an error if forbidden/impossible
    fn create_account(
        &self,
        username: &str,
        password_hash: &[u8],
        remote_addr: Option<SocketAddr>,
    ) -> Result<RegisterOutcome>;
    // Takes username, password hash, and remote IP. Tries to auth, and returns an AuthOutcome
    fn authenticate_user(
        &self,
        username: &str,
        password_hash: &[u8],
        remote_addr: Option<SocketAddr>,
    ) -> Result<AuthOutcome>;
    // Checks if a token came from this authservice. If it did, and the remote addr also matches, return the username for this user
    // TODO change return type?
    fn check_token(
        &self,
        username: &str,
        token: &str,
        remote_addr: Option<SocketAddr>,
    ) -> Result<TokenOutcome>;
}
