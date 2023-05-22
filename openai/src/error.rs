// Copyright 2023 Embyr
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Module containing OpenAIError implementation.

use std::error::Error;

use ryst_error::{InternalError, InvalidArgumentError, InvalidStateError};

/// Returned when an error occurs using the SDK.
#[derive(Debug)]
pub enum OpenAIError {
    /// An error which is returned for reasons internal to the function.
    Internal(InternalError),
    /// An error returned when an argument passed to a function does not match the expected format.
    InvalidArgument(InvalidArgumentError),
    /// An error returned when an operation cannot be completed because the state of the underlying
    // struct is inconsistent.
    InvalidState(InvalidStateError),
}

impl Error for OpenAIError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            OpenAIError::Internal(e) => Some(e),
            OpenAIError::InvalidArgument(e) => Some(e),
            OpenAIError::InvalidState(e) => Some(e),
        }
    }
}

impl std::fmt::Display for OpenAIError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            OpenAIError::Internal(e) => e.fmt(f),
            OpenAIError::InvalidArgument(e) => e.fmt(f),
            OpenAIError::InvalidState(e) => e.fmt(f),
        }
    }
}
