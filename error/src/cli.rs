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

use std::fmt;

/// CliError, the return type for main().
///
/// CliError does not implement Error because it is intended to only be used as the return type of
/// main(), in order to implement Debug as required for proper display. Other errors should be
/// converted to CliError within main() itself.
pub struct CliError(String);

impl fmt::Debug for CliError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<E> From<E> for CliError
where
    E: std::error::Error,
{
    fn from(err: E) -> Self {
        // trim_end() is used here because some errors, such as diesel::r2d2 errors can contain a
        // newline
        Self(err.to_string().trim_end().to_string())
    }
}
