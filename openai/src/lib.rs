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

//! SDK for the OpenAI API

extern crate serde;

mod completion;
mod error;

const OPEN_AI_URL: &str = "https://api.openai.com";

pub use completion::{Choice, CompletionRequest, CompletionResponse, Usage};
pub use error::OpenAIError;
