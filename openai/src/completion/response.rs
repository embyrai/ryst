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

use std::collections::HashMap;
use std::fmt;
use std::pin::Pin;

use bytes::{Bytes, BytesMut};
use futures::Stream;
use futures::StreamExt;
use reqwest::Result as ReqwestResult;
use ryst_error::{InternalError, InvalidStateError};
use serde::de::{Deserializer, Visitor};
use serde::Deserialize;

use crate::error::OpenAIError;

const STREAM_TERMINATION_STRING: &str = "[DONE]";

/// The response returned from a completion request.
#[derive(Debug, Deserialize, PartialEq)]
pub struct CompletionResponse {
    /// Request ID
    pub id: String,
    /// Response type
    pub object: String,
    /// Timestamp of the completion was created
    pub created: i32,
    /// The model the response was created with
    pub model: String,
    /// The list of generated completions
    pub choices: Vec<CompletionChoice>,
    /// The tokens used by this response and associated request
    pub usage: CompletionUsage,
}

/// The tokens consumed by the completion
#[derive(Debug, Deserialize, PartialEq)]
pub struct CompletionUsage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}

/// A generated completion
#[derive(Debug, Deserialize, PartialEq)]
pub struct CompletionChoice {
    pub text: String,
    pub index: i32,
    pub logprobs: Option<Logprobs>,
    pub finish_reason: String,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct Logprobs {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<f32>,
    #[serde(deserialize_with = "flatten_log_probs")]
    pub top_logprobs: HashMap<String, f32>,
    pub text_offset: Vec<i32>,
}

fn flatten_log_probs<'de, D>(deserializer: D) -> Result<HashMap<String, f32>, D::Error>
where
    D: Deserializer<'de>,
{
    struct LogProbsVisitor;

    impl<'de> Visitor<'de> for LogProbsVisitor {
        type Value = HashMap<String, f32>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a sequence of maps")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::SeqAccess<'de>,
        {
            let mut result = HashMap::new();

            while let Some(map) = seq.next_element::<HashMap<String, f32>>()? {
                for (key, value) in map {
                    result.insert(key, value);
                }
            }

            Ok(result)
        }
    }

    deserializer.deserialize_seq(LogProbsVisitor)
}

/// The response that contains a stream returned from a completion request.
pub struct CompletionResponseStream {
    stream: Pin<Box<dyn Stream<Item = ReqwestResult<Bytes>> + Send + 'static>>,
}

impl CompletionResponseStream {
    pub fn new(stream: Pin<Box<dyn Stream<Item = ReqwestResult<Bytes>> + Send + 'static>>) -> Self {
        Self { stream }
    }

    /// Use the stream to get the full response
    pub async fn next(&mut self) -> Result<Option<CompletionResponse>, OpenAIError> {
        let mut full_bytes = BytesMut::new();
        while let Some(value) = self.stream.next().await {
            match value {
                Ok(bytes) => {
                    if bytes != STREAM_TERMINATION_STRING.as_bytes() {
                        full_bytes.extend_from_slice(&bytes)
                    }
                }
                Err(err) => {
                    return Err(OpenAIError::Internal(InternalError::from_source(Box::new(
                        err,
                    ))))
                }
            }
        }

        if full_bytes.is_empty() {
            Ok(None)
        } else {
            Ok(Some(
                serde_json::from_slice::<CompletionResponse>(&full_bytes).map_err(|err| {
                    OpenAIError::InvalidState(InvalidStateError::with_message(err.to_string()))
                })?,
            ))
        }
    }
}
