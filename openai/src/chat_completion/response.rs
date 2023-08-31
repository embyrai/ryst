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

use std::pin::Pin;

use bytes::{Bytes, BytesMut};
use futures::Stream;
use futures::StreamExt;
use reqwest::Result as ReqwestResult;
use ryst_error::{InternalError, InvalidStateError};
use serde::Deserialize;

use crate::error::OpenAIError;

use super::request::Message;

const STREAM_TERMINATION_STRING: &str = "[DONE]";

/// The response returned from a completion request.
#[derive(Debug, Deserialize, PartialEq)]
pub struct ChatCompletionResponse {
    /// Request ID
    pub id: String,
    /// Response type
    pub object: String,
    /// Timestamp of the completion was created
    pub created: i32,
    /// The model the response was created with
    pub model: String,
    /// The list of generated completions
    pub choices: Vec<ChatChoice>,
    /// The tokens used by this response and associated request
    pub usage: ChatUsage,
}

/// The tokens consumed by the completion
#[derive(Debug, Deserialize, PartialEq)]
pub struct ChatUsage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}

/// A generated completion
#[derive(Debug, Deserialize, PartialEq)]
pub struct ChatChoice {
    pub message: Message,
    pub index: i32,
    pub finish_reason: String,
}

/// The response that contains a stream returned from a chat completion request.
pub struct ChatCompletionResponseStream {
    stream: Pin<Box<dyn Stream<Item = ReqwestResult<Bytes>> + Send + 'static>>,
}

impl ChatCompletionResponseStream {
    pub fn new(stream: Pin<Box<dyn Stream<Item = ReqwestResult<Bytes>> + Send + 'static>>) -> Self {
        Self { stream }
    }

    /// Use the stream to get the full response
    pub async fn next(&mut self) -> Result<Option<ChatCompletionResponse>, OpenAIError> {
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
                serde_json::from_slice::<ChatCompletionResponse>(&full_bytes).map_err(|err| {
                    OpenAIError::InvalidState(InvalidStateError::with_message(err.to_string()))
                })?,
            ))
        }
    }
}
