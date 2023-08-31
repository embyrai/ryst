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
use std::env;

use reqwest::Client;
use ryst_error::{InternalError, InvalidArgumentError, InvalidStateError};
use serde::Serialize;

use crate::error::OpenAIError;
use crate::OPEN_AI_URL;

use super::{CompletionResponse, CompletionResponseStream};

/// Builder for creating the completion request and submitting to OpenAI API.
#[derive(Debug, Serialize, PartialEq, Default)]
pub struct CompletionRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    suffix: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<i8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<i8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    echo: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    best_of: Option<i8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logit_bias: Option<HashMap<String, i8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

impl CompletionRequest {
    /// Create a new `CompletionRequest` builder
    ///
    /// Takes a model and prompt, as these are always required.
    pub fn new(model: &str, prompt: &str) -> Self {
        CompletionRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            ..Default::default()
        }
    }

    /// Submit the completion request to the OpenAI url.
    ///
    /// Requires that `OPENAI_API_KEY` environment variable is set. Optionally,
    /// the org will be added if `OPENAI_API_ORG` is set.
    pub async fn submit(self) -> Result<CompletionResponse, OpenAIError> {
        let api_key = env::var("OPENAI_API_KEY").map_err(|_| {
            OpenAIError::InvalidState(InvalidStateError::with_message(
                "OPENAI_API_KEY env variable must be set".to_string(),
            ))
        })?;

        let mut request = Client::new()
            .post(format!("{OPEN_AI_URL}/v1/completions"))
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .json(&self);

        if let Ok(org) = env::var("OPENAI_API_ORG") {
            request = request.header("OpenAI-Organization", org)
        };

        if let Some(stops) = self.stop {
            if stops.len() > 4 {
                return Err(OpenAIError::InvalidArgument(InvalidArgumentError::new(
                    "stop",
                    "You can only provide up to 4 stop sequences",
                )));
            }
        }

        if self.temperature.is_some() && self.top_p.is_some() {
            return Err(OpenAIError::InvalidArgument(InvalidArgumentError::new(
                "temperature",
                "Use temperature or top_p but not both",
            )));
        }

        if self.stream == Some(true) {
            return Err(OpenAIError::InvalidArgument(InvalidArgumentError::new(
                "stream",
                "Use stream() instead of submit",
            )));
        }

        match request.send().await {
            Ok(response) => {
                // Check if the status is a 2XX code.
                let status = response.status();
                if status.is_success() {
                    let result = response.json::<CompletionResponse>().await.map_err(|err| {
                        OpenAIError::InvalidState(InvalidStateError::with_message(err.to_string()))
                    })?;
                    Ok(result)
                } else {
                    let text = response.text().await.map_err(|err| {
                        OpenAIError::InvalidState(InvalidStateError::with_message(err.to_string()))
                    })?;
                    if status.is_client_error() {
                        Err(OpenAIError::InvalidArgument(InvalidArgumentError::new(
                            "request", text,
                        )))
                    } else {
                        Err(OpenAIError::Internal(InternalError::with_message(text)))
                    }
                }
            }
            Err(err) => Err(OpenAIError::Internal(InternalError::from_source(Box::new(
                err,
            )))),
        }
    }

    /// Submit the completion request to the OpenAI url and stream back the response.
    ///
    /// Requires that `OPENAI_API_KEY` environment variable is set. Optionally,
    /// the org will be added if `OPENAI_API_ORG` is set.
    /// Submit the completion request to the OpenAI url.
    ///
    /// Requires that `OPENAI_API_KEY` environment variable is set. Optionally,
    /// the org will be added if `OPENAI_API_ORG` is set.
    pub async fn stream(mut self) -> Result<CompletionResponseStream, OpenAIError> {
        let api_key = env::var("OPENAI_API_KEY").map_err(|_| {
            OpenAIError::InvalidState(InvalidStateError::with_message(
                "OPENAI_API_KEY env variable must be set".to_string(),
            ))
        })?;

        let mut request = Client::new()
            .post(format!("{OPEN_AI_URL}/v1/completions"))
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .json(&self);

        if let Ok(org) = env::var("OPENAI_API_ORG") {
            request = request.header("OpenAI-Organization", org)
        };

        if let Some(stops) = self.stop {
            if stops.len() > 4 {
                return Err(OpenAIError::InvalidArgument(InvalidArgumentError::new(
                    "stop",
                    "You can only provide up to 4 stop sequences",
                )));
            }
        }

        if self.temperature.is_some() && self.top_p.is_some() {
            return Err(OpenAIError::InvalidArgument(InvalidArgumentError::new(
                "temperature",
                "Use temperature or top_p but not both",
            )));
        }

        self.stream = Some(true);

        match request.send().await {
            Ok(response) => {
                // Check if the status is a 2XX code.
                let status = response.status();
                if status.is_success() {
                    Ok(CompletionResponseStream::new(Box::pin(
                        response.bytes_stream(),
                    )))
                } else {
                    let text = response.text().await.map_err(|err| {
                        OpenAIError::InvalidState(InvalidStateError::with_message(err.to_string()))
                    })?;
                    if status.is_client_error() {
                        Err(OpenAIError::InvalidArgument(InvalidArgumentError::new(
                            "request", text,
                        )))
                    } else {
                        Err(OpenAIError::Internal(InternalError::with_message(text)))
                    }
                }
            }
            Err(err) => Err(OpenAIError::Internal(InternalError::from_source(Box::new(
                err,
            )))),
        }
    }

    /// Add a suffix that comes after a completion of inserted text.
    ///
    /// Only works with some models.
    pub fn with_suffix(mut self, suffix: &str) -> Self {
        self.suffix = Some(suffix.to_string());
        self
    }

    /// The maximum number of tokens to generate in the completion.
    pub fn with_max_tokens(mut self, max_tokens: i32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// What sampling temperature to use
    ///
    /// This should not be used at the same time with top_p
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Nucleus sampling value
    ///
    /// Where the model considers the results of the tokens with top_p probability mass.
    /// This should not be used at the same time with temperature
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// How many completions to generate for each prompt.
    pub fn with_n(mut self, n: i8) -> Self {
        self.n = Some(n);
        self
    }

    /// Include the log probabilities on the logprobs most likely tokens, as well the chosen tokens.
    pub fn with_logprobs(mut self, logprobs: i8) -> Self {
        self.logprobs = Some(logprobs);
        self
    }

    /// Echo back the prompt in addition to the completion
    pub fn with_echo(mut self, echo: bool) -> Self {
        self.echo = Some(echo);
        self
    }

    /// The sequence where the API will stop generating further tokens.
    ///
    /// The returned text will not contain the stop sequence. Use of `with_stops` will overwrite this
    /// value.
    pub fn with_stop(mut self, stop: &str) -> Self {
        self.stop = Some(vec![stop.to_string()]);
        self
    }

    /// Up to 4 sequences where the API will stop generating further tokens.
    ///
    /// The returned text will not contain the stop sequence. Use of `with_stop` will overwrite this
    /// value.
    pub fn with_stops(mut self, stop: &[String]) -> Self {
        self.stop = Some(stop.to_vec());
        self
    }

    /// Positive values penalize new tokens based on whether they appear in the text so far.
    ///
    /// This increases the model's likelihood to talk about new topics.
    /// Takes a number between -2.0 and 2.0
    pub fn with_presence_penalty(mut self, presence_penalty: f32) -> Self {
        self.presence_penalty = Some(presence_penalty);
        self
    }

    /// Positive values penalize new tokens based on their existing frequency in the text so far.
    ///
    /// Decreases the model's likelihood to repeat the same line verbatim.
    /// Takes a number between -2.0 and 2.0
    pub fn with_frequency_penalty(mut self, frequency_penalty: f32) -> Self {
        self.frequency_penalty = Some(frequency_penalty);
        self
    }

    /// Generates best_of completions server-side and returns the "best"
    ///
    /// The one with the highest log probability per token. Results cannot be streamed.
    pub fn with_best_of(mut self, best_of: i8) -> Self {
        self.best_of = Some(best_of);
        self
    }

    /// Modify the likelihood of specified tokens appearing in the completion.
    ///
    /// Accepts a json object that maps tokens (specified by their token ID in the GPT tokenizer)
    /// to an associated bias value from -100 to 100.
    ///
    /// As an example, you can pass `{"50256": -100}` to prevent the `<|endoftext|>` token from
    /// being generated.
    pub fn with_logit_bias(mut self, logit_bias: &HashMap<String, i8>) -> Self {
        self.logit_bias = Some(logit_bias.clone());
        self
    }

    /// A unique ID representing your end-user, which can help OpenAI to monitor and detect abuse.
    pub fn with_user(mut self, user: &str) -> Self {
        self.user = Some(user.to_string());
        self
    }
}
