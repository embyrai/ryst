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

//! This module contains a set of structs for communicating with OpenAI
//! completions API.

mod request;
mod response;

pub use request::{ChatCompletionRequest, Message};
pub use response::{ChatChoice, ChatCompletionResponse, ChatCompletionResponseStream, ChatUsage};

// The following tests require that OPENAI_API_KEY (optionally OPENAI_API_ORG)
// are set. We are using the "ada" model as this is the cheapest and the tests
// will burn tokens.
#[cfg(feature = "integration")]
#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;

    #[tokio::test]
    // Verify that a simple chat completion submit returns a completion response
    async fn test_chat_completion_submit() {
        let response = ChatCompletionRequest::new(
            "gpt-3.5-turbo",
            &[Message::new("user", "Say this is a test.")],
        )
        .submit()
        .await
        .unwrap();

        assert!(!response.choices.is_empty());
    }

    #[tokio::test]
    // Verify that a simple chat completion stream returns a chat completion response
    async fn test_chat_completion_stream_small() {
        let mut stream = ChatCompletionRequest::new(
            "gpt-3.5-turbo",
            &[Message::new("user", "Say this is a test.")],
        )
        .stream()
        .await
        .unwrap();

        let response_some = stream.next().await.unwrap();
        let response_none = stream.next().await.unwrap();

        assert!(response_some.is_some());
        assert!(response_none.is_none());
    }

    #[tokio::test]
    // Verify that a simple chat completion stream returns a chat completion response
    async fn test_chat_completion_stream_large() {
        let mut stream = ChatCompletionRequest::new(
            "gpt-3.5-turbo",
            &[Message::new(
                "user",
                "Say this is a test in a long winded paragraph.",
            )],
        )
        .with_max_tokens(150)
        .with_n(2)
        .stream()
        .await
        .unwrap();

        let response_some = stream.next().await.unwrap();
        let response_none = stream.next().await.unwrap();

        assert!(response_some.is_some());
        assert!(response_none.is_none());
    }

    #[tokio::test]
    // Verify that a chat complicated completion returns as expected
    async fn test_chat_completion_max_tokens_n() {
        let response = ChatCompletionRequest::new(
            "gpt-3.5-turbo",
            &[Message::new("user", "Say this is a test.")],
        )
        .with_max_tokens(15)
        .with_temperature(0.0)
        .with_n(2)
        .submit()
        .await
        .unwrap();

        assert!(response.choices.len() == 2);
        // max tokens * 2 options
        assert!(response.usage.completion_tokens <= 30);
        assert!(response.choices[0].message.content == response.choices[1].message.content);
    }

    #[tokio::test]
    // Verify that a complicated completion returns as expected
    async fn test_chat_completion_stop_penalty_best_of_user() {
        let response = ChatCompletionRequest::new(
            "gpt-3.5-turbo",
            &[Message::new("user", "Say this is a test.")],
        )
        .with_stop("test")
        .with_max_tokens(15)
        .with_presence_penalty(-2.0)
        .with_frequency_penalty(-2.0)
        .with_user("USER_A")
        .submit()
        .await
        .unwrap();

        assert!(!response.choices.is_empty());
    }

    #[tokio::test]
    // Verify that a complicated completion returns as expected
    async fn test_chat_completion_top_p_logit_bias() {
        // prevents  <|endoftext|> token from being generated
        let bias: HashMap<String, i8> = HashMap::from([("50256".to_string(), -100)]);

        let response = ChatCompletionRequest::new(
            "gpt-3.5-turbo",
            &[Message::new("user", "Say this is a test.")],
        )
        .with_top_p(0.1)
        .with_max_tokens(15)
        .with_logit_bias(&bias)
        .submit()
        .await
        .unwrap();

        assert!(!response.choices.is_empty());
    }
}
