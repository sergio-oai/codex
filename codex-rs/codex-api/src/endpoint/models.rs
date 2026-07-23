use crate::auth::SharedAuthProvider;
use crate::endpoint::session::EndpointSession;
use crate::error::ApiError;
use crate::provider::Provider;
use codex_client::HttpTransport;
use codex_client::RequestTelemetry;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::openai_models::ModelsResponse;
use futures::StreamExt;
use http::HeaderMap;
use http::Method;
use http::header::ETAG;
use std::sync::Arc;

pub struct ModelsClient<T: HttpTransport> {
    session: EndpointSession<T>,
}

impl<T: HttpTransport> ModelsClient<T> {
    pub fn new(transport: T, provider: Provider, auth: SharedAuthProvider) -> Self {
        Self {
            session: EndpointSession::new(transport, provider, auth),
        }
    }

    pub fn with_telemetry(self, request: Option<Arc<dyn RequestTelemetry>>) -> Self {
        Self {
            session: self.session.with_request_telemetry(request),
        }
    }

    fn path() -> &'static str {
        "models"
    }

    fn append_client_version_query(req: &mut codex_client::Request, client_version: &str) {
        let separator = if req.url.contains('?') { '&' } else { '?' };
        req.url = format!("{}{}client_version={client_version}", req.url, separator);
    }

    pub fn request_url(provider: &Provider, client_version: &str) -> String {
        Self::request_url_for_path(provider, Self::path(), client_version)
    }

    /// Build a model metadata URL for a provider-owned relative path.
    pub fn request_url_for_path(provider: &Provider, path: &str, client_version: &str) -> String {
        let mut request = provider.build_request(Method::GET, path);
        Self::append_client_version_query(&mut request, client_version);
        request.url
    }

    /// Fetch raw model metadata from a provider-owned relative path.
    pub async fn fetch_model_metadata(
        &self,
        path: &str,
        request_url: String,
        extra_headers: HeaderMap,
    ) -> Result<(Vec<u8>, Option<String>), ApiError> {
        let resp = self
            .session
            .execute_with(
                Method::GET,
                path,
                extra_headers,
                /*body*/ None,
                move |req| {
                    req.url.clone_from(&request_url);
                },
            )
            .await?;

        let header_etag = resp
            .headers
            .get(ETAG)
            .and_then(|value| value.to_str().ok())
            .map(ToString::to_string);

        Ok((resp.body.to_vec(), header_etag))
    }

    /// Fetch provider-owned metadata while bounding the response before it is
    /// copied into a contiguous buffer.
    ///
    /// Provider manifests are opt-in but still remote input. Their parser has
    /// its own size cap; this streaming path enforces the same kind of cap at
    /// the transport boundary so an oversized response never reaches parser
    /// allocation.
    pub async fn fetch_model_metadata_limited(
        &self,
        path: &str,
        request_url: String,
        extra_headers: HeaderMap,
        max_bytes: usize,
    ) -> Result<(Vec<u8>, Option<String>), ApiError> {
        let stream_response = self
            .session
            .stream_encoded_json_with(
                Method::GET,
                path,
                extra_headers,
                /*body*/ None,
                move |req| {
                    req.url.clone_from(&request_url);
                },
            )
            .await?;

        let header_etag = stream_response
            .headers
            .get(ETAG)
            .and_then(|value| value.to_str().ok())
            .map(ToString::to_string);
        let mut body = Vec::new();
        let mut bytes = stream_response.bytes;
        while let Some(chunk) = bytes.next().await {
            let chunk = chunk?;
            let next_len = body.len().checked_add(chunk.len()).ok_or_else(|| {
                ApiError::Stream("model metadata response size overflowed".to_string())
            })?;
            if next_len > max_bytes {
                return Err(ApiError::Stream(format!(
                    "model metadata response exceeds maximum size of {max_bytes} bytes"
                )));
            }
            body.extend_from_slice(&chunk);
        }

        Ok((body, header_etag))
    }

    pub async fn list_models(
        &self,
        request_url: String,
        extra_headers: HeaderMap,
    ) -> Result<(Vec<ModelInfo>, Option<String>), ApiError> {
        let (body, header_etag) = self
            .fetch_model_metadata(Self::path(), request_url, extra_headers)
            .await?;
        let ModelsResponse { models } =
            serde_json::from_slice::<ModelsResponse>(&body).map_err(|e| {
                ApiError::Stream(format!(
                    "failed to decode models response: {e}; body: {}",
                    String::from_utf8_lossy(&body)
                ))
            })?;

        Ok((models, header_etag))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::AuthProvider;
    use crate::provider::RetryConfig;
    use bytes::Bytes;
    use codex_client::Request;
    use codex_client::Response;
    use codex_client::StreamResponse;
    use codex_client::TransportError;
    use http::HeaderMap;
    use http::StatusCode;
    use pretty_assertions::assert_eq;
    use serde_json::json;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    #[derive(Clone)]
    struct CapturingTransport {
        last_request: Arc<Mutex<Option<Request>>>,
        body: Arc<ModelsResponse>,
        etag: Option<String>,
    }

    impl Default for CapturingTransport {
        fn default() -> Self {
            Self {
                last_request: Arc::new(Mutex::new(None)),
                body: Arc::new(ModelsResponse { models: Vec::new() }),
                etag: None,
            }
        }
    }

    impl HttpTransport for CapturingTransport {
        async fn execute(&self, req: Request) -> Result<Response, TransportError> {
            *self.last_request.lock().unwrap() = Some(req);
            let body = serde_json::to_vec(&*self.body).unwrap();
            let mut headers = HeaderMap::new();
            if let Some(etag) = &self.etag {
                headers.insert(ETAG, etag.parse().unwrap());
            }
            Ok(Response {
                status: StatusCode::OK,
                headers,
                body: body.into(),
            })
        }

        async fn stream(&self, _req: Request) -> Result<StreamResponse, TransportError> {
            Err(TransportError::Build("stream should not run".to_string()))
        }
    }

    #[derive(Clone)]
    struct StreamingTransport {
        execute_called: Arc<AtomicBool>,
        chunks: Arc<Vec<Bytes>>,
        etag: Option<String>,
    }

    impl HttpTransport for StreamingTransport {
        async fn execute(&self, _req: Request) -> Result<Response, TransportError> {
            self.execute_called.store(true, Ordering::SeqCst);
            Err(TransportError::Build(
                "bounded metadata fetch should stream".to_string(),
            ))
        }

        async fn stream(&self, _req: Request) -> Result<StreamResponse, TransportError> {
            let mut headers = HeaderMap::new();
            if let Some(etag) = &self.etag {
                headers.insert(ETAG, etag.parse().unwrap());
            }
            let chunks = self
                .chunks
                .iter()
                .cloned()
                .map(Ok::<Bytes, TransportError>)
                .collect::<Vec<_>>();
            Ok(StreamResponse {
                status: StatusCode::OK,
                headers,
                bytes: Box::pin(futures::stream::iter(chunks)),
            })
        }
    }

    #[derive(Clone, Default)]
    struct DummyAuth;

    impl AuthProvider for DummyAuth {
        fn add_auth_headers(&self, _headers: &mut HeaderMap) {}
    }

    fn provider(base_url: &str) -> Provider {
        Provider {
            name: "test".to_string(),
            base_url: base_url.to_string(),
            query_params: None,
            headers: HeaderMap::new(),
            retry: RetryConfig {
                max_attempts: 1,
                base_delay: Duration::from_millis(1),
                retry_429: false,
                retry_5xx: true,
                retry_transport: true,
            },
            stream_idle_timeout: Duration::from_secs(1),
        }
    }

    #[tokio::test]
    async fn appends_client_version_query() {
        let response = ModelsResponse { models: Vec::new() };

        let transport = CapturingTransport {
            last_request: Arc::new(Mutex::new(None)),
            body: Arc::new(response),
            etag: None,
        };

        let provider = provider("https://example.com/api/codex");
        let request_url = ModelsClient::<CapturingTransport>::request_url(&provider, "0.99.0");
        let client = ModelsClient::new(transport.clone(), provider, Arc::new(DummyAuth));

        let (models, _) = client
            .list_models(request_url, HeaderMap::new())
            .await
            .expect("request should succeed");

        assert_eq!(models.len(), 0);

        let url = transport
            .last_request
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .url
            .clone();
        assert_eq!(
            url,
            "https://example.com/api/codex/models?client_version=0.99.0"
        );
    }

    #[tokio::test]
    async fn parses_models_response() {
        let response = ModelsResponse {
            models: vec![
                serde_json::from_value(json!({
                    "slug": "gpt-test",
                    "display_name": "gpt-test",
                    "description": "desc",
                    "default_reasoning_level": "medium",
                    "supported_reasoning_levels": [{"effort": "low", "description": "low"}, {"effort": "medium", "description": "medium"}, {"effort": "high", "description": "high"}],
                    "shell_type": "shell_command",
                    "visibility": "list",
                    "minimal_client_version": [0, 99, 0],
                    "supported_in_api": true,
                    "priority": 1,
                    "upgrade": null,
                    "base_instructions": "base instructions",
                    "support_verbosity": false,
                    "default_verbosity": null,
                    "apply_patch_tool_type": null,
                    "truncation_policy": {"mode": "bytes", "limit": 10_000},
                    "supports_parallel_tool_calls": false,
                    "supports_image_detail_original": false,
                    "context_window": 272_000,
                    "experimental_supported_tools": [],
                }))
                .unwrap(),
            ],
        };

        let transport = CapturingTransport {
            last_request: Arc::new(Mutex::new(None)),
            body: Arc::new(response),
            etag: None,
        };

        let provider = provider("https://example.com/api/codex");
        let request_url = ModelsClient::<CapturingTransport>::request_url(&provider, "0.99.0");
        let client = ModelsClient::new(transport, provider, Arc::new(DummyAuth));

        let (models, _) = client
            .list_models(request_url, HeaderMap::new())
            .await
            .expect("request should succeed");

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].slug, "gpt-test");
        assert_eq!(models[0].supported_in_api, true);
        assert_eq!(models[0].priority, 1);
    }

    #[tokio::test]
    async fn list_models_includes_etag() {
        let response = ModelsResponse { models: Vec::new() };

        let transport = CapturingTransport {
            last_request: Arc::new(Mutex::new(None)),
            body: Arc::new(response),
            etag: Some("\"abc\"".to_string()),
        };

        let provider = provider("https://example.com/api/codex");
        let request_url = ModelsClient::<CapturingTransport>::request_url(&provider, "0.1.0");
        let client = ModelsClient::new(transport, provider, Arc::new(DummyAuth));

        let (models, etag) = client
            .list_models(request_url, HeaderMap::new())
            .await
            .expect("request should succeed");

        assert_eq!(models.len(), 0);
        assert_eq!(etag, Some("\"abc\"".to_string()));
    }

    #[tokio::test]
    async fn bounded_metadata_fetch_streams_and_rejects_oversized_bodies() {
        let provider = provider("https://example.com/api/codex");
        let request_url = ModelsClient::<StreamingTransport>::request_url_for_path(
            &provider,
            "codex/provider-manifest",
            "0.1.0",
        );
        let exact_limit_body = br#"{"schema_version":1}"#.to_vec();
        let execute_called = Arc::new(AtomicBool::new(false));
        let client = ModelsClient::new(
            StreamingTransport {
                execute_called: Arc::clone(&execute_called),
                chunks: Arc::new(vec![Bytes::from(exact_limit_body.clone())]),
                etag: Some("\"manifest\"".to_string()),
            },
            provider.clone(),
            Arc::new(DummyAuth),
        );

        let (body, etag) = client
            .fetch_model_metadata_limited(
                "codex/provider-manifest",
                request_url.clone(),
                HeaderMap::new(),
                exact_limit_body.len(),
            )
            .await
            .expect("body at the limit should succeed");

        assert_eq!(body, exact_limit_body);
        assert_eq!(etag, Some("\"manifest\"".to_string()));
        assert!(
            !execute_called.load(Ordering::SeqCst),
            "bounded metadata fetch must not use the buffering execute path"
        );

        let client = ModelsClient::new(
            StreamingTransport {
                execute_called: Arc::new(AtomicBool::new(false)),
                chunks: Arc::new(vec![Bytes::from_static(b"1234"), Bytes::from_static(b"5")]),
                etag: None,
            },
            provider,
            Arc::new(DummyAuth),
        );
        let error = client
            .fetch_model_metadata_limited(
                "codex/provider-manifest",
                request_url,
                HeaderMap::new(),
                4,
            )
            .await
            .expect_err("body above the limit should fail");

        assert!(
            error.to_string().contains("exceeds maximum size"),
            "unexpected error: {error}"
        );
    }
}
