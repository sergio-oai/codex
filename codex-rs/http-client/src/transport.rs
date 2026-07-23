use crate::client::HttpClient;
use crate::client::RequestBuilder;
use crate::error::TransportError;
use crate::request::Request;
use crate::request::RequestBody;
use crate::request::Response;
use bytes::Bytes;
use futures::StreamExt;
use futures::stream::BoxStream;
use http::HeaderMap;
use http::Method;
use http::StatusCode;
use tracing::Level;
use tracing::enabled;
use tracing::trace;

pub type ByteStream = BoxStream<'static, Result<Bytes, TransportError>>;

pub struct StreamResponse {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub bytes: ByteStream,
}

pub trait HttpTransport: Send + Sync {
    fn execute(
        &self,
        req: Request,
    ) -> impl std::future::Future<Output = Result<Response, TransportError>> + Send;
    fn stream(
        &self,
        req: Request,
    ) -> impl std::future::Future<Output = Result<StreamResponse, TransportError>> + Send;
}

#[derive(Clone, Debug)]
pub struct ReqwestTransport {
    client: HttpClient,
    /// Optional bound for diagnostic bodies read after a non-success streaming
    /// response.
    ///
    /// Most streaming call sites preserve the historical unbounded error-body
    /// behavior. Callers that fetch bounded remote metadata can opt into this
    /// limit so a rejected response cannot bypass their success-body cap.
    max_stream_error_body_bytes: Option<usize>,
}

impl ReqwestTransport {
    pub fn new(client: reqwest::Client) -> Self {
        Self {
            client: HttpClient::new(client),
            max_stream_error_body_bytes: None,
        }
    }

    pub fn from_http_client(client: HttpClient) -> Self {
        Self {
            client,
            max_stream_error_body_bytes: None,
        }
    }

    /// Bound diagnostic bodies read from non-success streaming responses.
    ///
    /// This is intentionally opt-in so existing streaming endpoints retain
    /// their current error reporting behavior.
    pub fn with_max_stream_error_body_bytes(mut self, max_bytes: usize) -> Self {
        self.max_stream_error_body_bytes = Some(max_bytes);
        self
    }

    fn build(&self, req: Request) -> Result<RequestBuilder, TransportError> {
        let prepared = req.prepare_body_for_send().map_err(TransportError::Build)?;

        let Request {
            method,
            url,
            headers: _,
            body: _,
            compression: _,
            timeout,
        } = req;

        let mut builder = self.client.request(
            Method::from_bytes(method.as_str().as_bytes()).unwrap_or(Method::GET),
            &url,
        );

        if let Some(timeout) = timeout {
            builder = builder.timeout(timeout);
        }

        builder = builder.headers(prepared.headers);
        if let Some(body) = prepared.body {
            builder = builder.body(body);
        }
        Ok(builder)
    }

    fn map_error(err: reqwest::Error) -> TransportError {
        if err.is_timeout() {
            TransportError::Timeout
        } else {
            TransportError::Network(err.to_string())
        }
    }

    fn trace_request(&self, req: &Request) {
        if self.client.request_logging_enabled() && enabled!(Level::TRACE) {
            trace!(
                "{} to {}: {}",
                req.method,
                req.url,
                request_body_for_trace(req)
            );
        }
    }
}

fn request_body_for_trace(req: &Request) -> String {
    match req.body.as_ref() {
        Some(RequestBody::Json(body)) => body.to_string(),
        Some(RequestBody::EncodedJson(body)) => {
            String::from_utf8_lossy(body.trace_bytes()).into_owned()
        }
        Some(RequestBody::Raw(body)) => format!("<raw body: {} bytes>", body.len()),
        None => String::new(),
    }
}

impl HttpTransport for ReqwestTransport {
    async fn execute(&self, req: Request) -> Result<Response, TransportError> {
        self.trace_request(&req);

        let url = req.url.clone();
        let builder = self.build(req)?;
        let resp = builder.send().await.map_err(Self::map_error)?;
        let status = resp.status();
        let headers = resp.headers().clone();
        let bytes = resp.bytes().await.map_err(Self::map_error)?;
        if !status.is_success() {
            let body = String::from_utf8(bytes.to_vec()).ok();
            return Err(TransportError::Http {
                status,
                url: Some(url),
                headers: Some(headers),
                body,
            });
        }
        Ok(Response {
            status,
            headers,
            body: bytes,
        })
    }

    async fn stream(&self, req: Request) -> Result<StreamResponse, TransportError> {
        self.trace_request(&req);

        let url = req.url.clone();
        let builder = self.build(req)?;
        let resp = builder.send().await.map_err(Self::map_error)?;
        let status = resp.status();
        let headers = resp.headers().clone();
        if !status.is_success() {
            let body = match self.max_stream_error_body_bytes {
                Some(max_bytes) => read_limited_error_body(resp, max_bytes).await,
                None => resp.text().await.ok(),
            };
            return Err(TransportError::Http {
                status,
                url: Some(url),
                headers: Some(headers),
                body,
            });
        }
        let stream = resp
            .bytes_stream()
            .map(|result| result.map_err(Self::map_error));
        Ok(StreamResponse {
            status,
            headers,
            bytes: Box::pin(stream),
        })
    }
}

async fn read_limited_error_body(resp: reqwest::Response, max_bytes: usize) -> Option<String> {
    let mut body = Vec::with_capacity(max_bytes.min(8 * 1024));
    let mut bytes = resp.bytes_stream();
    while body.len() < max_bytes {
        let Some(chunk) = bytes.next().await else {
            break;
        };
        let chunk = chunk.ok()?;
        let remaining = max_bytes - body.len();
        let copy_len = remaining.min(chunk.len());
        body.extend_from_slice(&chunk[..copy_len]);
        if copy_len < chunk.len() {
            break;
        }
    }
    String::from_utf8(body).ok()
}

#[cfg(test)]
#[path = "transport_tests.rs"]
mod tests;
