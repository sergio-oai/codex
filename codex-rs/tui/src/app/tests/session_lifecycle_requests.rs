use super::*;
use crate::app::session_lifecycle::ThreadAttachPresentation;
use app_test_support::create_fake_parented_rollout_with_source;
use app_test_support::create_fake_rollout;
use app_test_support::rollout_path;
use codex_app_server_protocol::ClientNotification;
use codex_app_server_protocol::ClientRequest;
use codex_app_server_protocol::JSONRPCError;
use codex_app_server_protocol::JSONRPCErrorError;
use codex_app_server_protocol::JSONRPCMessage;
use codex_app_server_protocol::JSONRPCResponse;
use codex_protocol::AgentPath;
use futures::SinkExt;
use futures::StreamExt;
use pretty_assertions::assert_eq;
use std::sync::Mutex;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;

#[derive(Clone, Copy)]
enum ThreadScopedModelListBehavior {
    Forward,
    Fail,
}

/// Returns and resets `(thread/loaded/list, thread/read)` request counts.
fn take_backfill_counts(requests: &Arc<Mutex<Vec<String>>>) -> (usize, usize) {
    let requests = std::mem::take(&mut *requests.lock().expect("request recorder lock"));
    (
        requests
            .iter()
            .filter(|method| *method == "thread/loaded/list")
            .count(),
        requests
            .iter()
            .filter(|method| *method == "thread/read")
            .count(),
    )
}

/// Starts an embedded app server behind a loopback WebSocket proxy that records JSON-RPC methods.
async fn start_recording_app_server(
    config: &Config,
    thread_scoped_model_list_behavior: ThreadScopedModelListBehavior,
) -> Result<(
    AppServerSession,
    Arc<Mutex<Vec<String>>>,
    Arc<Mutex<Vec<String>>>,
    JoinHandle<Result<()>>,
)> {
    let state_db =
        crate::init_state_db_for_app_server_target(config, &crate::AppServerTarget::Embedded)
            .await?;
    let embedded = crate::start_embedded_app_server(
        codex_arg0::Arg0DispatchPaths::default(),
        config.clone(),
        Vec::new(),
        codex_config::LoaderOverrides::default(),
        /*strict_config*/ false,
        codex_config::CloudConfigBundleLoader::default(),
        codex_feedback::CodexFeedback::new(),
        /*log_db*/ None,
        state_db,
        Arc::new(codex_exec_server::EnvironmentManager::default_for_tests()),
    )
    .await?;
    let codex_home = config.codex_home.display().to_string();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let request_sink = Arc::clone(&requests);
    let unsubscribed_thread_ids = Arc::new(Mutex::new(Vec::new()));
    let unsubscribe_sink = Arc::clone(&unsubscribed_thread_ids);
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let websocket_url = format!("ws://{}", listener.local_addr()?);
    let proxy = tokio::spawn(async move {
        let (stream, _) = listener.accept().await?;
        let mut websocket = accept_async(stream).await?;
        while let Some(frame) = websocket.next().await {
            let Message::Text(text) = frame? else {
                continue;
            };
            let message = serde_json::from_str::<JSONRPCMessage>(&text)?;
            match message {
                JSONRPCMessage::Request(request) if request.method == "initialize" => {
                    websocket
                        .send(Message::Text(
                            serde_json::to_string(&JSONRPCMessage::Response(JSONRPCResponse {
                                id: request.id,
                                result: serde_json::json!({
                                    "userAgent": "codex-tui-test",
                                    "codexHome": codex_home,
                                }),
                            }))?
                            .into(),
                        ))
                        .await?;
                }
                JSONRPCMessage::Request(request) => {
                    request_sink
                        .lock()
                        .expect("request recorder lock")
                        .push(request.method.clone());
                    if request.method == "thread/unsubscribe"
                        && let Some(thread_id) = request
                            .params
                            .as_ref()
                            .and_then(|params| params.get("threadId"))
                            .and_then(serde_json::Value::as_str)
                    {
                        unsubscribe_sink
                            .lock()
                            .expect("unsubscribe recorder lock")
                            .push(thread_id.to_string());
                    }
                    let request_id = request.id.clone();
                    let should_fail_thread_scoped_model_list = matches!(
                        thread_scoped_model_list_behavior,
                        ThreadScopedModelListBehavior::Fail
                    ) && request.method == "model/list"
                        && request
                            .params
                            .as_ref()
                            .and_then(|params| params.get("threadId"))
                            .is_some_and(|thread_id| !thread_id.is_null());
                    let response = if should_fail_thread_scoped_model_list {
                        JSONRPCMessage::Error(JSONRPCError {
                            id: request_id,
                            error: JSONRPCErrorError {
                                code: -32603,
                                message: "forced thread-scoped model/list failure".to_string(),
                                data: None,
                            },
                        })
                    } else {
                        let request = serde_json::from_value::<ClientRequest>(
                            serde_json::to_value(request)?,
                        )?;
                        match embedded.request(request).await? {
                            Ok(result) => JSONRPCMessage::Response(JSONRPCResponse {
                                id: request_id,
                                result,
                            }),
                            Err(error) => JSONRPCMessage::Error(JSONRPCError {
                                id: request_id,
                                error,
                            }),
                        }
                    };
                    websocket
                        .send(Message::Text(serde_json::to_string(&response)?.into()))
                        .await?;
                }
                JSONRPCMessage::Notification(notification)
                    if notification.method == "initialized" => {}
                JSONRPCMessage::Notification(notification) => {
                    embedded
                        .notify(serde_json::from_value::<ClientNotification>(
                            serde_json::to_value(notification)?,
                        )?)
                        .await?;
                }
                JSONRPCMessage::Response(_) | JSONRPCMessage::Error(_) => {}
            }
        }
        embedded.shutdown().await?;
        Ok(())
    });
    let app_server = crate::connect_remote_app_server(crate::RemoteAppServerEndpoint::WebSocket {
        websocket_url,
        auth_token: None,
    })
    .await?;

    Ok((
        AppServerSession::new(
            app_server,
            crate::app_server_session::ThreadParamsMode::Embedded,
        ),
        requests,
        unsubscribed_thread_ids,
        proxy,
    ))
}

#[test]
fn fresh_session_applies_requested_name() -> Result<()> {
    const TEST_STACK_SIZE_BYTES: usize = 8 * 1024 * 1024;

    std::thread::Builder::new()
        .name("tui-named-fresh-session".to_string())
        .stack_size(TEST_STACK_SIZE_BYTES)
        .spawn(|| {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            runtime.block_on(async {
                let mut app = make_test_app().await;
                let codex_home = tempdir()?;
                app.config.codex_home = codex_home.path().to_path_buf().abs();
                app.config.sqlite_home = codex_home.path().to_path_buf();
                let (mut app_server, requests, _unsubscribed_thread_ids, proxy) =
                    start_recording_app_server(&app.config, ThreadScopedModelListBehavior::Forward)
                        .await?;
                let mut tui = crate::tui::test_support::make_test_tui()?;

                app.start_fresh_session_with_summary_hint(
                    &mut tui,
                    &mut app_server,
                    /*session_start_source*/ None,
                    /*initial_user_message*/ None,
                    /*new_thread_name*/ Some("Add User".to_string()),
                )
                .await;

                let thread_id = app
                    .chat_widget
                    .thread_id()
                    .expect("fresh session should have a thread id");
                assert_eq!(app.chat_widget.thread_name(), Some("Add User".to_string()));
                assert!(
                    requests
                        .lock()
                        .expect("request recorder lock")
                        .iter()
                        .any(|method| method == "thread/name/set"),
                    "fresh session should be named through the app server"
                );
                let thread = app_server
                    .thread_read(thread_id, /*include_turns*/ false)
                    .await?;
                assert_eq!(thread.name.as_deref(), Some("Add User"));

                app_server.shutdown().await?;
                proxy.await??;
                Ok(())
            })
        })?
        .join()
        .expect("named fresh session test thread")
}

#[test]
fn session_lifecycle_avoids_redundant_subagent_metadata_reads() -> Result<()> {
    const TEST_STACK_SIZE_BYTES: usize = 8 * 1024 * 1024;

    std::thread::Builder::new()
        .name("tui-session-lifecycle-requests".to_string())
        .stack_size(TEST_STACK_SIZE_BYTES)
        .spawn(|| {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            runtime.block_on(async {
                let mut app = make_test_app().await;
                let codex_home = tempdir()?;
                app.config.codex_home = codex_home.path().to_path_buf().abs();
                app.config.sqlite_home = codex_home.path().to_path_buf();
                let root_timestamp = "2026-01-01T00-00-00";
                let root_thread_id = ThreadId::from_string(
                    &create_fake_rollout(
                        codex_home.path(),
                        root_timestamp,
                        "2026-01-01T00:00:00Z",
                        "Saved user message",
                        Some(app.config.model_provider_id.as_str()),
                        /*git_info*/ None,
                    )
                    .expect("create root rollout"),
                )?;
                let child_thread_id = ThreadId::from_string(
                    &create_fake_parented_rollout_with_source(
                        codex_home.path(),
                        "2026-01-01T00-00-01",
                        "2026-01-01T00:00:01Z",
                        "Saved child message",
                        Some(app.config.model_provider_id.as_str()),
                        /*git_info*/ None,
                        RolloutSessionSource::SubAgent(SubAgentSource::ThreadSpawn {
                            parent_thread_id: root_thread_id,
                            depth: 1,
                            agent_path: Some(
                                AgentPath::try_from("/root/worker").expect("valid agent path"),
                            ),
                            agent_nickname: Some("worker".to_string()),
                            agent_role: Some("worker".to_string()),
                        }),
                        root_thread_id.into(),
                        root_thread_id,
                    )
                    .expect("create child rollout"),
                )?;
                let root_rollout_path = rollout_path(
                    codex_home.path(),
                    root_timestamp,
                    &root_thread_id.to_string(),
                );
                let (mut app_server, requests, _unsubscribed_thread_ids, proxy) =
                    start_recording_app_server(&app.config, ThreadScopedModelListBehavior::Forward)
                        .await?;
                let root = app_server
                    .resume_thread(
                        app.config.clone(),
                        root_thread_id,
                        app.resume_model_settings(),
                    )
                    .await?;
                app.enqueue_primary_thread_session(root.session, root.turns)
                    .await?;
                app_server
                    .resume_thread(
                        app.config.clone(),
                        child_thread_id,
                        app.resume_model_settings(),
                    )
                    .await?;
                let mut tui = crate::tui::test_support::make_test_tui()?;
                take_backfill_counts(&requests);

                let control = Box::pin(app.handle_event(
                    &mut tui,
                    &mut app_server,
                    AppEvent::ForkCurrentSession,
                ))
                .await?;

                assert!(matches!(control, AppRunControl::Continue));
                assert_ne!(app.chat_widget.thread_id(), Some(root_thread_id));
                // Forking may read the source metadata once when the response includes its parent
                // id. It must not scan or backfill loaded threads for the newly created fork.
                assert!(matches!(take_backfill_counts(&requests), (0, 0) | (0, 1)));

                app.start_fresh_session_with_summary_hint(
                    &mut tui,
                    &mut app_server,
                    /*session_start_source*/ None,
                    /*initial_user_message*/ None,
                    /*new_thread_name*/ None,
                )
                .await;

                assert_ne!(app.chat_widget.thread_id(), Some(root_thread_id));
                assert_eq!(take_backfill_counts(&requests), (0, 0));

                let loaded_threads = app_server
                    .thread_loaded_list(ThreadLoadedListParams {
                        cursor: None,
                        limit: None,
                    })
                    .await?
                    .data;
                let expected_reads = loaded_threads
                    .iter()
                    .filter(|thread_id| *thread_id != &root_thread_id.to_string())
                    .count();
                assert!(loaded_threads.contains(&child_thread_id.to_string()));
                take_backfill_counts(&requests);
                app.harness_overrides.cwd = Some(app.config.cwd.to_path_buf());

                let control = app
                    .resume_target_session(
                        &mut tui,
                        &mut app_server,
                        crate::resume_picker::SessionTarget {
                            path: Some(root_rollout_path),
                            thread_id: root_thread_id,
                        },
                    )
                    .await?;

                assert!(matches!(control, AppRunControl::Continue));
                assert_eq!(app.chat_widget.thread_id(), Some(root_thread_id));
                assert_eq!(take_backfill_counts(&requests), (1, expected_reads));
                assert_eq!(
                    app.agent_navigation.get(&child_thread_id),
                    Some(&AgentPickerThreadEntry {
                        agent_nickname: Some("worker".to_string()),
                        agent_role: Some("worker".to_string()),
                        agent_path: Some("/root/worker".to_string()),
                        is_running: false,
                        is_closed: false,
                    })
                );

                Box::pin(app.open_agent_picker(&mut app_server)).await;

                // The picker refreshes the primary thread once. Discovered children were already
                // refreshed by the picker's initial backfill and must not be read a second time.
                assert_eq!(take_backfill_counts(&requests), (1, expected_reads + 1));
                app_server.shutdown().await?;
                proxy.await??;
                Ok(())
            })
        })?
        .join()
        .expect("session lifecycle request test thread")
}

#[test]
fn manifest_catalog_failure_keeps_existing_thread_and_config() -> Result<()> {
    const TEST_STACK_SIZE_BYTES: usize = 8 * 1024 * 1024;

    std::thread::Builder::new()
        .name("tui-manifest-catalog-failure".to_string())
        .stack_size(TEST_STACK_SIZE_BYTES)
        .spawn(|| {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            runtime.block_on(async {
                let mut app = make_test_app().await;
                let codex_home = tempdir()?;
                app.config.codex_home = codex_home.path().to_path_buf().abs();
                app.config.sqlite_home = codex_home.path().to_path_buf();

                let manifest_server = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(path("/v1/codex/provider-manifest"))
                    .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                        "schema_version": 1,
                        "models": [{
                            "id": "venado-only",
                            "display_name": "Venado only",
                            "description": "Only advertised by the configured provider manifest",
                            "context_window": 32_000,
                            "max_input_tokens": 24_000,
                            "default_reasoning_effort": "medium",
                            "supported_reasoning_efforts": ["medium"],
                            "service_tiers": []
                        }]
                    })))
                    .expect(1..)
                    .mount(&manifest_server)
                    .await;
                let provider = codex_model_provider_info::ModelProviderInfo {
                    name: "Venado".to_string(),
                    base_url: Some(format!("{}/v1", manifest_server.uri())),
                    provider_manifest_path: Some("codex/provider-manifest".to_string()),
                    experimental_bearer_token: Some("venado-test-token".to_string()),
                    ..Default::default()
                };
                std::fs::write(
                    codex_home.path().join("config.toml"),
                    format!(
                        r#"
model = "venado-only"
model_provider = "venado"

[model_providers.venado]
name = "Venado"
base_url = "{}/v1"
experimental_bearer_token = "venado-test-token"
wire_api = "responses"
provider_manifest_path = "codex/provider-manifest"
"#,
                        manifest_server.uri()
                    ),
                )?;
                app.config.model_provider_id = "venado".to_string();
                app.config.model_provider = provider.clone();
                app.config
                    .model_providers
                    .insert("venado".to_string(), provider);
                app.config.model = Some("venado-only".to_string());

                let (mut app_server, requests, unsubscribed_thread_ids, proxy) =
                    start_recording_app_server(&app.config, ThreadScopedModelListBehavior::Fail)
                        .await?;
                let source_thread_id = ThreadId::from_string(
                    &create_fake_rollout(
                        codex_home.path(),
                        "2026-01-01T00-00-00",
                        "2026-01-01T00:00:00Z",
                        "Saved user message",
                        Some("venado"),
                        /*git_info*/ None,
                    )
                    .expect("create source rollout"),
                )?;
                let source = app_server
                    .resume_thread(
                        app.config.clone(),
                        source_thread_id,
                        app.resume_model_settings(),
                    )
                    .await?;
                app.enqueue_primary_thread_session(source.session, source.turns)
                    .await?;
                let original_model_catalog = Arc::clone(&app.model_catalog);
                let original_model_catalog_provenance = app.model_catalog_provenance;
                requests.lock().expect("request recorder lock").clear();

                let replacement = app_server
                    .fork_thread(app.config.clone(), source_thread_id)
                    .await?;
                let replacement_thread_id = replacement.session.thread_id;
                app.ensure_thread_channel(replacement_thread_id);
                app.agent_navigation.upsert(
                    replacement_thread_id,
                    /*agent_nickname*/ None,
                    /*agent_role*/ None,
                    /*is_closed*/ false,
                );
                let mut transition_config = app.config.clone();
                transition_config.model = Some("should-not-commit".to_string());
                let mut tui = crate::tui::test_support::make_test_tui()?;
                let err = app
                    .replace_chat_widget_with_app_server_thread(
                        &mut tui,
                        &mut app_server,
                        transition_config,
                        replacement,
                        ThreadAttachPresentation::SessionLineage,
                        /*initial_user_message*/ None,
                    )
                    .await
                    .expect_err("thread-scoped model/list failure should abort attachment");

                assert!(
                    err.to_string().contains("failed to load models for thread"),
                    "unexpected attachment error: {err}"
                );
                assert_eq!(app.chat_widget.thread_id(), Some(source_thread_id));
                assert_eq!(app.config.model.as_deref(), Some("venado-only"));
                assert!(Arc::ptr_eq(&app.model_catalog, &original_model_catalog));
                assert_eq!(
                    app.model_catalog_provenance,
                    original_model_catalog_provenance
                );
                assert!(
                    !app.thread_event_channels
                        .contains_key(&replacement_thread_id),
                    "failed preflight should discard the replacement channel"
                );
                assert!(
                    app.agent_navigation.get(&replacement_thread_id).is_none(),
                    "failed preflight should discard replacement navigation state"
                );
                let recorded_methods = requests.lock().expect("request recorder lock").clone();
                assert!(
                    recorded_methods.iter().any(|method| method == "model/list"),
                    "replacement should attempt the thread-scoped model catalog"
                );
                assert!(
                    recorded_methods
                        .iter()
                        .any(|method| method == "thread/unsubscribe"),
                    "failed preflight should clean up the unattached replacement"
                );
                let unsubscribed_thread_ids = unsubscribed_thread_ids
                    .lock()
                    .expect("unsubscribe recorder lock")
                    .clone();
                assert!(
                    unsubscribed_thread_ids.contains(&replacement_thread_id.to_string()),
                    "failed preflight should unsubscribe the unattached replacement"
                );
                assert!(
                    !unsubscribed_thread_ids.contains(&source_thread_id.to_string()),
                    "failed preflight must not unsubscribe the current thread"
                );

                manifest_server.verify().await;
                app_server.shutdown().await?;
                proxy.await??;
                Ok(())
            })
        })?
        .join()
        .expect("manifest catalog failure test thread")
}
