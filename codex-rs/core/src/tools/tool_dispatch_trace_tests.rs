use std::fs;
use std::future::Future;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Poll;

use codex_protocol::protocol::SessionSource;
use codex_rollout_trace::ExecutionStatus;
use codex_rollout_trace::ThreadStartedTraceMetadata;
use codex_rollout_trace::ToolCallRequester;
use pretty_assertions::assert_eq;
use tempfile::TempDir;
use tokio::sync::Barrier;
use tokio_util::sync::CancellationToken;

use crate::function_tool::FunctionCallError;
use crate::session::session::Session;
use crate::session::step_context::StepContext;
use crate::session::tests::make_session_and_context;
use crate::session::turn_context::TurnContext;
use crate::tools::code_mode::CodeModeService;
use crate::tools::code_mode::CodeModeWaitHandler;
use crate::tools::code_mode::WAIT_TOOL_NAME;
use crate::tools::context::FunctionToolOutput;
use crate::tools::context::ToolCallSource;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::tools::registry::CoreToolRuntime;
use crate::tools::registry::ToolExecutor;
use crate::tools::registry::ToolRegistry;
use crate::turn_diff_tracker::TurnDiffTracker;

struct TestHandler {
    tool_name: codex_tools::ToolName,
}

impl ToolExecutor<ToolInvocation> for TestHandler {
    fn tool_name(&self) -> codex_tools::ToolName {
        self.tool_name.clone()
    }

    fn spec(&self) -> codex_tools::ToolSpec {
        codex_tools::ToolSpec::Function(codex_tools::ResponsesApiTool {
            name: self.tool_name.name.clone(),
            description: "Test tool.".to_string(),
            strict: false,
            defer_loading: None,
            parameters: codex_tools::JsonSchema::default(),
            output_schema: None,
        })
    }

    fn handle<'a>(&'a self, _invocation: ToolInvocation) -> codex_tools::ToolExecutorFuture<'a>
    where
        ToolInvocation: 'a,
    {
        Box::pin(async {
            Ok(
                Box::new(FunctionToolOutput::from_text("ok".to_string(), Some(true)))
                    as Box<dyn crate::tools::context::ToolOutput>,
            )
        })
    }
}

impl CoreToolRuntime for TestHandler {}

#[derive(Clone, Copy, Debug)]
enum TestTerminateOutcome {
    Missing,
    Terminated,
    Result,
}

struct TestCodeModeSessionProvider {
    session: Arc<TestCodeModeSession>,
}

impl codex_code_mode::CodeModeSessionProvider for TestCodeModeSessionProvider {
    fn create_session<'a>(
        &'a self,
        _delegate: Arc<dyn codex_code_mode::CodeModeSessionDelegate>,
    ) -> codex_code_mode::CodeModeSessionProviderFuture<'a> {
        let session = Arc::clone(&self.session);
        Box::pin(async move { Ok(session as Arc<dyn codex_code_mode::CodeModeSession>) })
    }
}

struct TestCodeModeSession {
    terminate_outcome: TestTerminateOutcome,
    cell_id: codex_code_mode::CellId,
    execute_barrier: Option<Arc<Barrier>>,
    execute_count: AtomicUsize,
    terminate_count: AtomicUsize,
}

impl TestCodeModeSession {
    fn new(terminate_outcome: TestTerminateOutcome, cell_id: codex_code_mode::CellId) -> Self {
        Self {
            terminate_outcome,
            cell_id,
            execute_barrier: None,
            execute_count: AtomicUsize::new(0),
            terminate_count: AtomicUsize::new(0),
        }
    }
}

impl codex_code_mode::CodeModeSession for TestCodeModeSession {
    fn execute<'a>(
        &'a self,
        _request: codex_code_mode::ExecuteRequest,
    ) -> codex_code_mode::CodeModeSessionResultFuture<'a, codex_code_mode::StartedCell> {
        self.execute_count.fetch_add(1, Ordering::Relaxed);
        Box::pin(async move {
            if let Some(execute_barrier) = &self.execute_barrier {
                execute_barrier.wait().await;
                execute_barrier.wait().await;
            }
            let cell_id = self.cell_id.clone();
            let response_cell_id = cell_id.clone();
            Ok(codex_code_mode::StartedCell::from_future(
                cell_id,
                async move {
                    Ok(codex_code_mode::RuntimeResponse::Yielded {
                        code_mode_host_duration: None,
                        cell_id: response_cell_id,
                        content_items: Vec::new(),
                    })
                },
            ))
        })
    }

    fn wait<'a>(
        &'a self,
        request: codex_code_mode::WaitRequest,
    ) -> codex_code_mode::CodeModeSessionResultFuture<'a, codex_code_mode::WaitOutcome> {
        self.terminate(request.cell_id)
    }

    fn terminate<'a>(
        &'a self,
        cell_id: codex_code_mode::CellId,
    ) -> codex_code_mode::CodeModeSessionResultFuture<'a, codex_code_mode::WaitOutcome> {
        self.terminate_count.fetch_add(1, Ordering::Relaxed);
        let terminate_outcome = self.terminate_outcome;
        Box::pin(async move {
            let response = match terminate_outcome {
                TestTerminateOutcome::Missing | TestTerminateOutcome::Result => {
                    codex_code_mode::RuntimeResponse::Result {
                        code_mode_host_duration: None,
                        error_text: (matches!(terminate_outcome, TestTerminateOutcome::Missing))
                            .then(|| format!("exec cell {cell_id} not found")),
                        cell_id,
                        content_items: Vec::new(),
                    }
                }
                TestTerminateOutcome::Terminated => codex_code_mode::RuntimeResponse::Terminated {
                    code_mode_host_duration: None,
                    cell_id,
                    content_items: Vec::new(),
                },
            };
            Ok(match terminate_outcome {
                TestTerminateOutcome::Missing => {
                    codex_code_mode::WaitOutcome::MissingCell(response)
                }
                TestTerminateOutcome::Terminated | TestTerminateOutcome::Result => {
                    codex_code_mode::WaitOutcome::LiveCell(response)
                }
            })
        })
    }

    fn shutdown<'a>(&'a self) -> codex_code_mode::CodeModeSessionResultFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }
}

#[tokio::test]
async fn dispatch_lifecycle_trace_records_direct_and_code_mode_requesters() -> anyhow::Result<()> {
    let temp = TempDir::new()?;
    let (mut session, turn) = make_session_and_context().await;
    attach_test_trace(&mut session, &turn, temp.path())?;
    session.services.rollout_thread_trace.start_code_cell_trace(
        turn.sub_id.as_str(),
        "cell-1",
        "call-code",
        "await tools.test_tool({})",
    );

    let registry = ToolRegistry::with_handler_for_test(Arc::new(TestHandler {
        tool_name: codex_tools::ToolName::plain("test_tool"),
    }));
    let session = Arc::new(session);
    let turn = Arc::new(turn);

    registry
        .dispatch_any_with_terminal_outcome(
            test_invocation(
                Arc::clone(&session),
                Arc::clone(&turn),
                "direct-call",
                "test_tool",
                ToolCallSource::Direct,
                "{}",
            ),
            /*terminal_outcome_reached*/ None,
        )
        .await?;
    registry
        .dispatch_any_with_terminal_outcome(
            test_invocation(
                session,
                turn,
                "code-mode-call",
                "test_tool",
                ToolCallSource::CodeMode {
                    cell_id: "cell-1".to_string(),
                    runtime_tool_call_id: "tool-1".to_string(),
                },
                "{}",
            ),
            /*terminal_outcome_reached*/ None,
        )
        .await?;

    let replayed = codex_rollout_trace::replay_bundle(single_bundle_dir(temp.path())?)?;
    assert_eq!(
        replayed.tool_calls["direct-call"].model_visible_call_id,
        Some("direct-call".to_string()),
    );
    assert_eq!(
        replayed.tool_calls["direct-call"].requester,
        ToolCallRequester::Model,
    );
    assert!(
        replayed.tool_calls["direct-call"]
            .raw_invocation_payload_id
            .is_some(),
        "dispatch tracing should keep the tool invocation payload",
    );
    assert!(
        replayed.tool_calls["direct-call"]
            .raw_result_payload_id
            .is_some(),
        "direct calls should keep the model-facing result payload",
    );
    assert_eq!(
        replayed.tool_calls["code-mode-call"].model_visible_call_id,
        None,
    );
    assert_eq!(
        replayed.tool_calls["code-mode-call"].code_mode_runtime_tool_id,
        Some("tool-1".to_string()),
    );
    assert_eq!(
        replayed.tool_calls["code-mode-call"].requester,
        ToolCallRequester::CodeCell {
            code_cell_id: "code_cell:call-code".to_string(),
        },
    );
    assert!(
        replayed.tool_calls["code-mode-call"]
            .raw_result_payload_id
            .is_some(),
        "code-mode calls should keep the result returned to JavaScript",
    );

    Ok(())
}

#[tokio::test]
async fn dispatch_lifecycle_trace_records_unsupported_tool_failures() -> anyhow::Result<()> {
    let temp = TempDir::new()?;
    let (mut session, turn) = make_session_and_context().await;
    attach_test_trace(&mut session, &turn, temp.path())?;

    let registry = ToolRegistry::empty_for_test();
    let session = Arc::new(session);
    let turn = Arc::new(turn);

    let result = registry
        .dispatch_any_with_terminal_outcome(
            test_invocation(
                session,
                turn,
                "unsupported-call",
                "missing_tool",
                ToolCallSource::Direct,
                "{}",
            ),
            /*terminal_outcome_reached*/ None,
        )
        .await;

    assert!(matches!(result, Err(FunctionCallError::RespondToModel(_))));
    let replayed = codex_rollout_trace::replay_bundle(single_bundle_dir(temp.path())?)?;
    let tool_call = &replayed.tool_calls["unsupported-call"];
    assert_eq!(tool_call.execution.status, ExecutionStatus::Failed);
    assert!(tool_call.raw_result_payload_id.is_some());

    Ok(())
}

#[tokio::test]
async fn dispatch_lifecycle_trace_records_incompatible_payload_failures() -> anyhow::Result<()> {
    let temp = TempDir::new()?;
    let (mut session, turn) = make_session_and_context().await;
    attach_test_trace(&mut session, &turn, temp.path())?;

    let registry = ToolRegistry::with_handler_for_test(Arc::new(TestHandler {
        tool_name: codex_tools::ToolName::plain("test_tool"),
    }));
    let session = Arc::new(session);
    let turn = Arc::new(turn);

    let result = registry
        .dispatch_any_with_terminal_outcome(
            test_invocation_with_payload(
                session,
                turn,
                "incompatible-call",
                codex_tools::ToolName::plain("test_tool"),
                ToolCallSource::Direct,
                ToolPayload::Custom {
                    input: "{}".to_string(),
                },
            ),
            /*terminal_outcome_reached*/ None,
        )
        .await;

    assert!(matches!(result, Err(FunctionCallError::Fatal(_))));
    let replayed = codex_rollout_trace::replay_bundle(single_bundle_dir(temp.path())?)?;
    let tool_call = &replayed.tool_calls["incompatible-call"];
    assert_eq!(tool_call.execution.status, ExecutionStatus::Failed);
    assert!(tool_call.raw_result_payload_id.is_some());

    Ok(())
}

#[tokio::test]
async fn missing_code_mode_wait_traces_only_the_wait_tool_call() -> anyhow::Result<()> {
    let temp = TempDir::new()?;
    let (mut session, turn) = make_session_and_context().await;
    let missing_cell_id = codex_code_mode::CellId::new("noop".to_string());
    let code_mode_session = Arc::new(TestCodeModeSession::new(
        TestTerminateOutcome::Missing,
        missing_cell_id,
    ));
    session.services.code_mode_service = CodeModeService::new(
        Arc::new(TestCodeModeSessionProvider {
            session: Arc::clone(&code_mode_session),
        }),
        &turn.config.code_mode,
        session.services.executed_tool_calls.clone(),
    );
    drop(
        session
            .services
            .code_mode_service
            .execute(test_execute_request(), &CancellationToken::new())
            .await
            .map_err(anyhow::Error::msg)?,
    );
    attach_test_trace(&mut session, &turn, temp.path())?;

    let registry = ToolRegistry::with_handler_for_test(Arc::new(CodeModeWaitHandler));
    let session = Arc::new(session);
    let turn = Arc::new(turn);

    let mut invocation = test_invocation(
        Arc::clone(&session),
        turn,
        "wait-call",
        WAIT_TOOL_NAME,
        ToolCallSource::Direct,
        r#"{"cell_id":"noop","terminate":true}"#,
    );
    invocation.tool_name = invocation.tool_name.with_default_namespace();
    assert!(
        super::tool_dispatch_invocation(&invocation)
            .expect("wait calls should produce a trace invocation")
            .tool_namespace
            .is_none()
    );

    registry
        .dispatch_any_with_terminal_outcome(invocation, /*terminal_outcome_reached*/ None)
        .await?;

    session
        .services
        .code_mode_service
        .interrupt_active_cells()
        .await;
    assert_eq!(
        code_mode_session.terminate_count.load(Ordering::Relaxed),
        1,
        "a missing cell must not be terminated again after its dispatch gate closes"
    );

    let replayed = codex_rollout_trace::replay_bundle(single_bundle_dir(temp.path())?)?;
    assert_eq!(replayed.code_cells.len(), 0);
    assert!(
        replayed.tool_calls["wait-call"]
            .raw_result_payload_id
            .is_some()
    );

    Ok(())
}

#[tokio::test]
async fn interrupted_code_mode_cells_clear_terminal_dispatch_gates() -> anyhow::Result<()> {
    let (_, turn) = make_session_and_context().await;

    for terminate_outcome in [
        TestTerminateOutcome::Missing,
        TestTerminateOutcome::Terminated,
        TestTerminateOutcome::Result,
    ] {
        let cell_id = codex_code_mode::CellId::new(format!("{terminate_outcome:?}"));
        let code_mode_session =
            Arc::new(TestCodeModeSession::new(terminate_outcome, cell_id.clone()));
        let service = CodeModeService::new(
            Arc::new(TestCodeModeSessionProvider {
                session: Arc::clone(&code_mode_session),
            }),
            &turn.config.code_mode,
            /*executed_tool_calls*/ None,
        );
        drop(
            service
                .execute(test_execute_request(), &CancellationToken::new())
                .await
                .map_err(anyhow::Error::msg)?,
        );

        service.interrupt_active_cells().await;
        service.mark_cell_ready_for_dispatch(&cell_id, /*originating_item_id*/ None);
        service.interrupt_active_cells().await;

        assert_eq!(
            code_mode_session.terminate_count.load(Ordering::Relaxed),
            1,
            "{terminate_outcome:?} must close its dispatch gate",
        );
    }

    Ok(())
}

#[tokio::test]
async fn interrupt_waits_for_runtime_cell_admission_before_snapshotting_active_cells()
-> anyhow::Result<()> {
    let (_, turn) = make_session_and_context().await;
    let execute_barrier = Arc::new(Barrier::new(2));
    let cell_id = codex_code_mode::CellId::new("created-before-admission".to_string());
    let code_mode_session = Arc::new(TestCodeModeSession {
        terminate_outcome: TestTerminateOutcome::Terminated,
        cell_id,
        execute_barrier: Some(Arc::clone(&execute_barrier)),
        execute_count: AtomicUsize::new(0),
        terminate_count: AtomicUsize::new(0),
    });
    let service = Arc::new(CodeModeService::new(
        Arc::new(TestCodeModeSessionProvider {
            session: Arc::clone(&code_mode_session),
        }),
        &turn.config.code_mode,
        /*executed_tool_calls*/ None,
    ));
    let execute_service = Arc::clone(&service);
    let execute_task = tokio::spawn(async move {
        execute_service
            .execute(test_execute_request(), &CancellationToken::new())
            .await
            .map(drop)
            .map_err(anyhow::Error::msg)
    });
    execute_barrier.wait().await;

    let interrupt = service.interrupt_active_cells();
    tokio::pin!(interrupt);
    let interrupt_completed_before_admission =
        std::future::poll_fn(|context| Poll::Ready(interrupt.as_mut().poll(context).is_ready()))
            .await;
    execute_barrier.wait().await;
    if !interrupt_completed_before_admission {
        interrupt.await;
    }
    execute_task.await??;

    assert!(
        !interrupt_completed_before_admission,
        "interrupt must wait until a created runtime cell is admitted"
    );
    assert_eq!(
        code_mode_session.terminate_count.load(Ordering::Relaxed),
        1,
        "interrupt must terminate the cell admitted by the in-flight execution"
    );

    Ok(())
}

#[tokio::test]
async fn cancelled_runtime_cell_admission_does_not_start_a_cell() -> anyhow::Result<()> {
    let (_, turn) = make_session_and_context().await;
    let cell_id = codex_code_mode::CellId::new("cancelled-before-admission".to_string());
    let code_mode_session = Arc::new(TestCodeModeSession::new(
        TestTerminateOutcome::Terminated,
        cell_id,
    ));
    let service = CodeModeService::new(
        Arc::new(TestCodeModeSessionProvider {
            session: Arc::clone(&code_mode_session),
        }),
        &turn.config.code_mode,
        /*executed_tool_calls*/ None,
    );
    let cancellation_token = CancellationToken::new();
    cancellation_token.cancel();

    service.interrupt_active_cells().await;
    let execution = service
        .execute(test_execute_request(), &cancellation_token)
        .await;

    assert_eq!(
        execution.err().as_deref(),
        Some("code mode execution cancelled")
    );
    assert_eq!(
        code_mode_session.execute_count.load(Ordering::Relaxed),
        0,
        "an interrupt completed before admission must prevent runtime execution"
    );

    Ok(())
}

fn test_execute_request() -> codex_code_mode::ExecuteRequest {
    codex_code_mode::ExecuteRequest {
        tool_call_id: "test-call".to_string(),
        enabled_tools: Vec::new(),
        source: "await new Promise(() => {});".to_string(),
        yield_time_ms: Some(1),
        max_output_tokens: None,
    }
}

fn test_invocation(
    session: Arc<Session>,
    turn: Arc<TurnContext>,
    call_id: &str,
    tool_name: &str,
    source: ToolCallSource,
    arguments: &str,
) -> ToolInvocation {
    test_invocation_with_payload(
        session,
        turn,
        call_id,
        codex_tools::ToolName::plain(tool_name),
        source,
        ToolPayload::Function {
            arguments: arguments.to_string(),
        },
    )
}

fn test_invocation_with_payload(
    session: Arc<Session>,
    turn: Arc<TurnContext>,
    call_id: &str,
    tool_name: codex_tools::ToolName,
    source: ToolCallSource,
    payload: ToolPayload,
) -> ToolInvocation {
    let step_context = StepContext::for_test(Arc::clone(&turn));
    ToolInvocation {
        session,
        step_context,
        turn,
        cancellation_token: CancellationToken::new(),
        tracker: Arc::new(tokio::sync::Mutex::new(TurnDiffTracker::new())),
        call_id: call_id.to_string(),
        tool_name,
        source,
        payload,
    }
}

fn attach_test_trace(session: &mut Session, turn: &TurnContext, root: &Path) -> anyhow::Result<()> {
    let thread_id = session.thread_id;
    let rollout_thread_trace =
        codex_rollout_trace::ThreadTraceContext::start_root_in_root_for_test(
            root,
            ThreadStartedTraceMetadata {
                thread_id: thread_id.to_string(),
                agent_path: "/root".to_string(),
                task_name: None,
                nickname: None,
                agent_role: None,
                session_source: SessionSource::Exec,
                cwd: PathBuf::from("/workspace"),
                rollout_path: None,
                model: "gpt-test".to_string(),
                provider_name: "test-provider".to_string(),
                approval_policy: "never".to_string(),
                sandbox_policy: "danger-full-access".to_string(),
            },
        )?;
    rollout_thread_trace.record_codex_turn_started(turn.sub_id.as_str());
    session.services.rollout_thread_trace = rollout_thread_trace;
    Ok(())
}

fn single_bundle_dir(root: &Path) -> anyhow::Result<PathBuf> {
    let mut entries = fs::read_dir(root)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    entries.sort();
    assert_eq!(entries.len(), 1);
    Ok(entries.remove(0))
}
