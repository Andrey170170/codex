use super::*;

#[tokio::test]
async fn wait_until_clear_waits_for_every_registration() {
    let service = ElicitationService::new();
    let first = service.register();
    let second = service.register();
    let waiting = tokio::spawn({
        let service = service.clone();
        async move { service.wait_until_clear().await }
    });

    drop(first);
    tokio::task::yield_now().await;
    assert!(!waiting.is_finished());

    drop(second);
    waiting.await.expect("elicitation waiter should complete");
}

#[tokio::test]
async fn wait_until_clear_or_cancelled_stops_waiting_when_cancelled() {
    let service = ElicitationService::new();
    let _registration = service.register();
    let cancellation_token = CancellationToken::new();
    let waiting = tokio::spawn({
        let service = service.clone();
        let cancellation_token = cancellation_token.clone();
        async move {
            service
                .wait_until_clear_or_cancelled(&cancellation_token)
                .await
        }
    });

    tokio::task::yield_now().await;
    assert!(!waiting.is_finished());

    cancellation_token.cancel();
    waiting
        .await
        .expect("cancelled elicitation waiter should complete");
}
