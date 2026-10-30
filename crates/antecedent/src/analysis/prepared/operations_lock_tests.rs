use super::*;
#[test]
fn waiting_caller_can_cancel_and_poison_recovers() {
    let lock = std::sync::Arc::new(std::sync::Mutex::new(Some(7)));
    let guard = lock.lock().unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let waiting_ctx = ctx.clone();
    let waiting_lock = std::sync::Arc::clone(&lock);
    let (sent, received) = std::sync::mpsc::channel();
    let thread = std::thread::spawn(move || {
        let mut sent = Some(sent);
        lock_frozen_fit_with_wait(&waiting_lock, &waiting_ctx, || {
            if let Some(sent) = sent.take() {
                sent.send(()).unwrap();
            }
        })
        .map(|guard| *guard)
    });
    received.recv().unwrap();
    ctx.cancellation.cancel();
    assert!(thread.join().unwrap().is_err());
    drop(guard);
    let poison = std::sync::Arc::clone(&lock);
    assert!(
        std::thread::spawn(move || {
            let _guard = poison.lock().unwrap();
            panic!("fit panic");
        })
        .join()
        .is_err()
    );
    let fresh = ExecutionContext::for_tests(2);
    assert_eq!(*lock_frozen_fit(&lock, &fresh).unwrap(), Some(7));
}
