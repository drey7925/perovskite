use crate::lighting::{ChunkColumn, Lightfield};
use crate::sync::GenericRwLock;
use crate::sync::{DefaultSyncBackend, SyncBackend, TestonlyLoomBackend};
use std::sync::Arc;
#[test]
fn light_column_basic_test() {
    let mut col = ChunkColumn::<DefaultSyncBackend>::empty();
    col.insert_empty(3);
    col.insert_empty(5);
    col.insert_empty(7);
    let mut cursor = col.cursor_into(7);
    assert_eq!(cursor.current_pos, 7);
    assert_eq!(cursor.prev_pos, None);
    *cursor.current_occlusion_mut() = lf(1);
    cursor.mark_valid();
    cursor = cursor.advance().unwrap();

    assert_eq!(cursor.current_pos, 5);
    assert_eq!(cursor.prev_pos, Some(7));
    *cursor.current_occlusion_mut() = lf(2);
    cursor.mark_valid();
    cursor = cursor.advance().unwrap();

    assert_eq!(cursor.current_pos, 3);
    assert_eq!(cursor.prev_pos, Some(5));
    *cursor.current_occlusion_mut() = lf(3);
    cursor.mark_valid();
    assert!(cursor.advance().is_none());

    let mut cursor = col.cursor_into_first().unwrap();
    assert_eq!(cursor.current_pos, 7);
    assert_eq!(cursor.prev_pos, None);
    // 2 additional chunks
    assert_eq!(cursor.propagate_lighting(), 2);

    assert_eq!(col.get_incoming_light(8), None);
    assert_eq!(col.get_incoming_light(7), Some(Lightfield::all_on()));
    assert_eq!(
        col.get_incoming_light(5),
        Some(Lightfield::all_on() & !lf(1))
    );
    assert_eq!(
        col.get_incoming_light(3),
        Some(Lightfield::all_on() & !(lf(1) | lf(2)))
    );

    assert_eq!(col.get_incoming_light(1), None);
    col.insert_empty(1);
    let mut cursor = col.cursor_into(1);
    *cursor.current_occlusion_mut() = lf(7);
    cursor.mark_valid();
    cursor.propagate_lighting();

    assert_eq!(
        col.get_incoming_light(1),
        Some(Lightfield::all_on() & !(lf(1) | lf(2) | lf(3)))
    );
}
fn lf(i: u8) -> Lightfield {
    let mut lf = Lightfield::zero();
    lf.set(0, i, true);
    lf
}

#[test]
fn loom_test_concurrently_insert_remove() {
    loom::model(move || {
        let mut col = ChunkColumn::<TestonlyLoomBackend>::empty();

        assert_eq!(col.get_incoming_light(1), None);
        col.insert_empty(1);
        col.cursor_into(1).mark_valid();

        let mut col = Arc::new(<TestonlyLoomBackend as SyncBackend>::RwLock::new(col));
        let mut threads = vec![];
        let cc = col.clone();
        // threads.push(loom::thread::spawn(move || {
        //     let mut cc = cc.lock_write();
        //     cc.insert_empty(3);
        //     let cc = <TestonlyLoomBackend as SyncBackend>::RwLock::downgrade_writer(cc);
        //     let mut cursor = cc.cursor_into(3);
        //     *cursor.current_occlusion_mut() = lf(4) | lf(5) | lf(6) | lf(7);
        //     cursor.mark_valid();
        //     cursor.propagate_lighting();
        // }));
        let cc = col.clone();
        threads.push(loom::thread::spawn(move || {
            let mut ccl = cc.lock_write();
            ccl.insert_empty(5);
            let ccl = <TestonlyLoomBackend as SyncBackend>::RwLock::downgrade_writer(ccl);
            let mut cursor = ccl.cursor_into(5);
            *cursor.current_occlusion_mut() = lf(2) | lf(3) | lf(6) | lf(7);
            cursor.mark_valid();
            cursor.propagate_lighting();
            // cursor dropped here
            drop(ccl);

            let mut ccl = cc.lock_write();
            ccl.remove(5);
        }));
        let cc = col.clone();
        threads.push(loom::thread::spawn(move || {
            println!("4 add-rm-add thread");
            let mut ccl = cc.lock_write();
            ccl.insert_empty(4);
            let ccl = <TestonlyLoomBackend as SyncBackend>::RwLock::downgrade_writer(ccl);
            let mut cursor = ccl.cursor_into(4);
            *cursor.current_occlusion_mut() = lf(2) | lf(3) | lf(7) | lf(8);
            cursor.mark_valid();
            cursor.propagate_lighting();

            println!("4 add-rm-add added1");
            drop(ccl);

            let mut ccl = cc.lock_write();
            ccl.remove(4);
            println!("4 add-rm-add removed");
            drop(ccl);

            let mut ccl = cc.lock_write();

            ccl.insert_empty(4);
            let ccl = <TestonlyLoomBackend as SyncBackend>::RwLock::downgrade_writer(ccl);
            let mut cursor = ccl.cursor_into(4);
            *cursor.current_occlusion_mut() = lf(7) | lf(8);
            cursor.mark_valid();
            cursor.propagate_lighting();
            // cursor dropped here
            println!("4 add-rm-add added2");

            drop(ccl);

            println!("4 add-rm-add thread done");
        }));
        let cc = col.clone();
        threads.push(loom::thread::spawn(move || {
            let mut cc = cc.lock_write();
            cc.insert_empty(7);
            let cc = <TestonlyLoomBackend as SyncBackend>::RwLock::downgrade_writer(cc);
            let mut cursor = cc.cursor_into(7);
            *cursor.current_occlusion_mut() = lf(1) | lf(3) | lf(5) | lf(7);
            cursor.mark_valid();
            cursor.propagate_lighting();
        }));

        for thread in threads {
            thread.join().unwrap();
        }

        let mut col = col.lock_read();
        // let mut cursor = col.cursor_into(7);
        // cursor.propagate_lighting();

        let mut cursor = col.cursor_into(1);
        *cursor.current_occlusion_mut() = Lightfield::zero();
        cursor.propagate_lighting();

        // assert_eq!(
        //     col.get_incoming_light(1),
        //     Some(Lightfield::all_on() & !(lf(1) | lf(3) | lf(4) | lf(5) | lf(6) | lf(7) | lf(8)))
        // );
    });
}

#[test]
fn non_loom_corruption_test() {
    let mut ccl = ChunkColumn::<DefaultSyncBackend>::empty();
    ccl.insert_empty(1);
    ccl.cursor_into(1).mark_valid();

    ccl.insert_empty(5);
    let mut cursor = ccl.cursor_into(5);
    *cursor.current_occlusion_mut() = lf(2) | lf(3) | lf(6) | lf(7);
    cursor.mark_valid();
    cursor.propagate_lighting();
    // // cursor dropped here
    // ccl.remove(5);

    ccl.insert_empty(4);

    let mut cursor = ccl.cursor_into(4);
    *cursor.current_occlusion_mut() = lf(2) | lf(3) | lf(7) | lf(8);
    cursor.mark_valid();
    cursor.propagate_lighting();

    ccl.remove(4);

    // ccl.insert_empty(4);
    // let mut cursor = ccl.cursor_into(4);
    // *cursor.current_occlusion_mut() = lf(7) | lf(8);
    // cursor.mark_valid();
    // cursor.propagate_lighting();
    // cursor dropped here
    ccl.insert_empty(7);
    let mut cursor = ccl.cursor_into(7);
    *cursor.current_occlusion_mut() = lf(1) | lf(3) | lf(5) | lf(7);
    cursor.mark_valid();
    cursor.propagate_lighting();
}
