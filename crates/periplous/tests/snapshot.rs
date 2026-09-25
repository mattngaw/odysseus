use periplous::hardware::{Gpu, GpuProcesses, Host, Snapshot};

#[test]
fn json_distinguishes_missing_hardware_and_process_data_from_empty_lists() {
    let mut snapshot = Snapshot {
        schema_version: 1,
        collected_at_unix_ms: 0,
        collection_duration_ms: 1.0,
        host: Host::default(),
        gpus: None,
        issues: vec![],
    };
    let json = serde_json::to_value(&snapshot).unwrap();
    assert!(json["gpus"].is_null());
    assert!(json["host"]["cpu"].is_null());
    snapshot.gpus = Some(vec![Gpu {
        processes: GpuProcesses {
            compute: Some(vec![]),
            graphics: None,
        },
        ..Gpu::default()
    }]);
    let json = serde_json::to_value(&snapshot).unwrap();
    assert_eq!(
        json["gpus"][0]["processes"]["compute"],
        serde_json::json!([])
    );
    assert!(json["gpus"][0]["processes"]["graphics"].is_null());
    assert!(json["gpus"][0]["power_watts"].is_null());
}

#[test]
fn command_help_and_bad_arguments_do_not_collect_hardware() {
    let binary = env!("CARGO_BIN_EXE_periplous");
    let help = std::process::Command::new(binary)
        .arg("--help")
        .output()
        .unwrap();
    assert!(help.status.success());
    assert!(
        String::from_utf8(help.stdout)
            .unwrap()
            .contains("periplous snapshot")
    );
    let bad = std::process::Command::new(binary)
        .args(["snapshot", "--unsupported"])
        .output()
        .unwrap();
    assert_eq!(bad.status.code(), Some(2));
    assert!(bad.stdout.is_empty());
}
