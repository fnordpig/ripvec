use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn prints_version() {
    Command::cargo_bin("ripvec")
        .unwrap()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::starts_with("ripvec"));
}

#[test]
fn prints_help() {
    Command::cargo_bin("ripvec")
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Semantic code search"));
}

/// Requires model download — run with `cargo test -- --ignored`
#[test]
#[ignore = "requires model download; run with `cargo test -- --ignored`"]
fn searches_fixture_directory() {
    Command::cargo_bin("ripvec")
        .unwrap()
        .args(["find the main entry point", "tests/fixtures/", "-n", "3"])
        .assert()
        .success();
}

#[test]
fn fails_on_missing_query() {
    Command::cargo_bin("ripvec").unwrap().assert().failure();
}
