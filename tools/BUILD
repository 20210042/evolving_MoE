# """Handoff note: Bazel target definition for the CodeContests export helper.
# Build this only in an environment that has the CodeContests protobuf, Abseil, and Riegeli dependencies
# available."""

cc_binary(
    name = "export_codecontests_jsonl",
    srcs = ["export_codecontests_jsonl.cc"],
    deps = [
        "//:contest_problem_cc_proto",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_riegeli//riegeli/bytes:fd_reader",
        "@com_google_riegeli//riegeli/records:record_reader",
    ],
)
