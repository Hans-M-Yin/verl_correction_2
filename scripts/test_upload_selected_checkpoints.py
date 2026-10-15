"""Offline tests for archive visibility, file selection and integrity checks."""

import contextlib
import hashlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import upload_selected_checkpoints as archive


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = self.root / "manifest.json"
        rows = []
        for model_id, visibility, name in [
            ("main", "private", "main_experiment"),
            ("model-001", "public", "other_experiment"),
        ]:
            source = self.root / name
            source.mkdir()
            config = {"model_type": "qwen2_5_vl", "hidden_size": 2,
                      "_name_or_path": "/mnt/internal/model",
                      "text_config": {"api_key": "sk-" + "x" * 32, "hidden_size": 2}}
            (source / "config.json").write_text(json.dumps(config))
            for filename in ["tokenizer_config.json", "preprocessor_config.json"]:
                (source / filename).write_text('{"_name_or_path":"/home/internal/model"}')
            # These are vocabulary contents, not metadata; changing them breaks loading.
            (source / "tokenizer.json").write_text('{"vocab":{"MIRA":1,"paper":2}}')
            (source / "chat_template.jinja").write_text("{{ messages }}")
            (source / "results.json").write_text('{"do_not_upload":true}')
            (source / ".env").write_text("do_not_upload")
            header = json.dumps({"weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
                                 "__metadata__": {"format": "pt"}}).encode()
            (source / "model-00001.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")
            (source / archive.INDEX).write_text(json.dumps({"weight_map": {"weight": "model-00001.safetensors"}}))
            rows.append({"id": model_id, "visibility": visibility,
                         "source": name, "base_model": "Qwen/Qwen2.5-VL-7B-Instruct"})
        self.manifest.write_text(json.dumps({"version": 1, "models": rows}))

    def models(self):
        return archive.load_models(self.manifest, self.root)

    @staticmethod
    def fake_license(model, dest, token):
        path = dest / "LICENSE"
        path.write_text("Upstream license fixture\n")
        return [path]

    def prepare(self):
        with patch.object(archive, "attach_license", self.fake_license), contextlib.redirect_stdout(io.StringIO()):
            return archive.prepare(self.models(), self.root / "stage", None, "owner/private", "owner/public")

    def test_offline_preview_never_contacts_hub_or_stages(self):
        with patch("huggingface_hub.HfApi") as api, contextlib.redirect_stdout(io.StringIO()):
            result = archive.main(["--namespace", "owner", "--manifest", str(self.manifest),
                                   "--source-root", str(self.root), "--work-dir", str(self.root / "work")])
        self.assertEqual(result, 0)
        api.assert_not_called()
        self.assertFalse((self.root / "work").exists())

    def test_mapping_private_only_and_public_artifacts_whitelisted(self):
        original = (self.root / "other_experiment/config.json").read_bytes()
        plan = self.prepare()
        public = plan["stages"]["public"]
        private = plan["stages"]["private"]
        self.assertTrue((private / "archive_manifest.private.json").is_file())
        self.assertFalse((public / "archive_manifest.private.json").exists())
        self.assertFalse(any("main" in name for name in plan["expected"]["public"]))
        self.assertFalse(any(name.endswith(("results.json", ".env")) for name in plan["expected"]["public"]))
        self.assertNotIn("other_experiment", (public / "README.md").read_text())
        config = json.loads((public / "model-001/config.json").read_text())
        self.assertNotIn("_name_or_path", config)
        self.assertNotIn("api_key", config["text_config"])
        self.assertEqual(config["hidden_size"], 2)
        self.assertEqual((self.root / "other_experiment/config.json").read_bytes(), original)
        self.assertEqual((public / "model-001/tokenizer.json").read_bytes(),
                         (self.root / "other_experiment/tokenizer.json").read_bytes())
        self.assertTrue((public / "model-001/model-00001.safetensors").is_symlink())
        self.assertEqual((public / "model-001/model-00001.safetensors").resolve(),
                         self.root / "other_experiment/model-00001.safetensors")

    def test_private_target_cannot_be_existing_public_repo(self):
        api = Mock()
        api.model_info.return_value = SimpleNamespace(private=False, siblings=[])
        with self.assertRaisesRegex(archive.ArchiveError, "Visibility mismatch"):
            archive.ensure_repo(api, "owner/private", True, {})
        api.create_repo.assert_not_called()

    def test_existing_unrelated_files_block_upload_target(self):
        api = Mock()
        api.model_info.return_value = SimpleNamespace(
            private=False, siblings=[SimpleNamespace(rfilename="unrelated.json")])
        with self.assertRaisesRegex(archive.ArchiveError, "unexpected files"):
            archive.ensure_repo(api, "owner/public", False, {"README.md": {}})

    def test_execute_uploads_private_first_with_separate_allowlists(self):
        api = Mock()
        with patch("huggingface_hub.HfApi", return_value=api), \
                patch("huggingface_hub.get_token", return_value="test-token"), \
                patch.object(archive, "attach_license", self.fake_license), \
                patch.object(archive, "ensure_repo") as ensure, \
                patch.object(archive, "verify_remote") as verify, \
                contextlib.redirect_stdout(io.StringIO()):
            archive.main(["--namespace", "owner", "--manifest", str(self.manifest),
                          "--source-root", str(self.root), "--work-dir", str(self.root / "work"), "--execute"])
        calls = api.upload_large_folder.call_args_list
        self.assertEqual(len(calls), 2)
        private, public = (call.kwargs for call in calls)
        self.assertEqual(private["repo_id"], "owner/checkpoint-main-private")
        self.assertTrue(private["private"])
        self.assertEqual(public["repo_id"], "owner/vl-checkpoint-archive")
        self.assertFalse(public["private"])
        self.assertIn("archive_manifest.private.json", private["allow_patterns"])
        self.assertNotIn("archive_manifest.private.json", public["allow_patterns"])
        self.assertIn("model-001/model-00001.safetensors", public["allow_patterns"])
        self.assertNotIn("model-00001.safetensors", public["allow_patterns"])
        self.assertEqual(ensure.call_count, 4)
        self.assertEqual(verify.call_count, 2)

    def test_missing_or_path_traversing_shard_rejected(self):
        path = self.root / "other_experiment" / archive.INDEX
        for name in ["missing.safetensors", "../outside.safetensors"]:
            path.write_text(json.dumps({"weight_map": {"weight": name}}))
            with self.assertRaises(archive.ArchiveError):
                self.models()

    def test_public_weight_header_with_internal_provenance_rejected(self):
        model = self.models()[1]
        path = model.source / "model-00001.safetensors"
        header = json.dumps({"weight": {"data_offsets": [0, 2]},
                             "__metadata__": {"source": "/mnt/internal/model"}}).encode()
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")
        with self.assertRaisesRegex(archive.ArchiveError, "internal reference"):
            archive.check_weight_header(path, model, [])

    def test_missing_license_uses_verified_qwen_copy(self):
        model = self.models()[0]
        dest = self.root / "license-stage"
        dest.mkdir()
        readme = self.root / "upstream-readme.md"
        readme.write_text("\n---\nlicense: apache-2.0\nlanguage:\n- en\n---\nModel card\n")
        with patch("huggingface_hub.list_repo_files", return_value=["README.md", "config.json"]), \
                patch("huggingface_hub.hf_hub_download", return_value=str(readme)), \
                contextlib.redirect_stdout(io.StringIO()):
            files = archive.attach_license(model, dest, None)
        self.assertEqual({p.name for p in files}, {"LICENSE", "UPSTREAM_LICENSE.md"})
        self.assertEqual(hashlib.sha256((dest / "LICENSE").read_bytes()).hexdigest(), archive.LICENSE_SHA256)
        self.assertIn(model.base_model, (dest / "UPSTREAM_LICENSE.md").read_text())
        self.assertIn("QwenLM/Qwen2.5-VL", (dest / "UPSTREAM_LICENSE.md").read_text())

    def test_missing_license_with_different_declaration_is_rejected(self):
        dest = self.root / "license-stage"
        dest.mkdir()
        readme = self.root / "upstream-readme.md"
        readme.write_text("---\nlicense: other\n---\nNot an Apache-2.0 model\n")
        with patch("huggingface_hub.list_repo_files", return_value=["README.md"]), \
                patch("huggingface_hub.hf_hub_download", return_value=str(readme)):
            with self.assertRaisesRegex(archive.ArchiveError, "Cannot confirm Apache-2.0"):
                archive.attach_license(self.models()[0], dest, None)
        self.assertFalse((dest / "LICENSE").exists())

    def test_upstream_license_and_notice_take_priority(self):
        dest = self.root / "license-stage"
        dest.mkdir()
        legal = self.root / "upstream-legal"
        legal.mkdir()
        (legal / "LICENSE.txt").write_text("Upstream license\n")
        (legal / "NOTICE").write_text("Upstream notices\n")
        with patch("huggingface_hub.list_repo_files", return_value=["LICENSE.txt", "NOTICE", "README.md"]), \
                patch("huggingface_hub.hf_hub_download", side_effect=lambda repo, name, **kw: str(legal / name)):
            files = archive.attach_license(self.models()[0], dest, None)
        self.assertEqual({p.name for p in files}, {"LICENSE.txt", "NOTICE"})
        self.assertEqual((dest / "NOTICE").read_text(), "Upstream notices\n")
        self.assertFalse((dest / "UPSTREAM_LICENSE.md").exists())

    def test_qwen3_missing_license_uses_its_own_official_source(self):
        model = self.models()[1]
        model.base_model = "Qwen/Qwen3-VL-4B-Instruct"
        dest = self.root / "license-stage"
        dest.mkdir()
        readme = self.root / "upstream-readme.md"
        readme.write_text("---\nlicense: apache-2.0\n---\n")
        with patch("huggingface_hub.list_repo_files", return_value=["README.md"]), \
                patch("huggingface_hub.hf_hub_download", return_value=str(readme)), \
                contextlib.redirect_stdout(io.StringIO()):
            archive.attach_license(model, dest, None)
        self.assertIn("QwenLM/Qwen3-VL", (dest / "UPSTREAM_LICENSE.md").read_text())

    def test_remote_hashes_visibility_and_size_are_verified(self):
        plan = self.prepare()
        stage = plan["stages"]["public"]
        expected = plan["expected"]["public"]
        siblings = []
        for name, wanted in expected.items():
            contents = (stage / name).read_bytes()
            siblings.append(SimpleNamespace(
                rfilename=name, size=wanted["size"],
                lfs={"sha256": wanted["sha256"]} if name.endswith(".safetensors") else None,
                blob_id=hashlib.sha1(f"blob {len(contents)}\0".encode() + contents).hexdigest()))
        api = Mock()
        api.model_info.return_value = SimpleNamespace(private=False, siblings=siblings)
        with contextlib.redirect_stdout(io.StringIO()):
            archive.verify_remote(api, "owner/public", expected, stage, False)
        siblings[0].size += 1
        with self.assertRaisesRegex(archive.ArchiveError, "size mismatch"):
            archive.verify_remote(api, "owner/public", expected, stage, False)


if __name__ == "__main__":
    unittest.main()
