import os
import traceback
from typing import Any, Dict, List, Optional

from crewai.tools import tool
from github import Github, GithubException
from github.PullRequest import PullRequest
from github.Repository import Repository

from .memory_tool import GlobalMemory
from .web_tool import clear_proxy_in_process, restore_proxy_in_process


def _resolve_repo(github_config: Dict[str, Any]) -> Dict[str, str]:
    repo_owner = (github_config.get("repo_owner") or "").strip()
    repo_name = (github_config.get("repo_name") or "").strip()
    repo_full = (github_config.get("repo") or "").strip()

    if (not repo_owner or not repo_name) and repo_full and "/" in repo_full:
        repo_owner, repo_name = repo_full.split("/", 1)

    return {"repo_owner": repo_owner, "repo_name": repo_name}


class GithubPRTool:
    global_memory = None

    @staticmethod
    def _proxy_restore():
        return restore_proxy_in_process(
            proxy_backup={
                "http_proxy": "http://proxy-dmz.intel.com:912",
                "https_proxy": "http://proxy-dmz.intel.com:912",
            }
        )

    @staticmethod
    def _proxy_clear(proxy_backup):
        clear_proxy_in_process(proxy_backup=proxy_backup)

    @staticmethod
    def _read_github_config() -> Dict[str, Any]:
        cfg = GithubPRTool.global_memory.memory_retrieve("github_config") or {}
        token = (
            (cfg.get("github_token") or "").strip()
            or os.getenv("README_GENERATOR_GITHUB_TOKEN", "").strip()
            or os.getenv("GITHUB_TOKEN", "").strip()
            or os.getenv("GH_TOKEN", "").strip()
        )
        if token:
            cfg["github_token"] = token
        return cfg

    @staticmethod
    def _read_publish_content() -> str:
        family_content = GithubPRTool.global_memory.memory_retrieve("family_content") or ""
        return family_content if str(family_content).strip() else ""

    @staticmethod
    def _read_publish_artifacts() -> Dict[str, str]:
        family_md = GithubPRTool.global_memory.memory_retrieve("family_md") or ""
        family_index_js = GithubPRTool.global_memory.memory_retrieve("family_index_js") or ""
        review_failure_report = GithubPRTool.global_memory.memory_retrieve("review_failure_report") or ""
        family_js_files = GithubPRTool.global_memory.memory_retrieve("family_js_files") or []
        if not isinstance(family_js_files, list):
            family_js_files = []
        return {
            "family_md": str(family_md or ""),
            "family_index_js": str(family_index_js or ""),
            "family_content": str(GithubPRTool._read_publish_content() or ""),
            "review_failure_report": str(review_failure_report or ""),
            "family_js_files": family_js_files,
        }

    @staticmethod
    def _is_directory_like(path: str) -> bool:
        p = (path or "").strip()
        if not p:
            return False
        if p.endswith("/"):
            return True
        last = p.rsplit("/", 1)[-1]
        return "." not in last

    @staticmethod
    def _join_dir_file(dir_path: str, filename: str) -> str:
        d = (dir_path or "").strip().rstrip("/")
        return f"{d}/{filename}" if d else filename

    @staticmethod
    def _resolve_publish_targets(github_config: Dict[str, Any], artifacts: Dict[str, str]) -> List[Dict[str, str]]:
        publish_items = github_config.get("publish_items") or github_config.get("files") or github_config.get("upload_files")
        targets: List[Dict[str, str]] = []

        if isinstance(publish_items, list) and publish_items:
            for idx, item in enumerate(publish_items):
                if not isinstance(item, dict):
                    raise ValueError(f"publish_items[{idx}] must be an object")
                raw_path = str(item.get("path") or "").strip()
                if not raw_path:
                    raise ValueError(f"publish_items[{idx}].path is required")
                content = item.get("content")
                content_key = str(item.get("content_key") or item.get("artifact") or "").strip()
                if content is None:
                    if not content_key:
                        raise ValueError(f"publish_items[{idx}] must provide content or content_key/artifact")
                    if content_key not in artifacts:
                        raise ValueError(f"publish_items[{idx}].content_key {content_key} not found in artifacts")
                    content = artifacts[content_key]
                targets.append(
                    {
                        "path": raw_path,
                        "content": str(content),
                        "label": str(item.get("label") or content_key or f"file-{idx}"),
                    }
                )
            return targets

        # Legacy mode: publish family_md + family_index_js via md_path/js_path/path
        md_path = (github_config.get("md_path") or github_config.get("path") or "").strip()
        js_path = (github_config.get("js_path") or "").strip()
        if not md_path:
            raise ValueError("github_config.path (or md_path) is required for legacy publish mode")

        if GithubPRTool._is_directory_like(md_path):
            md_target = GithubPRTool._join_dir_file(md_path, "README.md")
            js_target = GithubPRTool._join_dir_file(md_path, "index.js")
        else:
            md_target = md_path
            if js_path:
                js_target = (
                    GithubPRTool._join_dir_file(js_path, "index.js")
                    if GithubPRTool._is_directory_like(js_path)
                    else js_path
                )
            elif "/" in md_target:
                js_target = f"{md_target.rsplit('/', 1)[0]}/index.js"
            else:
                js_target = "index.js"

        targets.append({"path": md_target, "content": artifacts["family_md"], "label": "README.md"})
        js_files = artifacts.get("family_js_files") or []
        if js_files:
            base_dir = md_target.rsplit("/", 1)[0] if "/" in md_target else ""
            for i, item in enumerate(js_files):
                if not isinstance(item, dict):
                    continue
                raw_name = str(item.get("path") or f"file_{i}.js").strip()
                name = raw_name.split("/")[-1]
                if not name.endswith(".js"):
                    name = f"{name}.js"
                js_path_multi = GithubPRTool._join_dir_file(base_dir, name) if base_dir else name
                targets.append(
                    {
                        "path": js_path_multi,
                        "content": str(item.get("content") or ""),
                        "label": name,
                    }
                )
        else:
            targets.append({"path": js_target, "content": artifacts["family_index_js"], "label": "index.js"})
        if artifacts.get("review_failure_report", "").strip():
            if GithubPRTool._is_directory_like(md_path):
                review_path = GithubPRTool._join_dir_file(md_path, "remote_test_review.md")
            elif "/" in md_target:
                review_path = f"{md_target.rsplit('/', 1)[0]}/remote_test_review.md"
            else:
                review_path = "remote_test_review.md"
            targets.append(
                {
                    "path": review_path,
                    "content": artifacts["review_failure_report"],
                    "label": "remote_test_review.md",
                }
            )
        return targets

    @staticmethod
    def _ensure_branch(repo: Repository, branch_name: str, base_branch: str) -> str:
        try:
            repo.get_branch(branch_name)
            return repo.get_branch(branch_name).commit.sha
        except GithubException:
            base_ref = repo.get_branch(base_branch)
            repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base_ref.commit.sha)
            return base_ref.commit.sha

    @staticmethod
    def _upsert_file(repo: Repository, branch: str, path: str, content: str, message: str) -> str:
        content_bytes = content.encode("utf-8")
        try:
            existing = repo.get_contents(path, ref=branch)
            if isinstance(existing, list):
                existing = existing[0]
            commit = repo.update_file(
                path=path,
                message=message,
                content=content_bytes,
                sha=existing.sha,
                branch=branch,
            )
            return commit["commit"].sha
        except GithubException as e:
            if 404 in e.args:
                commit = repo.create_file(
                    path=path,
                    message=message,
                    content=content_bytes,
                    branch=branch,
                )
                return commit["commit"].sha
            raise e

    @staticmethod
    def _ensure_pr(
        repo: Repository,
        repo_owner: str,
        head_branch: str,
        base_branch: str,
        pr_title: str,
        pr_description: str,
    ) -> PullRequest:
        # Prefer matching by branch ref on current repository to avoid owner/head mismatch.
        pulls = repo.get_pulls(state="open", base=base_branch)
        for pr in pulls:
            try:
                if pr.head.ref == head_branch and pr.head.repo and pr.head.repo.full_name == repo.full_name:
                    pr.edit(title=pr_title, body=pr_description)
                    return pr
            except Exception:
                continue

        # Backward-compatible fallback for owner-qualified head query.
        head_ref = f"{repo_owner}:{head_branch}"
        pulls_by_head = repo.get_pulls(state="open", head=head_ref, base=base_branch)
        if pulls_by_head.totalCount > 0:
            pr = pulls_by_head[0]
            pr.edit(title=pr_title, body=pr_description)
            return pr

        try:
            return repo.create_pull(
                title=pr_title,
                body=pr_description,
                head=head_branch,
                base=base_branch,
            )
        except GithubException:
            # If a PR already exists (e.g., race or head-owner mismatch), fetch and reuse it.
            pulls_retry = repo.get_pulls(state="open", base=base_branch)
            for pr in pulls_retry:
                try:
                    if pr.head.ref == head_branch and pr.head.repo and pr.head.repo.full_name == repo.full_name:
                        pr.edit(title=pr_title, body=pr_description)
                        return pr
                except Exception:
                    continue
            raise

    @tool("get_github_config")
    def get_github_config():
        """Retrieve GITHUB_CONFIG from GLOBAL_MEMORY."""
        return GithubPRTool._read_github_config()

    @tool("get_publish_context")
    def get_publish_context() -> Dict[str, Any]:
        """Get publish context including github_config and final publish content from memory."""
        artifacts = GithubPRTool._read_publish_artifacts()
        return {
            "github_config": GithubPRTool._read_github_config(),
            "family_md": artifacts["family_md"],
            "family_index_js": artifacts["family_index_js"],
            "family_js_files": artifacts["family_js_files"],
            "review_failure_report": artifacts["review_failure_report"],
        }

    @tool("validate_publish_context")
    def validate_publish_context(github_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate repo/branch/pr context before publishing."""
        proxy_backup = GithubPRTool._proxy_restore()
        try:
            repo_info = _resolve_repo(github_config)
            repo_owner = repo_info["repo_owner"]
            repo_name = repo_info["repo_name"]
            token = (
                (github_config.get("github_token") or "").strip()
                or os.getenv("README_GENERATOR_GITHUB_TOKEN", "").strip()
                or os.getenv("GITHUB_TOKEN", "").strip()
                or os.getenv("GH_TOKEN", "").strip()
            )
            base_branch = github_config.get("base_branch", "main").strip()
            head_branch = github_config.get("head_branch", "dev").strip()

            required_missing = []
            if not token:
                required_missing.append("github_token")
            if not repo_owner:
                required_missing.append("repo_owner")
            if not repo_name:
                required_missing.append("repo_name")
            if not base_branch:
                required_missing.append("base_branch")
            if not head_branch:
                required_missing.append("head_branch")
            has_legacy_path = bool((github_config.get("md_path") or github_config.get("path") or "").strip())
            has_publish_items = isinstance(github_config.get("publish_items") or github_config.get("files") or github_config.get("upload_files"), list)
            if not has_legacy_path and not has_publish_items:
                required_missing.append("path/md_path or publish_items/files/upload_files")
            if required_missing:
                return {"ok": False, "missing": required_missing}

            gh = Github(token)
            repo = gh.get_repo(f"{repo_owner}/{repo_name}")
            repo.get_branch(base_branch)
            return {"ok": True, "repo": f"{repo_owner}/{repo_name}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}
        finally:
            GithubPRTool._proxy_clear(proxy_backup)

    @tool("publish_family_artifacts")
    def publish_family_artifacts(github_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Publish family README.md + index.js and create/update PR."""
        proxy_backup = GithubPRTool._proxy_restore()
        try:
            # Keep repo/path/branch source stable from memory to avoid accidental split publishes.
            base_cfg = GithubPRTool._read_github_config()
            cfg = dict(base_cfg)
            if github_config:
                for k in ("pr_title", "pr_description", "commit_message", "github_token"):
                    v = github_config.get(k)
                    if v is not None:
                        cfg[k] = v
            artifacts = GithubPRTool._read_publish_artifacts()
            if not artifacts["family_md"] or not artifacts["family_index_js"]:
                raise ValueError("family_md and family_index_js are required before PR publish.")

            targets = GithubPRTool._resolve_publish_targets(cfg, artifacts)
            repo_info = _resolve_repo(cfg)
            repo_owner = repo_info["repo_owner"]
            repo_name = repo_info["repo_name"]
            token = (
                (cfg.get("github_token") or "").strip()
                or os.getenv("README_GENERATOR_GITHUB_TOKEN", "").strip()
                or os.getenv("GITHUB_TOKEN", "").strip()
                or os.getenv("GH_TOKEN", "").strip()
            )
            base_branch = (cfg.get("base_branch") or "main").strip()
            head_branch = (cfg.get("head_branch") or "dev").strip()
            pr_title = (cfg.get("pr_title") or "docs: update family README").strip()
            pr_description = (cfg.get("pr_description") or "").strip()
            commit_message = (cfg.get("commit_message") or "docs: update family README and index").strip()

            if not token:
                raise ValueError("github_token is required")
            if not repo_owner or not repo_name:
                raise ValueError("repo_owner and repo_name are required")

            gh = Github(token)
            repo = gh.get_repo(f"{repo_owner}/{repo_name}")
            GithubPRTool._ensure_branch(repo, head_branch, base_branch)

            file_commits: List[Dict[str, str]] = []
            for item in targets:
                p = str(item["path"]).strip()
                c = str(item["content"])
                label = str(item.get("label") or p)
                sha = GithubPRTool._upsert_file(
                    repo=repo,
                    branch=head_branch,
                    path=p,
                    content=c,
                    message=f"{commit_message} ({label})",
                )
                file_commits.append({"path": p, "label": label, "commit_sha": sha})

            pr = GithubPRTool._ensure_pr(
                repo=repo,
                repo_owner=repo_owner,
                head_branch=head_branch,
                base_branch=base_branch,
                pr_title=pr_title,
                pr_description=pr_description,
            )
            pr_info = {"number": pr.number, "url": pr.html_url, "status": "open"}
            GithubPRTool.global_memory.memory_store("pr_info", pr_info)
            return {
                "ok": True,
                "pr": pr_info,
                "files": file_commits,
                # Backward compatible fields for current md/js usage.
                "md_path": next((x["path"] for x in file_commits if x["label"] == "README.md"), ""),
                "js_path": next((x["path"] for x in file_commits if x["label"] == "index.js"), ""),
                "md_commit_sha": next((x["commit_sha"] for x in file_commits if x["label"] == "README.md"), ""),
                "js_commit_sha": next((x["commit_sha"] for x in file_commits if x["label"] == "index.js"), ""),
            }
        except Exception as e:
            traceback.print_exc()
            return {"ok": False, "error": str(e)}
        finally:
            GithubPRTool._proxy_clear(proxy_backup)

    @tool("memory_store_pr_info")
    def memory_store_pr_info(pr_num, url, status):
        """Store PR information into GLOBAL_MEMORY."""
        GithubPRTool.global_memory.memory_store(
            "pr_info",
            {
                "number": pr_num,
                "url": url,
                "status": status,
            },
        )
        return True
