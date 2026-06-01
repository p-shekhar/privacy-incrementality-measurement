from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_NAME = "privacy_robust_incrementality_measurement"


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    code_dir: Path
    data_dir: Path
    artifact_dir: Path
    figure_dir: Path
    table_dir: Path
    workspace_dir: Path

    @classmethod
    def from_cwd(cls, cwd: Path | None = None) -> "ProjectPaths":
        here = (cwd or Path.cwd()).resolve()
        project_root = here
        while project_root.name != PROJECT_NAME and project_root != project_root.parent:
            project_root = project_root.parent
        if project_root.name != PROJECT_NAME:
            raise RuntimeError(f"Could not find {PROJECT_NAME} above {here}")

        repo_root = project_root
        while repo_root.name != "ranking_sys" and repo_root != repo_root.parent:
            repo_root = repo_root.parent

        code_dir = project_root / "code"
        artifact_dir = code_dir / "artifacts"
        paths = cls(
            project_root=project_root,
            code_dir=code_dir,
            data_dir=repo_root / "data",
            artifact_dir=artifact_dir,
            figure_dir=artifact_dir / "figures",
            table_dir=artifact_dir / "tables",
            workspace_dir=artifact_dir / "workspace",
        )
        paths.ensure()
        return paths

    def ensure(self) -> None:
        for path in [self.artifact_dir, self.figure_dir, self.table_dir, self.workspace_dir]:
            path.mkdir(parents=True, exist_ok=True)

