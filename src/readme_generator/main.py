from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

if __package__ in (None, ""):
    # Support direct execution: python src/readme_generator/main.py
    project_src = Path(__file__).resolve().parent.parent
    if str(project_src) not in sys.path:
        sys.path.append(str(project_src))
    from readme_generator.crew import (
        ReadmeWorkflowCrew as WorkflowRunner,
        WorkflowInput,
        build_github_only_legacy_workflow_input,
        build_legacy_workflow_input,
        kickoff as crew_kickoff,
    )
else:
    from .crew import (
        ReadmeWorkflowCrew as WorkflowRunner,
        WorkflowInput,
        build_github_only_legacy_workflow_input,
        build_legacy_workflow_input,
        kickoff as crew_kickoff,
    )


def kickoff(
    workflow_input: Optional[WorkflowInput] = None,
    enabled_stages: Optional[List[str]] = None,
):
    return crew_kickoff(workflow_input=workflow_input, enabled_stages=enabled_stages)


if __name__ == "__main__":
    print("Running github-only workflow. To run full workflow, call kickoff(..., enabled_stages=None).")
    kickoff(
        workflow_input=build_legacy_workflow_input(),
        # enabled_stages=["github_pr"],
    )
