"""Workspace path resolution for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


VALID_SOURCES = {"slides", "textbooks"}
KNOWN_COURSE_DIR_NAMES = {
    "bi358": "BI358",
    "in3240": "IN3240",
    "cos226": "COS226",
}


@dataclass(frozen=True)
class SourcePaths:
    """Resolved source-specific paths inside one course workspace."""

    source: str
    input_dir: Path
    output_dir: Path
    text_dir: Path
    chunks_dir: Path
    generation_dir: Path
    retrieval_dir: Path
    milvus_dir: Path
    results_dir: Path


@dataclass(frozen=True)
class WorkspacePaths:
    """Resolved paths for one selected course workspace."""

    course: str
    data_root: Path
    course_root: Path
    input_root: Path
    output_root: Path
    milvus_root: Path
    results_root: Path
    test_set_root: Path
    test_set_with_labels_path: Path

    @classmethod
    def from_course(
        cls,
        course: str,
        repo_root: str | Path,
        data_root: str | Path | None = None,
        extra_course_roots: dict[str, str | Path] | None = None,
    ) -> "WorkspacePaths":
        """Resolve one selected course workspace.

        Args:
            course: Course identifier such as `bi358` or `in3240`.
            repo_root: Repository root used for default layout resolution.
            data_root: Optional shared data root containing multiple course
                subfolders, one per course.
            extra_course_roots: Optional explicit course-to-path overrides.

        Returns:
            A resolved workspace path bundle for the selected course.

        Raises:
            ValueError: If the course is unsupported or missing from the chosen
                layout.
            FileNotFoundError: If the resolved course root does not exist.
        """
        normalized_course = course.lower().strip()
        repo_root_path = Path(repo_root).resolve()
        overrides = {
            key.lower(): Path(value).resolve()
            for key, value in (extra_course_roots or {}).items()
        }

        if normalized_course in overrides:
            course_root = overrides[normalized_course]
            resolved_data_root = course_root.parent
        elif data_root is not None:
            resolved_data_root = Path(data_root).resolve()
            course_root = _resolve_course_root_in_data_root(
                resolved_data_root,
                normalized_course,
            )
        else:
            default_roots = default_course_roots(repo_root_path)
            if normalized_course not in default_roots:
                raise ValueError(f"Unsupported course: {course}")
            course_root = default_roots[normalized_course]
            resolved_data_root = course_root.parent

        if not course_root.exists():
            raise FileNotFoundError(f"Course workspace does not exist: {course_root}")

        input_root = _resolve_input_root(
            course_root=course_root,
            normalized_course=normalized_course,
        )
        output_root = course_root / "output"
        milvus_root = course_root / "milvus"
        results_root = course_root / "results"
        test_set_root = course_root / "test_set"

        return cls(
            course=normalized_course,
            data_root=resolved_data_root,
            course_root=course_root,
            input_root=input_root,
            output_root=output_root,
            milvus_root=milvus_root,
            results_root=results_root,
            test_set_root=test_set_root,
            test_set_with_labels_path=test_set_root / "test_set_with_labels.json",
        )

    def for_source(self, source: str) -> SourcePaths:
        """Resolve source-specific paths for `slides` or `textbooks`.

        Args:
            source: Source identifier.

        Returns:
            The source-specific path bundle.

        Raises:
            ValueError: If the source is unsupported.
        """
        normalized_source = source.lower().strip()
        if normalized_source not in VALID_SOURCES:
            raise ValueError(f"Unsupported source: {source}")

        source_output_dir = self.output_root / normalized_source
        return SourcePaths(
            source=normalized_source,
            input_dir=self.input_root / normalized_source,
            output_dir=source_output_dir,
            text_dir=source_output_dir / "text",
            chunks_dir=source_output_dir / "chunks",
            generation_dir=self.output_root / "generation" / normalized_source,
            retrieval_dir=self.output_root / "retrieval" / normalized_source,
            milvus_dir=self.milvus_root / normalized_source,
            results_dir=self.results_root / normalized_source,
        )


def _resolve_input_root(
    *,
    course_root: Path,
    normalized_course: str,
) -> Path:
    """Resolve the effective input root for one course workspace.

    Known built-in courses read raw inputs directly from the selected
    `<data-root>/<COURSE>` workspace.

    Unknown ad hoc course workspaces continue to read from `<course>/input`.
    """
    if normalized_course in KNOWN_COURSE_DIR_NAMES:
        return course_root
    return course_root / "input"


def default_course_roots(repo_root: Path) -> dict[str, Path]:
    """Return the default course layout for this repository."""
    shared_data_root = (repo_root / "data").resolve()
    return {
        course: _resolve_known_course_root(shared_data_root, course, dir_name)
        for course, dir_name in KNOWN_COURSE_DIR_NAMES.items()
    }


def _resolve_known_course_root(
    data_root: Path,
    normalized_course: str,
    preferred_dir_name: str,
) -> Path:
    """Resolve one known course root while tolerating filesystem case differences."""
    for candidate in data_root.iterdir() if data_root.exists() else ():
        if candidate.is_dir() and candidate.name.lower() == normalized_course:
            return candidate.resolve()

    return data_root / preferred_dir_name


def _resolve_course_root_in_data_root(data_root: Path, normalized_course: str) -> Path:
    """Resolve one course root from an explicit shared data root."""
    preferred_dir_name = KNOWN_COURSE_DIR_NAMES.get(normalized_course, normalized_course)
    for candidate in data_root.iterdir() if data_root.exists() else ():
        if candidate.is_dir() and candidate.name.lower() == normalized_course:
            return candidate.resolve()
    return data_root / preferred_dir_name
