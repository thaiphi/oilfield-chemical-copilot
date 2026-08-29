from __future__ import annotations

import errno
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from oilfield_chemical_copilot.evaluation import private_artifact_publication as publication


def _publish_single_member(
    bound: publication.AuthenticatedPublicationDirectory,
    *,
    content: bytes,
) -> None:
    staging = bound.create_staging(".v1.", ".tmp")
    staging.mkdir("sealed")
    staging.write_exclusive("sealed/payload.bin", content)
    staging.sync_directory("sealed")
    staging.sync_root()
    bound.publish_no_replace(staging, "v1")
    bound.sync_parent()


def test_capability_rejects_escape_before_opening_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    approved = tmp_path / "private"
    approved.mkdir()
    opened = False

    def forbidden_open(*_args: object, **_kwargs: object) -> object:
        nonlocal opened
        opened = True
        raise AssertionError("filesystem anchor must not be opened")

    monkeypatch.setattr(publication, "_open_authenticated_parent", forbidden_open)

    with pytest.raises(
        publication.PrivateArtifactPublicationError,
        match="^PRIVATE_ARTIFACT_PATH_REJECTED$",
    ):
        with publication.authenticated_publication_directory(
            approved_private_root=approved,
            publication_parent=tmp_path / "private-sibling" / "output",
            lock_name=".publish.lock",
        ):
            pytest.fail("escape unexpectedly acquired a capability")

    assert not opened


def test_capability_never_binds_retargeted_ancestor_during_real_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    approved = tmp_path / "private"
    output = approved / "output"
    parent = output / "e1a4"
    displaced = approved / "authenticated-output"
    replacement = approved / "replacement-output"
    approved.mkdir()
    output.mkdir()
    attack_happened = False
    platform_class = (
        publication._WindowsPublicationDirectory
        if os.name == "nt"
        else publication._PosixPublicationDirectory
    )
    real_take = platform_class.take

    def swap_then_take(cls: type[object], handle: object) -> object:
        nonlocal attack_happened
        attack_happened = True
        output.rename(displaced)
        parent.mkdir(parents=True)
        return real_take(handle)

    monkeypatch.setattr(platform_class, "take", classmethod(swap_then_take))

    try:
        try:
            with publication.authenticated_publication_directory(
                approved_private_root=approved,
                publication_parent=parent,
                lock_name=".publish.lock",
            ) as bound:
                _publish_single_member(bound, content=b"synthetic-secret")
        except publication.PrivateArtifactPublicationError:
            pass
    finally:
        if output.exists() and displaced.exists():
            output.rename(replacement)
            displaced.rename(output)

    leaked = tuple(
        candidate
        for candidate in replacement.rglob("*")
        if candidate.is_file() and candidate.read_bytes() == b"synthetic-secret"
    ) if replacement.exists() else ()
    assert attack_happened
    assert not leaked


def test_capability_reads_final_members_through_locked_parent(
    tmp_path: Path,
) -> None:
    approved = tmp_path / "private"
    output = approved / "output"
    parent = output / "e1a4"
    displaced = approved / "authenticated-output"
    replacement = approved / "replacement-output"
    approved.mkdir()
    output.mkdir()
    observed: dict[str, bytes] | None = None

    try:
        with publication.authenticated_publication_directory(
            approved_private_root=approved,
            publication_parent=parent,
            lock_name=".publish.lock",
        ) as bound:
            _publish_single_member(bound, content=b"authenticated")
            try:
                output.rename(displaced)
            except OSError:
                pass
            else:
                replacement_member = parent / "v1" / "sealed" / "payload.bin"
                replacement_member.parent.mkdir(parents=True)
                replacement_member.write_bytes(b"replacement")
                try:
                    observed = bound.read_exact_tree(
                        "v1", {"sealed": frozenset({"payload.bin"})}
                    )
                except publication.PrivateArtifactPublicationError:
                    pass
    finally:
        if output.exists() and displaced.exists():
            output.rename(replacement)
            displaced.rename(output)

    assert observed in (None, {"sealed/payload.bin": b"authenticated"})


def test_windows_creation_flushes_each_parent_before_releasing_it() -> None:
    events: list[tuple[str, object]] = []
    next_handle = 10

    class WindowsAPI:
        def open_directory_component(
            self,
            parent: object,
            name: str,
            *,
            create: bool,
            writable: bool,
        ) -> publication._OpenedWindowsDirectory:
            nonlocal next_handle
            events.append(("open", (parent, name, create, writable)))
            next_handle += 1
            return publication._OpenedWindowsDirectory(next_handle, created=True)

        def flush_handle(self, handle: object) -> None:
            events.append(("flush", handle))

        def close_handle(self, handle: object) -> None:
            events.append(("close", handle))

    final = publication._walk_windows_components(
        WindowsAPI(),
        10,
        ("private", "output", "e1a4"),
        create=True,
        writable=True,
    )

    assert final == 13
    assert events == [
        ("open", (10, "private", True, True)),
        ("flush", 10),
        ("close", 10),
        ("open", (11, "output", True, True)),
        ("flush", 11),
        ("close", 11),
        ("open", (12, "e1a4", True, True)),
        ("flush", 12),
        ("close", 12),
    ]


def test_windows_constructor_failure_closes_each_handle_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    approved = tmp_path / "private"
    approved.mkdir()
    api = SimpleNamespace(
        closed=[],
        close_handle=lambda handle: api.closed.append(handle),
        validate_handle=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("invalid handle")
        ),
    )
    acquired = publication._WindowsAcquiredHandle(api=api, value=41)

    monkeypatch.setattr(publication, "_platform_name", lambda: "nt")
    monkeypatch.setattr(
        publication,
        "_open_authenticated_parent",
        lambda *_args, **_kwargs: acquired,
    )
    with pytest.raises(
        publication.PrivateArtifactPublicationError,
        match="^PRIVATE_ARTIFACT_ACQUISITION_FAILED$",
    ):
        with publication.authenticated_publication_directory(
            approved_private_root=approved,
            publication_parent=approved / "output",
            lock_name=".publish.lock",
        ):
            pytest.fail("constructor failure unexpectedly yielded")

    assert api.closed == [41]


def test_posix_creation_is_component_relative_nofollow_and_parent_synced() -> None:
    calls: list[tuple[object, ...]] = []

    class ComponentOS:
        O_RDONLY = 0x01
        O_DIRECTORY = 0x02
        O_NOFOLLOW = 0x04
        O_CLOEXEC = 0x08

        def open(self, name: str, flags: int, *, dir_fd: int) -> int:
            calls.append(("open", name, flags, dir_fd))
            if sum(call[0] == "open" for call in calls) == 1:
                raise FileNotFoundError
            return 22

        def mkdir(self, name: str, mode: int, *, dir_fd: int) -> None:
            calls.append(("mkdir", name, mode, dir_fd))

        def fsync(self, descriptor: int) -> None:
            calls.append(("fsync", descriptor))

        def fstat(self, descriptor: int) -> object:
            calls.append(("fstat", descriptor))
            return SimpleNamespace(st_mode=stat.S_IFDIR | 0o700)

        def close(self, descriptor: int) -> None:
            calls.append(("close", descriptor))

    descriptor = publication._posix_open_directory_component(
        ComponentOS(), 11, "output", create=True
    )

    assert descriptor == 22
    assert calls[:4] == [
        ("open", "output", 0x0F, 11),
        ("mkdir", "output", 0o700, 11),
        ("fsync", 11),
        ("open", "output", 0x0F, 11),
    ]
    assert calls[4] == ("fstat", 22)


def test_capability_preserves_residue_and_never_deletes_untrusted_entries(
    tmp_path: Path,
) -> None:
    approved = tmp_path / "private"
    parent = approved / "output"
    residue = parent / ".v1.synthetic.tmp"
    residue.mkdir(parents=True)
    sentinel = residue / "preserve.bin"
    sentinel.write_bytes(b"preserve")

    with publication.authenticated_publication_directory(
        approved_private_root=approved,
        publication_parent=parent,
        lock_name=".publish.lock",
    ) as bound:
        with pytest.raises(
            publication.PrivateArtifactPublicationError,
            match="^PRIVATE_ARTIFACT_RESIDUE_PRESENT$",
        ):
            bound.ensure_no_staging(".v1.", ".tmp")

    assert sentinel.read_bytes() == b"preserve"


def test_capability_fails_closed_without_safe_relative_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    approved = tmp_path / "private"
    parent = approved / "output"
    approved.mkdir()

    with publication.authenticated_publication_directory(
        approved_private_root=approved,
        publication_parent=parent,
        lock_name=".publish.lock",
    ) as bound:
        staging = bound.create_staging(".v1.", ".tmp")
        staging.write_exclusive("payload.bin", b"synthetic")
        staging.sync_root()

        def unsupported(*_args: object, **_kwargs: object) -> None:
            raise OSError(errno.ENOTSUP, "relative no-replace rename unavailable")

        monkeypatch.setattr(type(bound), "_rename_no_replace", unsupported)
        with pytest.raises(
            publication.PrivateArtifactPublicationError,
            match="^PRIVATE_ARTIFACT_RENAME_UNAVAILABLE$",
        ):
            bound.publish_no_replace(staging, "v1")

    assert (parent / staging.name / "payload.bin").read_bytes() == b"synthetic"
    assert not (parent / "v1").exists()
