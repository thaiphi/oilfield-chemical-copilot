"""Authenticated, handle-relative publication of private artifact trees."""

from __future__ import annotations

import errno
import hashlib
import os
import secrets
import stat
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import ContextManager, Iterator, Mapping, TypeAlias

ArtifactLayout: TypeAlias = Mapping[str, frozenset[str]]

_PATH_REJECTED = "PRIVATE_ARTIFACT_PATH_REJECTED"
_ACQUISITION_FAILED = "PRIVATE_ARTIFACT_ACQUISITION_FAILED"
_LOCK_FAILED = "PRIVATE_ARTIFACT_LOCK_FAILED"
_OPERATION_FAILED = "PRIVATE_ARTIFACT_OPERATION_FAILED"
_RESIDUE_PRESENT = "PRIVATE_ARTIFACT_RESIDUE_PRESENT"
_TREE_INVALID = "PRIVATE_ARTIFACT_TREE_INVALID"
_RENAME_UNAVAILABLE = "PRIVATE_ARTIFACT_RENAME_UNAVAILABLE"
_CLEANUP_FAILED = "PRIVATE_ARTIFACT_CLEANUP_FAILED"
_FIXED_CODES = frozenset(
    {
        _PATH_REJECTED,
        _ACQUISITION_FAILED,
        _LOCK_FAILED,
        _OPERATION_FAILED,
        _RESIDUE_PRESENT,
        _TREE_INVALID,
        _RENAME_UNAVAILABLE,
        _CLEANUP_FAILED,
    }
)


class PrivateArtifactPublicationError(RuntimeError):
    """A fixed-code failure at the private-artifact filesystem boundary."""

    def __init__(self, code: str) -> None:
        if code not in _FIXED_CODES:
            raise ValueError("unknown private-artifact publication error code")
        self.code = code
        super().__init__(code)


def _publication_error(code: str, cause: BaseException) -> PrivateArtifactPublicationError:
    error = PrivateArtifactPublicationError(code)
    error.__cause__ = cause
    return error


def _component(name: str) -> str:
    if (
        type(name) is not str
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "\x00" in name
    ):
        raise ValueError("unsafe relative component")
    return name


def _relative_parts(name: str) -> tuple[str, ...]:
    if type(name) is not str or not name or "\\" in name or "\x00" in name:
        raise ValueError("unsafe relative name")
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts:
        raise ValueError("unsafe relative name")
    parts = tuple(path.parts)
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("unsafe relative name")
    return parts


def _platform_name() -> str:
    return os.name


def _validate_paths(
    approved_private_root: Path, publication_parent: Path
) -> tuple[Path, tuple[str, ...]]:
    if not isinstance(approved_private_root, Path) or not isinstance(publication_parent, Path):
        raise ValueError("publication paths must be Path instances")
    root = approved_private_root.absolute()
    parent = publication_parent.absolute()
    relative = parent.relative_to(root)
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("unsafe publication parent")
    repository_root = Path(__file__).absolute().parents[3]
    if os.path.normcase(str(root)) == os.path.normcase(str(repository_root)):
        raise ValueError("public repository cannot be a private root")
    if _platform_name() == "nt":
        anchor = Path(root.anchor)
        if (
            not root.drive
            or root.drive.startswith("\\")
            or root.anchor.casefold() != f"{root.drive}\\".casefold()
            or not root.relative_to(anchor).parts
        ):
            raise ValueError("unsafe Windows private root")
    elif root.anchor != os.path.sep or not root.parts[1:]:
        raise ValueError("unsafe POSIX private root")
    return root, tuple(relative.parts)


def _required_posix_flag(os_api: object, name: str) -> int:
    value = getattr(os_api, name, None)
    if type(value) is not int or value == 0:
        raise OSError(errno.ENOTSUP, "required relative primitive unavailable")
    return value


def _posix_directory_flags(os_api: object) -> int:
    return (
        os_api.O_RDONLY
        | _required_posix_flag(os_api, "O_DIRECTORY")
        | _required_posix_flag(os_api, "O_NOFOLLOW")
        | getattr(os_api, "O_CLOEXEC", 0)
    )


def _posix_open_directory_component(
    os_api: object,
    parent_descriptor: int,
    component: str,
    *,
    create: bool,
) -> int:
    _component(component)
    flags = _posix_directory_flags(os_api)
    try:
        descriptor = os_api.open(component, flags, dir_fd=parent_descriptor)
    except FileNotFoundError:
        if not create:
            raise
        try:
            os_api.mkdir(component, 0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            pass
        else:
            os_api.fsync(parent_descriptor)
        descriptor = os_api.open(component, flags, dir_fd=parent_descriptor)
    try:
        if not stat.S_ISDIR(os_api.fstat(descriptor).st_mode):
            raise OSError(errno.ENOTDIR, "relative object is not a directory")
    except BaseException:
        os_api.close(descriptor)
        raise
    return descriptor


@dataclass(frozen=True)
class _PosixAcquiredHandle:
    os_api: object
    value: int


def _acquire_posix_publication_parent(
    root: Path, relative: tuple[str, ...], *, os_api: object
) -> _PosixAcquiredHandle:
    descriptor = os_api.open(os.path.sep, _posix_directory_flags(os_api))
    try:
        for component in root.parts[1:]:
            child = _posix_open_directory_component(
                os_api, descriptor, component, create=False
            )
            parent = descriptor
            descriptor = child
            os_api.close(parent)
        for component in relative:
            child = _posix_open_directory_component(
                os_api, descriptor, component, create=True
            )
            parent = descriptor
            descriptor = child
            os_api.close(parent)
        return _PosixAcquiredHandle(os_api=os_api, value=descriptor)
    except BaseException:
        os_api.close(descriptor)
        raise


def _rename_no_replace_at(descriptor: int, staged_name: str, final_name: str) -> None:
    import ctypes

    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        rename = getattr(library, "renameat2", None)
        flag = 1
    elif sys.platform == "darwin":
        rename = getattr(library, "renameatx_np", None)
        flag = 0x00000004
    else:
        rename = None
        flag = 0
    if rename is None:
        raise OSError(errno.ENOTSUP, "exclusive relative rename unavailable")
    rename.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    rename.restype = ctypes.c_int
    result = rename(
        descriptor,
        os.fsencode(staged_name),
        descriptor,
        os.fsencode(final_name),
        flag,
    )
    if result == 0:
        return
    number = ctypes.get_errno()
    if number == errno.EEXIST:
        raise FileExistsError(number, "destination exists")
    raise OSError(number, "relative publication rename failed")


@dataclass(frozen=True)
class _OpenedWindowsDirectory:
    value: object
    created: bool


@dataclass(frozen=True)
class _WindowsAcquiredHandle:
    api: object
    value: object


def _walk_windows_components(
    api: object,
    descriptor: object,
    components: tuple[str, ...],
    *,
    create: bool,
    writable: bool,
) -> object:
    for component in components:
        opened: _OpenedWindowsDirectory | None = None
        close_attempted = False
        try:
            opened = api.open_directory_component(
                descriptor,
                _component(component),
                create=create,
                writable=writable,
            )
            if opened.created:
                api.flush_handle(descriptor)
            close_attempted = True
            api.close_handle(descriptor)
            descriptor = opened.value
            opened = None
        except BaseException:
            if opened is not None:
                try:
                    api.close_handle(opened.value)
                except BaseException:
                    pass
            if not close_attempted:
                try:
                    api.close_handle(descriptor)
                except BaseException:
                    pass
            raise
    return descriptor


@dataclass(frozen=True)
class _WindowsSecurity:
    attributes: object
    descriptor: object


class _NativeWindowsMutex:
    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        class SID_AND_ATTRIBUTES(ctypes.Structure):
            _fields_ = (("Sid", wintypes.LPVOID), ("Attributes", wintypes.DWORD))

        class TOKEN_USER(ctypes.Structure):
            _fields_ = (("User", SID_AND_ATTRIBUTES),)

        class SECURITY_ATTRIBUTES(ctypes.Structure):
            _fields_ = (
                ("nLength", wintypes.DWORD),
                ("lpSecurityDescriptor", wintypes.LPVOID),
                ("bInheritHandle", wintypes.BOOL),
            )

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = (
            ctypes.POINTER(SECURITY_ATTRIBUTES),
            wintypes.BOOL,
            wintypes.LPCWSTR,
        )
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.ReleaseMutex.argtypes = (wintypes.HANDLE,)
        kernel32.ReleaseMutex.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.GetCurrentProcess.argtypes = ()
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.LocalFree.argtypes = (wintypes.HLOCAL,)
        kernel32.LocalFree.restype = wintypes.HLOCAL
        advapi32.OpenProcessToken.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.HANDLE),
        )
        advapi32.OpenProcessToken.restype = wintypes.BOOL
        advapi32.GetTokenInformation.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.LPDWORD,
        )
        advapi32.GetTokenInformation.restype = wintypes.BOOL
        advapi32.ConvertSidToStringSidW.argtypes = (
            wintypes.LPVOID,
            ctypes.POINTER(wintypes.LPWSTR),
        )
        advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
        advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.LPVOID),
            wintypes.LPDWORD,
        )
        advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = (
            wintypes.BOOL
        )
        self._ctypes = ctypes
        self._kernel32 = kernel32
        self._advapi32 = advapi32
        self._TOKEN_USER = TOKEN_USER
        self._SECURITY_ATTRIBUTES = SECURITY_ATTRIBUTES

    def owner_sid(self) -> str:
        token = self._ctypes.c_void_p()
        if not self._advapi32.OpenProcessToken(
            self._kernel32.GetCurrentProcess(), 0x0008, self._ctypes.byref(token)
        ):
            raise OSError(self._ctypes.get_last_error(), "token unavailable")
        try:
            required = self._ctypes.c_ulong()
            self._advapi32.GetTokenInformation(token, 1, None, 0, required)
            if required.value == 0:
                raise OSError(self._ctypes.get_last_error(), "token query failed")
            buffer = self._ctypes.create_string_buffer(required.value)
            if not self._advapi32.GetTokenInformation(
                token,
                1,
                self._ctypes.cast(buffer, self._ctypes.c_void_p),
                required.value,
                required,
            ):
                raise OSError(self._ctypes.get_last_error(), "token query failed")
            token_user = self._ctypes.cast(
                buffer, self._ctypes.POINTER(self._TOKEN_USER)
            ).contents
            sid_text = self._ctypes.c_wchar_p()
            if not self._advapi32.ConvertSidToStringSidW(
                token_user.User.Sid, self._ctypes.byref(sid_text)
            ):
                raise OSError(self._ctypes.get_last_error(), "SID conversion failed")
            try:
                if sid_text.value is None:
                    raise OSError(errno.EIO, "SID conversion failed")
                return sid_text.value
            finally:
                if self._kernel32.LocalFree(sid_text):
                    raise OSError(self._ctypes.get_last_error(), "SID free failed")
        finally:
            if not self._kernel32.CloseHandle(token):
                raise OSError(self._ctypes.get_last_error(), "token close failed")

    def build_security_attributes(self, policy: str) -> _WindowsSecurity:
        descriptor = self._ctypes.c_void_p()
        converted = self._advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            policy, 1, self._ctypes.byref(descriptor), None
        )
        if not converted:
            raise OSError(
                self._ctypes.get_last_error(), "security descriptor conversion failed"
            )
        attributes = self._SECURITY_ATTRIBUTES(
            self._ctypes.sizeof(self._SECURITY_ATTRIBUTES), descriptor, False
        )
        return _WindowsSecurity(attributes=attributes, descriptor=descriptor)

    def free_security_descriptor(self, security: _WindowsSecurity) -> None:
        if self._kernel32.LocalFree(security.descriptor):
            raise OSError(self._ctypes.get_last_error(), "security descriptor free failed")

    def create_mutex(self, name: str, attributes: object) -> int:
        handle = self._kernel32.CreateMutexW(
            self._ctypes.byref(attributes), False, name
        )
        if not handle:
            raise OSError(self._ctypes.get_last_error(), "mutex creation failed")
        return int(handle)

    def wait(self, handle: int, timeout_ms: int) -> int:
        return int(self._kernel32.WaitForSingleObject(handle, timeout_ms))

    def release_mutex(self, handle: int) -> None:
        if not self._kernel32.ReleaseMutex(handle):
            raise OSError(self._ctypes.get_last_error(), "mutex release failed")

    def close_handle(self, handle: int) -> None:
        if not self._kernel32.CloseHandle(handle):
            raise OSError(self._ctypes.get_last_error(), "mutex close failed")


def _windows_mutex_policy(owner_sid: str) -> str:
    parts = owner_sid.split("-")
    if len(parts) < 3 or parts[0] != "S" or not all(
        part.isdecimal() for part in parts[1:]
    ):
        raise OSError(errno.EINVAL, "invalid owner SID")
    return (
        f"O:{owner_sid}D:P"
        f"(A;;0x001F0001;;;{owner_sid})"
        "(A;;0x001F0001;;;SY)"
    )


class _NativeWindowsSealReader:
    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        class UNICODE_STRING(ctypes.Structure):
            _fields_ = (
                ("Length", wintypes.USHORT),
                ("MaximumLength", wintypes.USHORT),
                ("Buffer", wintypes.LPWSTR),
            )

        class OBJECT_ATTRIBUTES(ctypes.Structure):
            _fields_ = (
                ("Length", wintypes.ULONG),
                ("RootDirectory", wintypes.HANDLE),
                ("ObjectName", ctypes.POINTER(UNICODE_STRING)),
                ("Attributes", wintypes.ULONG),
                ("SecurityDescriptor", wintypes.LPVOID),
                ("SecurityQualityOfService", wintypes.LPVOID),
            )

        class IO_STATUS_BLOCK(ctypes.Structure):
            _fields_ = (
                ("Status", ctypes.c_ssize_t),
                ("Information", ctypes.c_size_t),
            )

        class FILE_STANDARD_INFO(ctypes.Structure):
            _fields_ = (
                ("AllocationSize", ctypes.c_longlong),
                ("EndOfFile", ctypes.c_longlong),
                ("NumberOfLinks", wintypes.DWORD),
                ("DeletePending", wintypes.BOOLEAN),
                ("Directory", wintypes.BOOLEAN),
            )

        class FILE_BASIC_INFO(ctypes.Structure):
            _fields_ = (
                ("CreationTime", ctypes.c_longlong),
                ("LastAccessTime", ctypes.c_longlong),
                ("LastWriteTime", ctypes.c_longlong),
                ("ChangeTime", ctypes.c_longlong),
                ("FileAttributes", wintypes.DWORD),
            )

        class FILE_ATTRIBUTE_TAG_INFO(ctypes.Structure):
            _fields_ = (
                ("FileAttributes", wintypes.DWORD),
                ("ReparseTag", wintypes.DWORD),
            )

        class FILE_ID_INFO(ctypes.Structure):
            _fields_ = (
                ("VolumeSerialNumber", ctypes.c_ulonglong),
                ("FileId", ctypes.c_ubyte * 16),
            )

        class FILE_ID_BOTH_DIR_INFO(ctypes.Structure):
            _fields_ = (
                ("NextEntryOffset", wintypes.DWORD),
                ("FileIndex", wintypes.DWORD),
                ("CreationTime", ctypes.c_longlong),
                ("LastAccessTime", ctypes.c_longlong),
                ("LastWriteTime", ctypes.c_longlong),
                ("ChangeTime", ctypes.c_longlong),
                ("EndOfFile", ctypes.c_longlong),
                ("AllocationSize", ctypes.c_longlong),
                ("FileAttributes", wintypes.DWORD),
                ("FileNameLength", wintypes.DWORD),
                ("EaSize", wintypes.DWORD),
                ("ShortNameLength", ctypes.c_ubyte),
                ("ShortName", wintypes.WCHAR * 12),
                ("FileId", ctypes.c_longlong),
                ("FileName", wintypes.WCHAR * 1),
            )

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
        kernel32.CreateFileW.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        kernel32.CreateFileW.restype = wintypes.HANDLE
        kernel32.GetFileType.argtypes = (wintypes.HANDLE,)
        kernel32.GetFileType.restype = wintypes.DWORD
        kernel32.GetFileInformationByHandleEx.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        )
        kernel32.GetFileInformationByHandleEx.restype = wintypes.BOOL
        kernel32.ReadFile.argtypes = (
            wintypes.HANDLE,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.LPDWORD,
            wintypes.LPVOID,
        )
        kernel32.ReadFile.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        ntdll.NtCreateFile.argtypes = (
            ctypes.POINTER(wintypes.HANDLE),
            wintypes.DWORD,
            ctypes.POINTER(OBJECT_ATTRIBUTES),
            ctypes.POINTER(IO_STATUS_BLOCK),
            ctypes.POINTER(ctypes.c_longlong),
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
        )
        ntdll.NtCreateFile.restype = ctypes.c_long
        ntdll.RtlNtStatusToDosError.argtypes = (ctypes.c_long,)
        ntdll.RtlNtStatusToDosError.restype = wintypes.ULONG
        self._ctypes = ctypes
        self._wintypes = wintypes
        self._kernel32 = kernel32
        self._ntdll = ntdll
        self._UNICODE_STRING = UNICODE_STRING
        self._OBJECT_ATTRIBUTES = OBJECT_ATTRIBUTES
        self._IO_STATUS_BLOCK = IO_STATUS_BLOCK
        self._FILE_STANDARD_INFO = FILE_STANDARD_INFO
        self._FILE_BASIC_INFO = FILE_BASIC_INFO
        self._FILE_ATTRIBUTE_TAG_INFO = FILE_ATTRIBUTE_TAG_INFO
        self._FILE_ID_INFO = FILE_ID_INFO
        self._directory_info = FILE_ID_BOTH_DIR_INFO
        self._directory_name_offset = FILE_ID_BOTH_DIR_INFO.FileName.offset

    def _query(self, handle: object, info_class: int, result: object) -> None:
        if not self._kernel32.GetFileInformationByHandleEx(
            handle,
            info_class,
            self._ctypes.byref(result),
            self._ctypes.sizeof(result),
        ):
            raise OSError(self._ctypes.get_last_error(), "file information unavailable")

    def _validate_handle(
        self, handle: object, *, directory: bool
    ) -> tuple[object, ...]:
        if self._kernel32.GetFileType(handle) != 1:
            raise OSError(errno.EPERM, "unsafe publication object")
        attributes = self._FILE_ATTRIBUTE_TAG_INFO()
        standard = self._FILE_STANDARD_INFO()
        basic = self._FILE_BASIC_INFO()
        identity = self._FILE_ID_INFO()
        self._query(handle, 9, attributes)
        self._query(handle, 1, standard)
        self._query(handle, 0, basic)
        self._query(handle, 18, identity)
        if (
            attributes.FileAttributes & 0x00000400
            or bool(standard.Directory) != directory
            or (directory and not attributes.FileAttributes & 0x00000010)
            or (not directory and standard.NumberOfLinks != 1)
        ):
            raise OSError(errno.EPERM, "unsafe publication object")
        return (
            int(identity.VolumeSerialNumber),
            bytes(identity.FileId),
            int(standard.EndOfFile),
            int(basic.LastWriteTime),
            int(basic.ChangeTime),
        )

    def open_directory(self, path: Path) -> int:
        handle = self._kernel32.CreateFileW(
            str(path), 0x00100081, 0x00000007, None, 3, 0x02200000, None
        )
        if handle == self._ctypes.c_void_p(-1).value:
            raise OSError(self._ctypes.get_last_error(), "directory unavailable")
        value = int(handle)
        try:
            self._validate_handle(value, directory=True)
        except BaseException:
            self.close_handle(value)
            raise
        return value

    def directory_entries(self, handle: object) -> set[str]:
        names: set[str] = set()
        while True:
            buffer = self._ctypes.create_string_buffer(65536)
            if not self._kernel32.GetFileInformationByHandleEx(handle, 10, buffer, len(buffer)):
                number = self._ctypes.get_last_error()
                if number == 18:
                    return names
                raise OSError(number, "directory enumeration failed")
            offset = 0
            while True:
                entry = self._directory_info.from_buffer(buffer, offset)
                name = self._ctypes.wstring_at(
                    self._ctypes.addressof(buffer) + offset + self._directory_name_offset,
                    entry.FileNameLength // 2,
                )
                if name not in {".", ".."}:
                    names.add(name)
                if entry.NextEntryOffset == 0:
                    break
                offset += entry.NextEntryOffset

    def read_member(self, handle: object) -> bytes:
        chunks: list[bytes] = []
        while True:
            buffer = self._ctypes.create_string_buffer(65536)
            read = self._wintypes.DWORD()
            if not self._kernel32.ReadFile(
                handle, buffer, len(buffer), self._ctypes.byref(read), None
            ):
                raise OSError(self._ctypes.get_last_error(), "member read failed")
            if read.value == 0:
                return b"".join(chunks)
            chunks.append(buffer.raw[: read.value])

    def close_handle(self, handle: object) -> None:
        if not self._kernel32.CloseHandle(handle):
            raise OSError(self._ctypes.get_last_error(), "handle close failed")


class _NativeWindowsPublicationAPI(_NativeWindowsSealReader):
    def __init__(self) -> None:
        super().__init__()

        class OVERLAPPED(self._ctypes.Structure):
            _fields_ = (
                ("Internal", self._ctypes.c_size_t),
                ("InternalHigh", self._ctypes.c_size_t),
                ("Offset", self._wintypes.DWORD),
                ("OffsetHigh", self._wintypes.DWORD),
                ("hEvent", self._wintypes.HANDLE),
            )

        self._OVERLAPPED = OVERLAPPED
        self._kernel32.WriteFile.argtypes = (
            self._wintypes.HANDLE,
            self._wintypes.LPVOID,
            self._wintypes.DWORD,
            self._wintypes.LPDWORD,
            self._wintypes.LPVOID,
        )
        self._kernel32.WriteFile.restype = self._wintypes.BOOL
        self._kernel32.FlushFileBuffers.argtypes = (self._wintypes.HANDLE,)
        self._kernel32.FlushFileBuffers.restype = self._wintypes.BOOL
        self._ntdll.NtSetInformationFile.argtypes = (
            self._wintypes.HANDLE,
            self._ctypes.POINTER(self._IO_STATUS_BLOCK),
            self._wintypes.LPVOID,
            self._wintypes.ULONG,
            self._ctypes.c_int,
        )
        self._ntdll.NtSetInformationFile.restype = self._ctypes.c_long

    def validate_handle(self, handle: object, *, directory: bool) -> tuple[object, ...]:
        return self._validate_handle(handle, directory=directory)

    def open_anchor(self, path: Path) -> int:
        return self.open_directory(path)

    def _attributes(self, parent: object, name: str) -> tuple[object, ...]:
        name_buffer = self._ctypes.create_unicode_buffer(_component(name))
        name_length = len(name.encode("utf-16-le"))
        unicode_name = self._UNICODE_STRING(
            name_length,
            name_length + 2,
            self._ctypes.cast(name_buffer, self._wintypes.LPWSTR),
        )
        attributes = self._OBJECT_ATTRIBUTES(
            self._ctypes.sizeof(self._OBJECT_ATTRIBUTES),
            parent,
            self._ctypes.pointer(unicode_name),
            0x00000040,
            None,
            None,
        )
        return name_buffer, unicode_name, attributes

    def _open_relative(
        self,
        parent: object,
        name: str,
        *,
        directory: bool,
        disposition: int,
        writable: bool,
    ) -> int:
        buffers = self._attributes(parent, name)
        status_block = self._IO_STATUS_BLOCK()
        handle = self._wintypes.HANDLE()
        status = self._ntdll.NtCreateFile(
            self._ctypes.byref(handle),
            0x001F01FF if writable else 0x00100081,
            self._ctypes.byref(buffers[2]),
            self._ctypes.byref(status_block),
            None,
            0x00000010 if directory else 0x00000080,
            0x00000007,
            disposition,
            0x00200021 if directory else 0x00200060,
            None,
            0,
        )
        if status < 0:
            number = int(self._ntdll.RtlNtStatusToDosError(status))
            raise OSError(number, "relative publication open failed")
        value = int(handle.value)
        try:
            self.validate_handle(value, directory=directory)
        except BaseException:
            self.close_handle(value)
            raise
        return value

    def open_directory_component(
        self,
        parent: object,
        name: str,
        *,
        create: bool,
        writable: bool,
    ) -> _OpenedWindowsDirectory:
        try:
            value = self._open_relative(
                parent, name, directory=True, disposition=1, writable=writable
            )
            return _OpenedWindowsDirectory(value, created=False)
        except OSError as error:
            if not create or error.errno not in {2, 3}:
                raise
        try:
            value = self._open_relative(
                parent, name, directory=True, disposition=2, writable=writable
            )
            return _OpenedWindowsDirectory(value, created=True)
        except OSError as error:
            if error.errno not in {80, 183}:
                raise
        value = self._open_relative(
            parent, name, directory=True, disposition=1, writable=writable
        )
        return _OpenedWindowsDirectory(value, created=False)

    def create_directory_relative(self, parent: object, name: str) -> int:
        return self._open_relative(
            parent, name, directory=True, disposition=2, writable=True
        )

    def open_directory_relative(self, parent: object, name: str) -> int:
        return self._open_relative(
            parent, name, directory=True, disposition=1, writable=False
        )

    def create_file_relative(self, parent: object, name: str) -> int:
        return self._open_relative(
            parent, name, directory=False, disposition=2, writable=True
        )

    def open_file_relative(
        self, parent: object, name: str, *, writable: bool = False
    ) -> int:
        return self._open_relative(
            parent, name, directory=False, disposition=1, writable=writable
        )

    def write_all(self, handle: object, content: bytes) -> None:
        offset = 0
        while offset < len(content):
            chunk = content[offset : offset + 65536]
            buffer = self._ctypes.create_string_buffer(chunk)
            written = self._wintypes.DWORD()
            if not self._kernel32.WriteFile(
                handle,
                buffer,
                len(chunk),
                self._ctypes.byref(written),
                None,
            ):
                raise OSError(self._ctypes.get_last_error(), "member write failed")
            if written.value < 1:
                raise OSError(errno.EIO, "member write failed")
            offset += written.value

    def flush_handle(self, handle: object) -> None:
        if not self._kernel32.FlushFileBuffers(handle):
            raise OSError(self._ctypes.get_last_error(), "directory sync failed")

    def rename_no_replace(self, source: object, parent: object, final_name: str) -> None:
        class FILE_RENAME_INFORMATION(self._ctypes.Structure):
            _fields_ = (
                ("ReplaceIfExists", self._wintypes.BOOLEAN),
                ("RootDirectory", self._wintypes.HANDLE),
                ("FileNameLength", self._wintypes.DWORD),
                ("FileName", self._wintypes.WCHAR * 1),
            )

        encoded = _component(final_name).encode("utf-16-le")
        offset = FILE_RENAME_INFORMATION.FileName.offset
        buffer = self._ctypes.create_string_buffer(offset + len(encoded))
        information = self._ctypes.cast(
            buffer, self._ctypes.POINTER(FILE_RENAME_INFORMATION)
        ).contents
        information.ReplaceIfExists = False
        information.RootDirectory = parent
        information.FileNameLength = len(encoded)
        self._ctypes.memmove(
            self._ctypes.addressof(buffer) + offset, encoded, len(encoded)
        )
        status_block = self._IO_STATUS_BLOCK()
        status = self._ntdll.NtSetInformationFile(
            source,
            self._ctypes.byref(status_block),
            buffer,
            len(buffer),
            10,
        )
        if status < 0:
            number = int(self._ntdll.RtlNtStatusToDosError(status))
            raise OSError(number, "relative publication rename failed")

def _windows_api() -> _NativeWindowsPublicationAPI:
    return _NativeWindowsPublicationAPI()


def _acquire_windows_publication_parent(
    root: Path, relative: tuple[str, ...], *, api: object
) -> _WindowsAcquiredHandle:
    anchor = Path(root.anchor)
    descriptor = api.open_anchor(anchor)
    root_components = tuple(root.relative_to(anchor).parts)
    for index, component in enumerate(root_components):
        descriptor = _walk_windows_components(
            api,
            descriptor,
            (component,),
            create=False,
            writable=index == len(root_components) - 1,
        )
    descriptor = _walk_windows_components(
        api,
        descriptor,
        relative,
        create=True,
        writable=True,
    )
    return _WindowsAcquiredHandle(api=api, value=descriptor)


def _open_authenticated_parent(root: Path, relative: tuple[str, ...]) -> object:
    if _platform_name() == "nt":
        return _acquire_windows_publication_parent(root, relative, api=_windows_api())
    return _acquire_posix_publication_parent(root, relative, os_api=os)


class BoundStagingDirectory:
    """A staging tree whose names are resolved only beneath its bound handle."""

    def __init__(
        self,
        owner: AuthenticatedPublicationDirectory,
        name: str,
        root: object,
    ) -> None:
        self._owner = owner
        self.name = name
        self._root = root
        self._directories: dict[tuple[str, ...], object] = {(): root}
        self._closed = False
        self._published = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise OSError(errno.EBADF, "staging capability is closed")

    def mkdir(self, relative_name: str) -> None:
        try:
            self._ensure_open()
            parts = _relative_parts(relative_name)
            parent = self._directories.get(parts[:-1])
            if parent is None or parts in self._directories:
                raise OSError(errno.ENOENT, "staging parent unavailable")
            child = self._owner._create_directory(parent, parts[-1])
            try:
                self._directories[parts] = child
            except BaseException:
                self._owner._close_handle(child)
                raise
        except PrivateArtifactPublicationError:
            raise
        except BaseException as error:
            raise _publication_error(_OPERATION_FAILED, error)

    def write_exclusive(self, relative_name: str, content: bytes) -> None:
        try:
            self._ensure_open()
            if type(content) is not bytes:
                raise TypeError("publication content must be bytes")
            parts = _relative_parts(relative_name)
            parent = self._directories.get(parts[:-1])
            if parent is None:
                raise OSError(errno.ENOENT, "staging parent unavailable")
            self._owner._write_exclusive(parent, parts[-1], content)
        except PrivateArtifactPublicationError:
            raise
        except BaseException as error:
            raise _publication_error(_OPERATION_FAILED, error)

    def sync_directory(self, relative_name: str) -> None:
        try:
            self._ensure_open()
            self._owner._sync(self._directories[_relative_parts(relative_name)])
        except PrivateArtifactPublicationError:
            raise
        except BaseException as error:
            raise _publication_error(_OPERATION_FAILED, error)

    def sync_root(self) -> None:
        try:
            self._ensure_open()
            self._owner._sync(self._root)
        except PrivateArtifactPublicationError:
            raise
        except BaseException as error:
            raise _publication_error(_OPERATION_FAILED, error)

    def _close_children(self) -> None:
        error: BaseException | None = None
        child_paths = sorted(
            (parts for parts in self._directories if parts), key=len, reverse=True
        )
        for parts in child_paths:
            handle = self._directories.pop(parts)
            try:
                self._owner._close_handle(handle)
            except BaseException as close_error:
                error = error or close_error
        if error is not None:
            raise error

    def _close(self) -> None:
        if self._closed:
            return
        error: BaseException | None = None
        try:
            self._close_children()
        except BaseException as close_error:
            error = close_error
        try:
            self._owner._close_handle(self._root)
        except BaseException as close_error:
            error = error or close_error
        self._directories.clear()
        self._closed = True
        if error is not None:
            raise error


class AuthenticatedPublicationDirectory:
    """A locked publication parent anchored by an authenticated directory handle."""

    def __init__(self, descriptor: object) -> None:
        self._descriptor = descriptor
        self._staging: list[BoundStagingDirectory] = []
        self._closed = False

    def _list(self, directory: object) -> set[str]:
        raise NotImplementedError

    def _open_directory(self, parent: object, name: str) -> object:
        raise NotImplementedError

    def _create_directory(self, parent: object, name: str) -> object:
        raise NotImplementedError

    def _write_exclusive(self, parent: object, name: str, content: bytes) -> None:
        raise NotImplementedError

    def _read_member(self, parent: object, name: str) -> bytes:
        raise NotImplementedError

    def _sync(self, descriptor: object) -> None:
        raise NotImplementedError

    def _close_handle(self, descriptor: object) -> None:
        raise NotImplementedError

    def _acquire_lock(self, lock_name: str) -> object:
        raise NotImplementedError

    def _release_lock(self, lock: object) -> None:
        raise NotImplementedError

    def _rename_no_replace(
        self, staging: BoundStagingDirectory, final_name: str
    ) -> None:
        raise NotImplementedError

    def ensure_no_staging(self, prefix: str, suffix: str) -> None:
        try:
            if (
                type(prefix) is not str
                or type(suffix) is not str
                or not prefix
                or not suffix
                or any(item in prefix + suffix for item in ("/", "\\", "\x00"))
            ):
                raise ValueError("invalid staging markers")
            residue = any(
                name.startswith(prefix) and name.endswith(suffix)
                for name in self._list(self._descriptor)
            )
            if residue:
                raise PrivateArtifactPublicationError(_RESIDUE_PRESENT)
        except PrivateArtifactPublicationError:
            raise
        except BaseException as error:
            raise _publication_error(_OPERATION_FAILED, error)

    def final_exists(self, final_name: str) -> bool:
        handle: object | None = None
        try:
            handle = self._open_directory(self._descriptor, _component(final_name))
            return True
        except FileNotFoundError:
            return False
        except OSError as error:
            if error.errno in {2, 3}:
                return False
            raise _publication_error(_OPERATION_FAILED, error)
        except BaseException as error:
            raise _publication_error(_OPERATION_FAILED, error)
        finally:
            if handle is not None:
                try:
                    self._close_handle(handle)
                except BaseException as error:
                    raise _publication_error(_CLEANUP_FAILED, error)

    def create_staging(self, prefix: str, suffix: str) -> BoundStagingDirectory:
        try:
            if type(prefix) is not str or type(suffix) is not str or not prefix or not suffix:
                raise ValueError("invalid staging markers")
            name = _component(f"{prefix}{secrets.token_hex(16)}{suffix}")
            root = self._create_directory(self._descriptor, name)
            try:
                staging = BoundStagingDirectory(self, name, root)
            except BaseException:
                self._close_handle(root)
                raise
            try:
                self._staging.append(staging)
            except BaseException:
                staging._close()
                raise
            return staging
        except PrivateArtifactPublicationError:
            raise
        except BaseException as error:
            raise _publication_error(_OPERATION_FAILED, error)

    def publish_no_replace(
        self, staging: BoundStagingDirectory, final_name: str
    ) -> None:
        try:
            if staging._owner is not self or staging._closed or staging._published:
                raise OSError(errno.EBADF, "invalid staging capability")
            final_name = _component(final_name)
            staging._close_children()
            self._rename_no_replace(staging, final_name)
            staging._published = True
            staging._close()
        except PrivateArtifactPublicationError:
            raise
        except OSError as error:
            code = _RENAME_UNAVAILABLE if error.errno == errno.ENOTSUP else _OPERATION_FAILED
            raise _publication_error(code, error)
        except BaseException as error:
            raise _publication_error(_OPERATION_FAILED, error)

    def read_exact_tree(
        self, final_name: str, layout: ArtifactLayout
    ) -> dict[str, bytes]:
        final: object | None = None
        children: list[object] = []
        cleanup_error: BaseException | None = None
        try:
            final = self._open_directory(self._descriptor, _component(final_name))
            normalized: dict[str, frozenset[str]] = {}
            for directory_name, member_names in layout.items():
                if not isinstance(member_names, frozenset):
                    raise TypeError("artifact layout members must be frozen")
                normalized[_component(directory_name)] = frozenset(
                    _component(name) for name in member_names
                )
            if self._list(final) != set(normalized):
                raise OSError(errno.EINVAL, "artifact root layout mismatch")
            result: dict[str, bytes] = {}
            for directory_name, member_names in normalized.items():
                child = self._open_directory(final, directory_name)
                children.append(child)
                if self._list(child) != set(member_names):
                    raise OSError(errno.EINVAL, "artifact member layout mismatch")
                for member_name in member_names:
                    result[f"{directory_name}/{member_name}"] = self._read_member(
                        child, member_name
                    )
            return result
        except PrivateArtifactPublicationError:
            raise
        except BaseException as error:
            raise _publication_error(_TREE_INVALID, error)
        finally:
            active_error = sys.exc_info()[0] is not None
            for handle in reversed(children):
                try:
                    self._close_handle(handle)
                except BaseException as error:
                    cleanup_error = cleanup_error or error
            if final is not None:
                try:
                    self._close_handle(final)
                except BaseException as error:
                    cleanup_error = cleanup_error or error
            if cleanup_error is not None and not active_error:
                raise _publication_error(_CLEANUP_FAILED, cleanup_error)

    def sync_parent(self) -> None:
        try:
            self._sync(self._descriptor)
        except BaseException as error:
            raise _publication_error(_OPERATION_FAILED, error)

    def _close(self) -> None:
        if self._closed:
            return
        error: BaseException | None = None
        for staging in reversed(self._staging):
            try:
                staging._close()
            except BaseException as close_error:
                error = error or close_error
        try:
            self._close_handle(self._descriptor)
        except BaseException as close_error:
            error = error or close_error
        self._closed = True
        if error is not None:
            raise error


@dataclass
class _PosixLock:
    descriptor: int
    flock_api: object


class _PosixPublicationDirectory(AuthenticatedPublicationDirectory):
    def __init__(self, descriptor: int, *, os_api: object) -> None:
        super().__init__(descriptor)
        self._os = os_api
        _required_posix_flag(os_api, "O_NOFOLLOW")

    @classmethod
    def take(cls, handle: object) -> _PosixPublicationDirectory:
        if not isinstance(handle, _PosixAcquiredHandle):
            raise TypeError("invalid POSIX acquired handle")
        if not stat.S_ISDIR(handle.os_api.fstat(handle.value).st_mode):
            raise OSError(errno.ENOTDIR, "publication parent is not a directory")
        return cls(handle.value, os_api=handle.os_api)

    def _list(self, directory: object) -> set[str]:
        return set(self._os.listdir(directory))

    def _open_directory(self, parent: object, name: str) -> int:
        return _posix_open_directory_component(self._os, parent, name, create=False)

    def _create_directory(self, parent: object, name: str) -> int:
        self._os.mkdir(_component(name), 0o700, dir_fd=parent)
        self._os.fsync(parent)
        return _posix_open_directory_component(self._os, parent, name, create=False)

    def _write_exclusive(self, parent: object, name: str, content: bytes) -> None:
        descriptor = self._os.open(
            _component(name),
            self._os.O_WRONLY
            | self._os.O_CREAT
            | self._os.O_EXCL
            | _required_posix_flag(self._os, "O_NOFOLLOW")
            | getattr(self._os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent,
        )
        try:
            offset = 0
            while offset < len(content):
                written = self._os.write(descriptor, content[offset:])
                if type(written) is not int or written < 1:
                    raise OSError(errno.EIO, "member write failed")
                offset += written
            self._os.fsync(descriptor)
        finally:
            self._os.close(descriptor)

    def _read_member(self, parent: object, name: str) -> bytes:
        name = _component(name)
        descriptor = self._os.open(
            name,
            self._os.O_RDONLY
            | _required_posix_flag(self._os, "O_NOFOLLOW")
            | getattr(self._os, "O_CLOEXEC", 0),
            dir_fd=parent,
        )
        try:
            opened = self._os.fstat(descriptor)
            candidate = self._os.stat(name, dir_fd=parent, follow_symlinks=False)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or not os.path.samestat(opened, candidate)
            ):
                raise OSError(errno.EPERM, "unsafe artifact member")
            chunks: list[bytes] = []
            while True:
                chunk = self._os.read(descriptor, 65536)
                if not chunk:
                    break
                chunks.append(chunk)
            final_opened = self._os.fstat(descriptor)
            final_candidate = self._os.stat(name, dir_fd=parent, follow_symlinks=False)
            if (
                not os.path.samestat(opened, final_opened)
                or not os.path.samestat(opened, final_candidate)
                or opened.st_size != final_opened.st_size
                or opened.st_mtime_ns != final_opened.st_mtime_ns
                or opened.st_ctime_ns != final_opened.st_ctime_ns
            ):
                raise OSError(errno.EAGAIN, "artifact member changed")
            return b"".join(chunks)
        finally:
            self._os.close(descriptor)

    def _sync(self, descriptor: object) -> None:
        self._os.fsync(descriptor)

    def _close_handle(self, descriptor: object) -> None:
        self._os.close(descriptor)

    def _acquire_lock(self, lock_name: str) -> _PosixLock:
        import fcntl

        name = _component(lock_name)
        flags = (
            self._os.O_RDWR
            | _required_posix_flag(self._os, "O_NOFOLLOW")
            | _required_posix_flag(self._os, "O_NONBLOCK")
            | getattr(self._os, "O_CLOEXEC", 0)
        )
        created = False
        try:
            descriptor = self._os.open(name, flags, dir_fd=self._descriptor)
        except FileNotFoundError:
            descriptor = self._os.open(
                name,
                flags | self._os.O_CREAT | self._os.O_EXCL,
                0o600,
                dir_fd=self._descriptor,
            )
            created = True
        try:
            observed = self._os.fstat(descriptor)
            candidate = self._os.stat(
                name, dir_fd=self._descriptor, follow_symlinks=False
            )
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_nlink != 1
                or not os.path.samestat(observed, candidate)
            ):
                raise OSError(errno.EPERM, "unsafe publisher lock")
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if created:
                self._os.fsync(self._descriptor)
            return _PosixLock(descriptor, fcntl)
        except BaseException:
            self._os.close(descriptor)
            raise

    def _release_lock(self, lock: object) -> None:
        if not isinstance(lock, _PosixLock):
            raise TypeError("invalid POSIX lock")
        error: BaseException | None = None
        try:
            lock.flock_api.flock(lock.descriptor, lock.flock_api.LOCK_UN)
        except BaseException as unlock_error:
            error = unlock_error
        try:
            self._os.close(lock.descriptor)
        except BaseException as close_error:
            error = error or close_error
        if error is not None:
            raise error

    def _rename_no_replace(
        self, staging: BoundStagingDirectory, final_name: str
    ) -> None:
        _rename_no_replace_at(self._descriptor, staging.name, final_name)


@dataclass
class _WindowsLock:
    descriptor: object
    api: object


class _WindowsPublicationDirectory(AuthenticatedPublicationDirectory):
    def __init__(self, descriptor: object, *, api: object) -> None:
        super().__init__(descriptor)
        self._api = api

    @classmethod
    def take(cls, handle: object) -> _WindowsPublicationDirectory:
        if not isinstance(handle, _WindowsAcquiredHandle):
            raise TypeError("invalid Windows acquired handle")
        handle.api.validate_handle(handle.value, directory=True)
        return cls(handle.value, api=handle.api)

    def _list(self, directory: object) -> set[str]:
        return self._api.directory_entries(directory)

    def _open_directory(self, parent: object, name: str) -> object:
        return self._api.open_directory_relative(parent, _component(name))

    def _create_directory(self, parent: object, name: str) -> object:
        handle = self._api.create_directory_relative(parent, _component(name))
        try:
            self._api.flush_handle(parent)
        except BaseException:
            self._api.close_handle(handle)
            raise
        return handle

    def _write_exclusive(self, parent: object, name: str, content: bytes) -> None:
        handle = self._api.create_file_relative(parent, _component(name))
        try:
            self._api.write_all(handle, content)
            self._api.flush_handle(handle)
        finally:
            self._api.close_handle(handle)

    def _read_member(self, parent: object, name: str) -> bytes:
        name = _component(name)
        handle = self._api.open_file_relative(parent, name)
        try:
            before = self._api.validate_handle(handle, directory=False)
            content = self._api.read_member(handle)
            after = self._api.validate_handle(handle, directory=False)
            rebound = self._api.open_file_relative(parent, name)
            try:
                rebound_snapshot = self._api.validate_handle(rebound, directory=False)
            finally:
                self._api.close_handle(rebound)
            if before != after or before != rebound_snapshot:
                raise OSError(errno.EAGAIN, "artifact member changed")
            return content
        finally:
            self._api.close_handle(handle)

    def _sync(self, descriptor: object) -> None:
        self._api.flush_handle(descriptor)

    def _close_handle(self, descriptor: object) -> None:
        self._api.close_handle(descriptor)

    def _acquire_lock(self, lock_name: str) -> _WindowsLock:
        name = _component(lock_name)
        mutex_api = _NativeWindowsMutex()
        security = mutex_api.build_security_attributes(
            _windows_mutex_policy(mutex_api.owner_sid())
        )
        try:
            identity = self._api.validate_handle(self._descriptor, directory=True)
            digest = hashlib.sha256(repr((identity[:2], name)).encode("utf-8")).hexdigest()
            descriptor = mutex_api.create_mutex(
                f"Global\\PrivateArtifactPublication-{digest}",
                security.attributes,
            )
        finally:
            mutex_api.free_security_descriptor(security)
        try:
            outcome = mutex_api.wait(descriptor, 0)
            if outcome not in {0x00000000, 0x00000080}:
                number = errno.EBUSY if outcome == 0x00000102 else errno.EIO
                raise OSError(number, "publisher mutex unavailable")
            return _WindowsLock(descriptor, mutex_api)
        except BaseException:
            mutex_api.close_handle(descriptor)
            raise

    def _release_lock(self, lock: object) -> None:
        if not isinstance(lock, _WindowsLock):
            raise TypeError("invalid Windows lock")
        error: BaseException | None = None
        try:
            lock.api.release_mutex(lock.descriptor)
        except BaseException as unlock_error:
            error = unlock_error
        try:
            lock.api.close_handle(lock.descriptor)
        except BaseException as close_error:
            error = error or close_error
        if error is not None:
            raise error

    def _rename_no_replace(
        self, staging: BoundStagingDirectory, final_name: str
    ) -> None:
        self._api.rename_no_replace(staging._root, self._descriptor, final_name)


def _close_acquired(handle: object) -> None:
    if isinstance(handle, _WindowsAcquiredHandle):
        handle.api.close_handle(handle.value)
    elif isinstance(handle, _PosixAcquiredHandle):
        handle.os_api.close(handle.value)
    else:
        api = getattr(handle, "api", None)
        value = getattr(handle, "value", None)
        if api is None or value is None:
            raise TypeError("invalid acquired handle")
        api.close_handle(value)


@contextmanager
def _authenticated_publication_directory(
    *,
    root: Path,
    relative: tuple[str, ...],
    lock_name: str,
) -> Iterator[AuthenticatedPublicationDirectory]:
    publication: AuthenticatedPublicationDirectory | None = None
    lock: object | None = None
    try:
        acquired = _open_authenticated_parent(root, relative)
        handle: object | None = acquired
        try:
            if _platform_name() == "nt":
                publication = _WindowsPublicationDirectory.take(handle)
            else:
                publication = _PosixPublicationDirectory.take(handle)
            handle = None
        finally:
            if handle is not None:
                _close_acquired(handle)
        try:
            lock = publication._acquire_lock(lock_name)
        except BaseException as error:
            raise _publication_error(_LOCK_FAILED, error)
    except PrivateArtifactPublicationError:
        if publication is not None:
            try:
                publication._close()
            except BaseException:
                pass
        raise
    except BaseException as error:
        if publication is not None:
            try:
                publication._close()
            except BaseException:
                pass
        raise _publication_error(_ACQUISITION_FAILED, error)

    try:
        assert publication is not None
        yield publication
    finally:
        cleanup_error: BaseException | None = None
        if lock is not None:
            try:
                publication._release_lock(lock)
            except BaseException as error:
                cleanup_error = error
        try:
            publication._close()
        except BaseException as error:
            cleanup_error = cleanup_error or error
        if cleanup_error is not None:
            raise _publication_error(_CLEANUP_FAILED, cleanup_error)


def authenticated_publication_directory(
    *,
    approved_private_root: Path,
    publication_parent: Path,
    lock_name: str,
) -> ContextManager[AuthenticatedPublicationDirectory]:
    """Acquire a locked, authenticated capability for a publication parent."""

    try:
        root, relative = _validate_paths(approved_private_root, publication_parent)
        _component(lock_name)
    except BaseException as error:
        return _failing_context(_publication_error(_PATH_REJECTED, error))
    return _authenticated_publication_directory(
        root=root,
        relative=relative,
        lock_name=lock_name,
    )


@contextmanager
def _failing_context(
    error: PrivateArtifactPublicationError,
) -> Iterator[AuthenticatedPublicationDirectory]:
    raise error
    yield  # pragma: no cover
