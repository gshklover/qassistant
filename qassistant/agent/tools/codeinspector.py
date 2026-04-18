"""
Code inspector tool for extracting and searching public APIs from Python modules.

Uses griffe for runtime module inspection and embedding-based semantic search
for finding relevant API entries.
"""
import dataclasses
import griffe
import numpy
from typing import Awaitable, Callable

from .knowledgebase import cosine_distance


@dataclasses.dataclass(slots=True)
class ApiEntry:
    """
    A single public API entry extracted from a module.
    """
    full_name: str
    kind: str
    signature: str
    docstring: str
    alias: str = ""
    embedding: numpy.ndarray | None = None

    @property
    def embedding_text(self) -> str:
        """
        Return combined text used for embedding: signature followed by docstring.
        """
        parts = [self.signature]
        if self.docstring:
            parts.append(self.docstring)
        return "\n".join(parts)


def _build_signature(member: griffe.Object) -> str:
    """
    Build a human-readable signature string from a griffe object.
    """
    if member.kind == griffe.Kind.CLASS:
        init = member.members.get("__init__")
        if init is not None and hasattr(init, "parameters"):
            params = _format_parameters(init.parameters, skip_self=True)
            return f"class {member.path}({params})"
        return f"class {member.path}"

    if member.kind == griffe.Kind.FUNCTION:
        params = _format_parameters(member.parameters, skip_self=True)
        return f"{member.path}({params})"

    # attribute / property – no callable signature
    return member.path


def _format_parameters(parameters: griffe.Parameters, skip_self: bool = True) -> str:
    """
    Format a griffe Parameters sequence into a comma-separated string.
    """
    parts: list[str] = []
    for param in parameters:
        if skip_self and param.name in ("self", "cls"):
            continue
        token = param.name
        annotation = str(param.annotation) if param.annotation is not None else ""
        if annotation and annotation != "empty":
            token += f": {annotation}"
        default = str(param.default) if param.default is not None else ""
        if default and default != "empty":
            token += f" = {default}"
        parts.append(token)
    return ", ".join(parts)


def _get_docstring(member: griffe.Object) -> str:
    """
    Safely extract a docstring from a griffe object.
    """
    if member.docstring and member.docstring.value:
        return member.docstring.value.strip()
    return ""


def _is_public(name: str) -> bool:
    """
    Return True if the name represents a public symbol.
    """
    return not name.startswith("_")


def _collect_aliases(module: griffe.Module) -> dict[str, str]:
    """
    Build a mapping from target_path to the shortest alias path exposed by the module.

    Scans ALIAS members to discover re-exports. For example, in the ``json``
    module ``json.JSONEncoder`` is an alias for ``json.encoder.JSONEncoder``,
    so the returned dict would contain
    ``{"json.encoder.JSONEncoder": "json.JSONEncoder"}``.
    """
    aliases: dict[str, str] = {}
    for name, member in module.members.items():
        if not _is_public(name):
            continue
        if member.kind != griffe.Kind.ALIAS:
            continue
        alias_path = member.path          # e.g. json.JSONEncoder
        target_path = member.target_path  # e.g. json.encoder.JSONEncoder
        if target_path and (target_path not in aliases or len(alias_path) < len(aliases[target_path])):
            aliases[target_path] = alias_path
    return aliases


def _resolve_alias(member: griffe.Alias) -> griffe.Object | None:
    """
    Resolve a griffe Alias to its target Object by inspecting the target module.

    Returns None if the target cannot be resolved.
    """
    target_path = member.target_path
    parts = target_path.rsplit(".", 1)
    if len(parts) != 2:
        return None
    mod_path, obj_name = parts
    try:
        mod = griffe.inspect(mod_path)
        target = mod.members.get(obj_name)
        if target is not None and target.kind != griffe.Kind.ALIAS:
            return target
    except Exception:
        pass
    return None


def _extract_member(
    member: griffe.Object,
    alias: str,
) -> list[ApiEntry]:
    """
    Extract ApiEntry objects from a single griffe member (and its children for classes).
    """
    entries: list[ApiEntry] = []

    if member.kind == griffe.Kind.FUNCTION:
        entries.append(ApiEntry(
            full_name=member.path,
            kind="function",
            signature=_build_signature(member),
            docstring=_get_docstring(member),
            alias=alias,
        ))

    elif member.kind == griffe.Kind.CLASS:
        entries.append(ApiEntry(
            full_name=member.path,
            kind="class",
            signature=_build_signature(member),
            docstring=_get_docstring(member),
            alias=alias,
        ))
        # extract public methods, properties, and class attributes:
        for mname, child in member.members.items():
            if not _is_public(mname):
                continue
            if child.kind == griffe.Kind.ALIAS:
                continue
            if child.kind == griffe.Kind.FUNCTION:
                entries.append(ApiEntry(
                    full_name=child.path,
                    kind="method",
                    signature=_build_signature(child),
                    docstring=_get_docstring(child),
                    alias=f"{alias}.{mname}",
                ))
            elif child.kind == griffe.Kind.ATTRIBUTE:
                entries.append(ApiEntry(
                    full_name=child.path,
                    kind="property",
                    signature=child.path,
                    docstring=_get_docstring(child),
                    alias=f"{alias}.{mname}",
                ))

    elif member.kind == griffe.Kind.ATTRIBUTE:
        entries.append(ApiEntry(
            full_name=member.path,
            kind="attribute",
            signature=member.path,
            docstring=_get_docstring(member),
            alias=alias,
        ))

    return entries


def _extract_entries(module: griffe.Module) -> list[ApiEntry]:
    """
    Walk a griffe module tree and extract public classes, functions,
    methods, and properties as ApiEntry objects.

    Re-exported aliases (e.g. ``json.JSONEncoder`` pointing to
    ``json.encoder.JSONEncoder``) are resolved so that each entry carries
    the shortest public alias exposed by the inspected module.
    """
    entries: list[ApiEntry] = []
    alias_map = _collect_aliases(module)

    for name, member in module.members.items():
        if not _is_public(name):
            continue

        if member.kind == griffe.Kind.ALIAS:
            resolved = _resolve_alias(member)
            if resolved is not None:
                entries.extend(_extract_member(resolved, alias=member.path))
            continue

        alias = alias_map.get(member.path, member.path)
        entries.extend(_extract_member(member, alias=alias))

    return entries


class CodeInspector:
    """
    Tool for inspecting Python modules and performing semantic search over public APIs.
    """

    def __init__(self, embedding: Callable[[str], Awaitable[list[float]]]):
        self._embedder = embedding
        self._entries: list[ApiEntry] = []
        self._name_index: dict[str, ApiEntry] = {}

    @property
    def entries(self) -> list[ApiEntry]:
        """
        Return all registered API entries.
        """
        return list(self._entries)

    async def register(self, module: str):
        """
        Inspect the named module and add its public API entries to the index.
        """
        mod = griffe.inspect(module)
        new_entries = _extract_entries(mod)

        for entry in new_entries:
            embedding = await self._embed(entry.embedding_text)
            entry.embedding = embedding
            self._entries.append(entry)
            self._name_index[entry.full_name] = entry
            if entry.alias and entry.alias not in self._name_index:
                self._name_index[entry.alias] = entry

    @staticmethod
    def _entry_to_dict(entry: ApiEntry, max_doc_lines: int | None = None) -> dict[str, str]:
        """
        Convert an ApiEntry to a summary dict with name, full_name, alias,
        type, signature, and description.

        When max_doc_lines is None the full docstring is included.
        When set to an integer, only that many lines are kept and '...' is
        appended if the original was longer.
        """
        description = entry.docstring
        if max_doc_lines is not None and description:
            lines = description.split("\n", max_doc_lines)
            truncated = "\n".join(lines[:max_doc_lines]).strip()
            if len(lines) > max_doc_lines:
                truncated += "..."
            description = truncated
        return {
            "name": entry.full_name.rsplit(".", 1)[-1],
            "full_name": entry.full_name,
            "alias": entry.alias,
            "type": entry.kind,
            "signature": entry.signature,
            "description": description,
        }

    async def search(
        self,
        text: str,
        start_index: int = 0,
        end_index: int = 20,
        context: str | None = None,
    ) -> list[dict[str, str]]:
        """
        Perform semantic search for API entries matching the query text.
        Returns dicts ranked by similarity, sliced by [start_index:end_index].

        If context is provided, only entries whose full_name starts with the
        given prefix are considered (e.g. 'json', 'pathlib.Path').
        """
        if not self._entries:
            return []

        prefix = context + "." if context else None
        query_embedding = await self._embed(text)
        scored: list[tuple[float, ApiEntry]] = []
        for entry in self._entries:
            if prefix and not entry.full_name.startswith(prefix):
                continue
            if entry.embedding is not None:
                dist = cosine_distance(query_embedding, entry.embedding)
                scored.append((dist, entry))

        scored.sort(key=lambda x: x[0])
        return [self._entry_to_dict(entry) for _, entry in scored[start_index:end_index]]

    def list_members(self, name: str) -> list[dict[str, str]]:
        """
        List direct members of a class or module.

        Returns a list of dicts, each containing: name, full_name, alias, type,
        signature, and description (first line of the docstring, with '...'
        appended when the docstring spans multiple lines).
        Looks up by full_name or alias prefix.
        """
        prefix_full = name + "."
        results: list[dict[str, str]] = []
        seen: set[str] = set()
        for entry in self._entries:
            is_child = (
                entry.full_name.startswith(prefix_full)
                or entry.alias.startswith(prefix_full)
            )
            if not is_child:
                continue
            # only direct children: no further dots after the prefix
            suffix = entry.full_name[len(prefix_full):] if entry.full_name.startswith(prefix_full) else entry.alias[len(prefix_full):]
            if "." in suffix:
                continue
            if entry.full_name in seen:
                continue
            seen.add(entry.full_name)
            results.append(self._entry_to_dict(entry, max_doc_lines=1))
        return results

    def get_signatures(self, names: list[str]) -> dict[str, str | None]:
        """
        Return signatures for the specified object names.
        Names can be fully qualified (e.g. 'pathlib.Path.glob') or short aliases.
        Returns a dict mapping each requested name to its signature string, or None if not found.
        """
        result: dict[str, str | None] = {}
        for name in names:
            entry = self._name_index.get(name)
            if entry is not None:
                result[name] = entry.signature
            else:
                result[name] = None
        return result

    async def _embed(self, text: str) -> numpy.ndarray:
        """
        Run embedding on the specified text and return it as a numpy array.
        """
        raw = await self._embedder(text)
        return numpy.array(raw, dtype=numpy.float32)
