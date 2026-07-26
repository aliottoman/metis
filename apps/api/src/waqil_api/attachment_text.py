from __future__ import annotations

import io
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree


TEXT_SUFFIXES = {
    ".txt", ".md", ".rst", ".rtf", ".log", ".csv", ".tsv", ".ipynb",
    ".py", ".js", ".jsx", ".ts", ".tsx", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".conf", ".xml", ".html", ".css", ".sql",
    ".sh", ".zsh", ".go", ".rs", ".java", ".kt", ".rb", ".php", ".c",
    ".h", ".cpp", ".hpp", ".cs", ".tf", ".tfvars", ".graphql", ".gql",
}
BINARY_DOCUMENT_SUFFIXES = {".pdf", ".docx", ".pptx", ".xlsx"}
IMAGE_SUFFIX_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
TEXT_MEDIA_TYPES = {
    "application/json",
    "application/ld+json",
    "application/xml",
    "application/javascript",
    "application/x-yaml",
    "application/yaml",
    "application/toml",
    "application/rtf",
}
TEXT_BASENAMES = {"readme", "license", "dockerfile", "makefile", "procfile"}
_MAX_ZIP_MEMBERS = 5_000
_MAX_UNCOMPRESSED_BYTES = 32 * 1024 * 1024
_MAX_IMAGE_DIMENSION = 30_000
_MAX_IMAGE_PIXELS = 100_000_000


class AttachmentExtractionError(ValueError):
    pass


class AttachmentTextTooLargeError(AttachmentExtractionError):
    pass


def image_attachment_media_type(filename: str, media_type: str) -> str | None:
    """Return the canonical type when an image extension and media type agree."""
    expected = IMAGE_SUFFIX_MEDIA_TYPES.get(Path(filename).suffix.lower())
    normalized = media_type.split(";", 1)[0].lower()
    if expected and normalized in {expected, "application/octet-stream"}:
        return expected
    return None


def supports_attachment(filename: str, media_type: str) -> bool:
    path = Path(filename)
    suffix = path.suffix.lower()
    basename = path.stem.lower()
    normalized_media = media_type.split(";", 1)[0].lower()
    if suffix in IMAGE_SUFFIX_MEDIA_TYPES or normalized_media.startswith("image/"):
        return image_attachment_media_type(filename, media_type) is not None
    return bool(
        suffix in TEXT_SUFFIXES
        or suffix in BINARY_DOCUMENT_SUFFIXES
        or basename in TEXT_BASENAMES
        or normalized_media.startswith("text/")
        or normalized_media in TEXT_MEDIA_TYPES
    )


def extract_attachment_text(
    filename: str,
    media_type: str,
    content: bytes,
    *,
    max_bytes: int,
) -> str:
    """Convert one supported attachment into bounded, untrusted plain text."""
    suffix = Path(filename).suffix.lower()
    normalized_media = media_type.split(";", 1)[0].lower()
    try:
        if suffix in IMAGE_SUFFIX_MEDIA_TYPES or normalized_media.startswith("image/"):
            image_media_type = image_attachment_media_type(filename, media_type)
            if image_media_type is None:
                raise AttachmentExtractionError(
                    "the image filename extension and media type do not match"
                )
            width, height = _image_dimensions(image_media_type, content)
            safe_name = Path(filename).name
            text = (
                f'Image attachment "{safe_name}" '
                f"({image_media_type}, {width}\u00d7{height}). "
                "The image bytes were validated and stored locally; pixel content is not "
                "extracted by the current text-only chat pipeline."
            )
        elif suffix == ".pdf":
            text = _extract_pdf(content)
        elif suffix == ".docx":
            text = _extract_docx(content)
        elif suffix == ".pptx":
            text = _extract_pptx(content)
        elif suffix == ".xlsx":
            text = _extract_xlsx(content)
        elif supports_attachment(filename, media_type):
            text = _decode_text(content)
        else:
            raise AttachmentExtractionError(
                "supported attachments are text, source code, PDF, DOCX, PPTX, XLSX, "
                "PNG, JPEG, WebP, or GIF"
            )
    except AttachmentExtractionError:
        raise
    except (zipfile.BadZipFile, RuntimeError, OSError, EOFError, NotImplementedError) as exc:
        raise AttachmentExtractionError(
            "the Office document package is corrupt or unsupported"
        ) from exc

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise AttachmentExtractionError("the attachment contains no extractable text")
    encoded_size = len(normalized.encode("utf-8"))
    if encoded_size > max_bytes:
        raise AttachmentTextTooLargeError(
            f"extracted attachment text exceeds the {max_bytes}-byte context budget"
        )
    return normalized


def _bounded_image_dimensions(width: int, height: int) -> tuple[int, int]:
    if width < 1 or height < 1:
        raise AttachmentExtractionError("the image dimensions are invalid")
    if width > _MAX_IMAGE_DIMENSION or height > _MAX_IMAGE_DIMENSION:
        raise AttachmentExtractionError("the image dimensions exceed the safe limit")
    if width * height > _MAX_IMAGE_PIXELS:
        raise AttachmentExtractionError("the image pixel count exceeds the safe limit")
    return width, height


def _image_dimensions(media_type: str, content: bytes) -> tuple[int, int]:
    if media_type == "image/png":
        if (
            len(content) < 24
            or content[:8] != b"\x89PNG\r\n\x1a\n"
            or content[12:16] != b"IHDR"
        ):
            raise AttachmentExtractionError("the PNG image is corrupt or unsupported")
        return _bounded_image_dimensions(
            int.from_bytes(content[16:20], "big"),
            int.from_bytes(content[20:24], "big"),
        )

    if media_type == "image/gif":
        if len(content) < 10 or content[:6] not in {b"GIF87a", b"GIF89a"}:
            raise AttachmentExtractionError("the GIF image is corrupt or unsupported")
        return _bounded_image_dimensions(
            int.from_bytes(content[6:8], "little"),
            int.from_bytes(content[8:10], "little"),
        )

    if media_type == "image/webp":
        if len(content) < 30 or content[:4] != b"RIFF" or content[8:12] != b"WEBP":
            raise AttachmentExtractionError("the WebP image is corrupt or unsupported")
        chunk = content[12:16]
        if chunk == b"VP8X":
            width = int.from_bytes(content[24:27], "little") + 1
            height = int.from_bytes(content[27:30], "little") + 1
        elif chunk == b"VP8L" and content[20] == 0x2F:
            dimensions = int.from_bytes(content[21:25], "little")
            width = (dimensions & 0x3FFF) + 1
            height = ((dimensions >> 14) & 0x3FFF) + 1
        elif chunk == b"VP8 " and content[23:26] == b"\x9d\x01\x2a":
            width = int.from_bytes(content[26:28], "little") & 0x3FFF
            height = int.from_bytes(content[28:30], "little") & 0x3FFF
        else:
            raise AttachmentExtractionError("the WebP image is corrupt or unsupported")
        return _bounded_image_dimensions(width, height)

    if media_type == "image/jpeg":
        if len(content) < 4 or not content.startswith(b"\xff\xd8"):
            raise AttachmentExtractionError("the JPEG image is corrupt or unsupported")
        sof_markers = {
            0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
            0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
        }
        offset = 2
        while offset + 4 <= len(content):
            if content[offset] != 0xFF:
                raise AttachmentExtractionError("the JPEG image is corrupt or unsupported")
            while offset < len(content) and content[offset] == 0xFF:
                offset += 1
            if offset >= len(content):
                break
            marker = content[offset]
            offset += 1
            if marker in {0x01, *range(0xD0, 0xDA)}:
                continue
            if offset + 2 > len(content):
                break
            segment_length = int.from_bytes(content[offset:offset + 2], "big")
            if segment_length < 2 or offset + segment_length > len(content):
                raise AttachmentExtractionError("the JPEG image is corrupt or unsupported")
            if marker in sof_markers:
                if segment_length < 7:
                    raise AttachmentExtractionError("the JPEG image is corrupt or unsupported")
                height = int.from_bytes(content[offset + 3:offset + 5], "big")
                width = int.from_bytes(content[offset + 5:offset + 7], "big")
                return _bounded_image_dimensions(width, height)
            offset += segment_length
        raise AttachmentExtractionError("the JPEG image is corrupt or unsupported")

    raise AttachmentExtractionError("the image format is unsupported")


def _decode_text(content: bytes) -> str:
    if b"\x00" in content:
        raise AttachmentExtractionError("text attachments may not contain NUL bytes")
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise AttachmentExtractionError("text attachments must be UTF-8") from exc


def _safe_zip(content: bytes) -> zipfile.ZipFile:
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except (zipfile.BadZipFile, OSError) as exc:
        raise AttachmentExtractionError("the Office document is not a valid ZIP package") from exc
    infos = archive.infolist()
    if len(infos) > _MAX_ZIP_MEMBERS:
        archive.close()
        raise AttachmentExtractionError("the Office document contains too many files")
    if sum(item.file_size for item in infos) > _MAX_UNCOMPRESSED_BYTES:
        archive.close()
        raise AttachmentExtractionError("the Office document expands beyond the safe limit")
    if any(
        name.startswith(("/", "\\")) or ".." in Path(name).parts
        for name in (item.filename for item in infos)
    ):
        archive.close()
        raise AttachmentExtractionError("the Office document contains an unsafe path")
    return archive


def _xml_text(payload: bytes, *, paragraph_tags: tuple[str, ...]) -> str:
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as exc:
        raise AttachmentExtractionError("the Office document contains invalid XML") from exc
    pieces: list[str] = []
    for element in root.iter():
        local = element.tag.rsplit("}", 1)[-1]
        if local in {"t", "v"} and element.text:
            pieces.append(element.text)
        elif local == "tab":
            pieces.append("\t")
        elif local in paragraph_tags:
            pieces.append("\n")
    return "".join(pieces)


def _extract_docx(content: bytes) -> str:
    with _safe_zip(content) as archive:
        names = [
            name
            for name in archive.namelist()
            if name == "word/document.xml"
            or re.fullmatch(r"word/(?:header|footer)\d+\.xml", name)
        ]
        if "word/document.xml" not in names:
            raise AttachmentExtractionError("the DOCX document is missing word/document.xml")
        return "\n".join(
            _xml_text(archive.read(name), paragraph_tags=("p", "tr")) for name in names
        )


def _slide_number(name: str) -> int:
    match = re.search(r"slide(\d+)\.xml$", name)
    return int(match.group(1)) if match else 0


def _extract_pptx(content: bytes) -> str:
    with _safe_zip(content) as archive:
        slides = sorted(
            (
                name
                for name in archive.namelist()
                if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
            ),
            key=_slide_number,
        )
        if not slides:
            raise AttachmentExtractionError("the PPTX document contains no slides")
        return "\n\n".join(
            f"Slide {index}\n{_xml_text(archive.read(name), paragraph_tags=('p',))}"
            for index, name in enumerate(slides, 1)
        )


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    try:
        root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    except ElementTree.ParseError as exc:
        raise AttachmentExtractionError("the XLSX shared strings are invalid") from exc
    values: list[str] = []
    for item in root.iter():
        if item.tag.rsplit("}", 1)[-1] != "si":
            continue
        values.append("".join(
            node.text or ""
            for node in item.iter()
            if node.tag.rsplit("}", 1)[-1] == "t"
        ))
    return values


def _extract_xlsx(content: bytes) -> str:
    with _safe_zip(content) as archive:
        sheets = sorted(
            name
            for name in archive.namelist()
            if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)
        )
        if not sheets:
            raise AttachmentExtractionError("the XLSX workbook contains no worksheets")
        shared = _shared_strings(archive)
        rendered: list[str] = []
        for sheet_index, name in enumerate(sheets, 1):
            try:
                root = ElementTree.fromstring(archive.read(name))
            except ElementTree.ParseError as exc:
                raise AttachmentExtractionError("the XLSX worksheet XML is invalid") from exc
            rows: list[str] = []
            for row in root.iter():
                if row.tag.rsplit("}", 1)[-1] != "row":
                    continue
                values: list[str] = []
                for cell in row:
                    if cell.tag.rsplit("}", 1)[-1] != "c":
                        continue
                    cell_type = cell.attrib.get("t", "")
                    raw = next(
                        (
                            node.text or ""
                            for node in cell.iter()
                            if node.tag.rsplit("}", 1)[-1] == "v"
                        ),
                        "",
                    )
                    if cell_type == "s" and raw.isdigit() and int(raw) < len(shared):
                        value = shared[int(raw)]
                    elif cell_type == "inlineStr":
                        value = "".join(
                            node.text or ""
                            for node in cell.iter()
                            if node.tag.rsplit("}", 1)[-1] == "t"
                        )
                    else:
                        value = raw
                    values.append(value)
                if any(value for value in values):
                    rows.append("\t".join(values))
            rendered.append(f"Worksheet {sheet_index}\n" + "\n".join(rows))
        return "\n\n".join(rendered)


def _extract_pdf(content: bytes) -> str:
    if not content.startswith(b"%PDF-"):
        raise AttachmentExtractionError("the PDF signature is invalid")
    executable = shutil.which("pdftotext")
    if executable is None:
        raise AttachmentExtractionError("PDF extraction requires the local pdftotext utility")
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as source:
            source.write(content)
            source.flush()
            result = subprocess.run(
                [executable, "-q", "-layout", "-enc", "UTF-8", source.name, "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=20,
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AttachmentExtractionError("the PDF could not be converted to text") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()[:180]
        raise AttachmentExtractionError(detail or "the PDF could not be converted to text")
    extracted = result.stdout.decode("utf-8", errors="replace")
    pages = [page.strip() for page in extracted.split("\f")]
    pages = [page for page in pages if page]
    return "\n\n".join(
        f"[PDF page {page_number}]\n{page}"
        for page_number, page in enumerate(pages, start=1)
    )
