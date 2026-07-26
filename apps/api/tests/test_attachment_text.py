from __future__ import annotations

import io
import shutil
import struct
import zipfile
import zlib

import pytest

from waqil_api.attachment_text import (
    AttachmentExtractionError,
    AttachmentTextTooLargeError,
    extract_attachment_text,
    supports_attachment,
)


def packaged(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def png_image(width: int = 2, height: int = 3) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    scanlines = b"".join(b"\x00" + (b"\x00" * width * 4) for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


def text_pdf(text: str) -> bytes:
    """Build a tiny dependency-free PDF fixture with one text page."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 18 Tf 72 720 Td ({escaped}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    result = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(result))
        result.extend(f"{number} 0 obj\n".encode())
        result.extend(body)
        result.extend(b"\nendobj\n")
    xref = len(result)
    result.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    result.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        result.extend(f"{offset:010d} 00000 n \n".encode())
    result.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode()
    )
    return bytes(result)


def test_supported_text_and_source_files_are_bounded() -> None:
    assert supports_attachment("notes.md", "text/markdown")
    assert supports_attachment("analysis.ipynb", "application/json")
    assert extract_attachment_text(
        "notes.md", "text/markdown", b"hello\r\nworld", max_bytes=64
    ) == "hello\nworld"
    with pytest.raises(AttachmentTextTooLargeError):
        extract_attachment_text("notes.md", "text/plain", b"too long", max_bytes=3)


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="pdftotext is unavailable")
def test_pdf_text_is_extracted_with_page_provenance() -> None:
    extracted = extract_attachment_text(
        "brief.pdf",
        "application/pdf",
        text_pdf("The launch code is ORCHID-73."),
        max_bytes=2_000,
    )
    assert extracted.startswith("[PDF page 1]")
    assert "ORCHID-73" in extracted


def test_docx_pptx_and_xlsx_are_normalized_without_optional_packages() -> None:
    docx = packaged({
        "word/document.xml": (
            '<w:document xmlns:w="urn:w"><w:body><w:p><w:r><w:t>Hello DOCX</w:t>'
            "</w:r></w:p></w:body></w:document>"
        )
    })
    assert "Hello DOCX" in extract_attachment_text(
        "brief.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        docx,
        max_bytes=1_000,
    )

    pptx = packaged({
        "ppt/slides/slide1.xml": (
            '<p:sld xmlns:p="urn:p" xmlns:a="urn:a"><p:cSld><a:p><a:r>'
            "<a:t>Slide copy</a:t></a:r></a:p></p:cSld></p:sld>"
        )
    })
    presentation = extract_attachment_text(
        "deck.pptx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        pptx,
        max_bytes=1_000,
    )
    assert "Slide 1" in presentation and "Slide copy" in presentation

    xlsx = packaged({
        "xl/sharedStrings.xml": (
            '<sst xmlns="urn:x"><si><t>Revenue</t></si></sst>'
        ),
        "xl/worksheets/sheet1.xml": (
            '<worksheet xmlns="urn:x"><sheetData><row><c t="s"><v>0</v></c>'
            "<c><v>42</v></c></row></sheetData></worksheet>"
        ),
    })
    workbook = extract_attachment_text(
        "data.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        xlsx,
        max_bytes=1_000,
    )
    assert "Revenue\t42" in workbook


def test_raster_images_are_validated_and_represented_as_bounded_metadata() -> None:
    assert supports_attachment("photo.png", "image/png")
    assert supports_attachment("photo.PNG", "application/octet-stream")
    metadata = extract_attachment_text(
        "photo.png", "image/png", png_image(), max_bytes=500
    )
    assert "image/png, 2×3" in metadata
    assert "pixel content is not extracted" in metadata
    assert not supports_attachment("photo.jpg", "image/png")
    assert not supports_attachment("graphic.svg", "image/svg+xml")
    with pytest.raises(AttachmentExtractionError):
        extract_attachment_text("photo.png", "image/png", b"png", max_bytes=100)


def test_invalid_or_unsafe_packages_are_rejected() -> None:
    unsafe = packaged({"../word/document.xml": "<document />"})
    with pytest.raises(AttachmentExtractionError, match="unsafe path"):
        extract_attachment_text("brief.docx", "application/octet-stream", unsafe, max_bytes=100)

    corrupt_buffer = io.BytesIO()
    with zipfile.ZipFile(corrupt_buffer, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("word/document.xml", "<document>known-payload</document>")
    corrupt = corrupt_buffer.getvalue().replace(b"known-payload", b"broken-payload")
    with pytest.raises(AttachmentExtractionError, match="corrupt or unsupported"):
        extract_attachment_text("brief.docx", "application/octet-stream", corrupt, max_bytes=100)
