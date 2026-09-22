from dataclasses import dataclass, field
from docling.document_converter import DocumentConverter
from utils.log import get_logger
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = get_logger(__name__)

@dataclass
class Chunk:
    """One unit of text that will become one vector in Qdrant"""

    text: str
    source: str
    page: int
    chunk_id: int
    metadata: dict = field(default_factory=dict)

def parse_document(file_path: str):
    """
    Uses docling to convert a raw file into a structured document object
    with page-level text already extracted.
    """
    logger.info("Parsing document", file_path = file_path)

    converter = DocumentConverter()
    result = converter.convert(file_path)

    num_pages = len(result.document.pages)
    logger.info("Document parsed", file_path = file_path, num_pages = num_pages)

    return result.document

def get_text_splitter(
        chuck_size: int = 1000,
        chunk_overlap: int = 150
) -> RecursiveCharacterTextSplitter:
    """
    Builds a splitter that splits the text by paragraph first, 
    then sentence, words and then raw characters -- falling back only as far as
    it needs to in order to hit chunk_size.
    """

    return RecursiveCharacterTextSplitter(
        separators=["/n/n", "/n", ". ", ", ", " ", ""],
        chunk_size = chuck_size, 
        chunk_overlap = chunk_overlap,
        length_function = len
    )

def split_into_chunks(text: str, 
                      chunk_size: int = 1000, 
                      chunk_overlap: int = 150) -> list[str]:
    """Splits the text into overlapping, boundary-aware chunks"""
    splitter = get_text_splitter(chunk_size, chunk_overlap)
    return splitter.split_text(text)

def chunk_document(file_path: str, 
                   chunk_size: int = 1000, 
                   chunk_overlap: int = 150
                   ) -> list[Chunk]:
    """
    Full pipeline: parse a file with docling, then split each page's text
    into overlapping chunks with metadata attached
    """

    document = parse_document(file_path)
    all_chunks: list[Chunk] = []
    chunk_counter = 0

    for page_num, page in enumerate(document.pages, start=1):
        page_text = page.export_to_text()

        if not page_text.strip():
            logger.info("Entry page skipped", file_path=file_path, page=page_num)
            continue

        page_chunks = split_into_chunks(page_text, chunk_size, chunk_overlap)

        for chunk_text in page_chunks:
            all_chunks.append(
                Chunk(
                    text=chunk_text, 
                    source=file_path, 
                    page=page_num, 
                    chunk_id=chunk_counter
                )
            )
            chunk_counter += 1


    logger.info("Chunking complete", file_path=file_path, total_chunks=len(all_chunks)
                )
    return all_chunks