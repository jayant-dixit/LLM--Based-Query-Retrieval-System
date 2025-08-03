import aiohttp
import tempfile
import fitz

async def download_pdf_from_url(url: str) -> str:
    """
    Downloads a PDF file from a URL and saves it to a temporary file.
    Returns the local file path.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"Failed to download PDF: Status code {resp.status}")
            content = await resp.read()
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file.write(content)
    temp_file.close()
    return temp_file.name

def extract_text_by_page(pdf_path: str) -> list[str]:
    """
    Extracts text from each page of the PDF.
    Returns a list of strings, one per page.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text().strip()
        if text:
            pages.append(text)
    return pages