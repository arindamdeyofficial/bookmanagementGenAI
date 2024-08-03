
#region reading pdf
import re
import PyPDF2

class PdfHelper:
    def extract_text_from_pdf(pdf_path):
        """
        Extracts text content from a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            A string containing the extracted text content.
        """
        try:
            with open(pdf_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
                return text
        except FileNotFoundError:
            print(f"Error: PDF file not found at {pdf_path}")
            return None
        except Exception as e:  # Handle broader exceptions
            print(e)

    def clean_text(text):
        """
        Performs basic cleaning on the extracted text.

        Args:
            text: The extracted text content from the PDF.

        Returns:
            A string containing the cleaned text.
        """
        # Replace common non-alphanumeric characters
        cleaned_text = text.replace("\\n", " ").replace("\\t", " ")  # Replace newlines and tabs
        cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", "", cleaned_text)  # Remove non-alphanumeric characters (except space)

        # You can add more cleaning steps here, such as:
        # - Lowercasing all characters
        # - Removing punctuation
        # - Removing stop words

        return cleaned_text

#endregion reading pdf