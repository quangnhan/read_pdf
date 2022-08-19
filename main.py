from pdf_format import PdfFormat

class PdfParser:
    def __init__(self):
        self.pdf_formats: dict[str, PdfFormat] = {}

    def read_pdf(self, pdf_file: str):
        format_file = self.get_format_file(pdf_file)
        if format_file == '':
            return None
        if format_file not in self.pdf_formats:
            if pdf_format := PdfFormat.load_format_file(format_file):
                self.pdf_formats[format_file] = pdf_format
            else:
                return None
        return self.pdf_formats[format_file].read_text_from_pdf(pdf_file)

    def get_info_from_file(self, pdf_format_file: str, pdf_doc_file: str):
        if pdf_format := PdfFormat.load_format_file(pdf_format_file):
            return pdf_format.read_text_from_pdf(pdf_doc_file)
        else:
            return {}

    def get_format_file(self, file_name: str):
        pdf_format = ''
        if 'Khai Minh' in file_name:
            pdf_format = f'.\KHAI MINH GLOBAL.json'
        return pdf_format


parser = PdfParser()
pdf_file = '.\Khai Minh.pdf'
data = parser.read_pdf(pdf_file)
print(data)