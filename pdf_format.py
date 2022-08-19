import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy import ndarray

from base_ocr import Rectangle, get_roi, read_pdf, read_text_auto


@dataclass
class DetectionArea:
    _id       : int
    name      : str
    page_index: int
    area      : Rectangle


class PdfFormat:
    """PDF Format files
    - Stored in json format (`*.json`)
    - Created by 'PDF Format Creator App' (in `tools\\pdf_format_creator_app.py`).
    """

    def __init__(self, file_path: str = '', areas: dict[int, DetectionArea] = None):
        self.file_path = file_path
        self.areas = areas if areas is not None else {}

    def add_new_area(self, area_id: int, name: str, page_index: int, x: int, y: int, w: int, h: int):
        rect = Rectangle(x, y, w, h)
        self.areas[area_id] = DetectionArea(area_id, name, page_index, rect)

    @staticmethod
    def load_format_file(file_path: str):
        """Load a PDF format from `*.json` file.\n
        Return a `PdfFormat` instance if success or `None` if fail.
        """
        json_str: str
        with open(file_path, 'r', encoding='utf-8') as f:
            json_str = f.read()
        obj_dict = json.loads(json_str)
        if 'file_path' not in obj_dict or 'areas' not in obj_dict:
            return None
        areas: dict[int, DetectionArea] = {}
        for area_id, detection_area in obj_dict['areas'].items():
            if 'area' not in detection_area:
                return None
            rect = Rectangle(*detection_area['area'])
            name: str = detection_area['name']
            page_index: int = detection_area['page_index']
            area = DetectionArea(area_id, name, page_index, rect)
            areas[area_id] = area
        return PdfFormat(file_path, areas)

    def read_text_in_area(self, bgr_img: ndarray, area_id: int):
        """Read the text in the defined area of an image.\n
        Return the detected text (`str`) if success. Otherwise, return `None` if the id is wrong or error occurs.
        """
        if area_id not in self.areas:
            return None
        dectection_area = self.areas[area_id]
        try:
            roi_img = get_roi(bgr_img, dectection_area.area)
            return read_text_auto(roi_img)
        except Exception as ex:
            return None

    def read_text_from_pdf(self, pdf_file: str):
        texts: dict[str, str] = {}
        pdf_imgs = read_pdf(pdf_file)
        for area_id, detection_area in self.areas.items():
            name = detection_area.name
            img = pdf_imgs[detection_area.page_index]
            text = self.read_text_in_area(img, area_id)
            texts[name] = str(text)
        return texts


class PdfFormats:
    def __init__(self, formats_folder: str):
        """Load all PDF Format files from a folder.
        """
        if not isinstance(formats_folder, Path):
            formats_folder = Path(formats_folder)
        self.formats: dict[str, PdfFormat] = {}
        for f in formats_folder.glob('*.json'):
            if not f.is_file():
                continue
            pdf_format = PdfFormat.load_format_file(str(f))
            if not pdf_format:
                continue
            self.formats[str(f.name).upper()] = pdf_format
        self.format_selector: Callable[..., str] = None

    def read_text_from_pdf(self, pdf_file: Path, *selector_args: Any, **selector_kwargs: Any):
        if not self.format_selector:
            return None
        format_name = self.format_selector(*selector_args, **selector_kwargs)
        pdf_format = self.formats.get(format_name.upper(), None)
        if not pdf_format:
            return None
        return pdf_format.read_text_from_pdf(str(pdf_file))

    def add_pdf_format_selector(self, selector_func: Callable[..., str]):
        self.format_selector = selector_func
